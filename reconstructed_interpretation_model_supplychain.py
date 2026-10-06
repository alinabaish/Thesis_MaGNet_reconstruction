"""Reconstructed MaGNet node + feature interpretation for Supply Chain.

This is a reconstruction of the interpretation procedure, not the original
MaGNet authors' released interpretation implementation.

Input: supplychain_interpretation_checkpoint.pt
Output: node, node-feature, global-feature and edge importance for one test graph.
"""

import argparse
import csv
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool

try:
    from convlayer import MaGNetConv
except ImportError:
    from model.convlayer import MaGNetConv


FEATURE_NAMES = ["production", "sales_order", "delivery", "factory_issue"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_checkpoint(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def clean_state_dict(state):
    if isinstance(state, dict):
        for k in ("state_dict", "model_state_dict", "model"):
            if k in state and isinstance(state[k], dict):
                state = state[k]
                break
    cleaned = {}
    for k, v in state.items():
        cleaned[k[7:] if k.startswith("module.") else k] = v
    return cleaned


class SupplyFirst(nn.Module):
    def __init__(self, num_features=4):
        super().__init__()
        self.conv1 = MaGNetConv(num_features, 64)
        self.conv2 = MaGNetConv(64, 128)
        self.conv3 = MaGNetConv(128, 64)
        self.local_size = 7
        self.lr1 = nn.Linear(7 * 64, 256)
        self.lr2 = nn.Linear(256, 32)
        self.lr3 = nn.Linear(32, 2)


class SupplyMiddle(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MaGNetConv(64, 64)
        self.local_size = 7
        self.lr1 = nn.Linear(7 * 64, 128)
        self.lr2 = nn.Linear(128, 48)
        self.lr3 = nn.Linear(48, 2)


class WeakLast(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(torch.relu(self.fc1(x))))


def make_pool_assignment(num_nodes, local_size, seed, device):
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    required = torch.arange(local_size, device=device)
    extra = torch.randint(0, local_size, (num_nodes - local_size,),
                          generator=g, device=device)
    assignment = torch.cat([required, extra])
    return assignment[torch.randperm(num_nodes, generator=g, device=device)]


def first_forward(model, x, edge_index, pool, node_mask=None):
    if node_mask is not None:
        x = x * node_mask.unsqueeze(1)

    x = model.conv1(x, edge_index)
    if node_mask is not None:
        x = x * node_mask.unsqueeze(1)
    x = F.relu(x)

    x = model.conv2(x, edge_index)
    if node_mask is not None:
        x = x * node_mask.unsqueeze(1)
    x = F.relu(x)

    x = model.conv3(x, edge_index)
    if node_mask is not None:
        x = x * node_mask.unsqueeze(1)
    lap = x

    x = torch.flatten(global_max_pool(x, pool))
    x = F.relu(model.lr1(x))
    x = F.relu(model.lr2(x))
    logits = model.lr3(x)
    return logits, lap


def middle_forward(model, x, edge_index, pool, node_mask=None):
    if node_mask is not None:
        x = x * node_mask.unsqueeze(1)

    x = model.conv1(x, edge_index)
    if node_mask is not None:
        x = x * node_mask.unsqueeze(1)
    lap = x

    x = F.relu(x)
    x = torch.flatten(global_max_pool(x, pool))
    x = F.relu(model.lr1(x))
    x = F.relu(model.lr2(x))
    logits = model.lr3(x)
    return logits, lap


def full_forward(models, classifier, quality, x, edge_index, pools, node_mask=None):
    current = x
    latents = []

    for j, model in enumerate(models):
        if j == 0:
            logits, lap = first_forward(model, current, edge_index, pools[j], node_mask)
        else:
            logits, lap = middle_forward(model, current, edge_index, pools[j], node_mask)
        latents.append(logits)
        current = lap

    final_emb = torch.zeros_like(latents[0])
    for j in range(len(latents)):
        final_emb = final_emb + quality[j] * latents[j]

    prob1 = classifier(final_emb).view(-1)[0]
    probs_final = torch.stack([1.0 - prob1, prob1])
    return probs_final, latents


def build_logical_edges(edge_index):
    logical = []
    lookup = {}
    for k in range(edge_index.size(1)):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        if u == v:
            continue
        pair = (min(u, v), max(u, v))
        if pair not in lookup:
            lookup[pair] = len(logical)
            logical.append(pair)
    mapping = torch.empty(edge_index.size(1), dtype=torch.long, device=edge_index.device)
    for k in range(edge_index.size(1)):
        u, v = int(edge_index[0, k]), int(edge_index[1, k])
        mapping[k] = -1 if u == v else lookup[(min(u, v), max(u, v))]
    return logical, mapping


def concrete_sample(logits, temperature):
    u = torch.rand_like(logits).clamp_(1e-6, 1.0 - 1e-6)
    noise = torch.log(u) - torch.log1p(-u)
    return torch.sigmoid((logits + noise) / temperature)


def interpret_graph(graph_idx, x, true_label, models, classifier, quality, edge_index, pools,
                    stored_prediction, node_names, args, device):
    n, f = x.shape

    with torch.no_grad():
        reconstructed_prob, _ = full_forward(models, classifier, quality, x, edge_index, pools)
    reconstructed_prob = reconstructed_prob.squeeze(0)
    reconstructed_prediction = int(float(reconstructed_prob[1]) >= 0.30)

    # The interpretation target is the original stored MaGNet prediction.
    target = torch.tensor(
        [1.0 - stored_prediction, float(stored_prediction)],
        dtype=x.dtype, device=device
    )

    node_logits = nn.Parameter(torch.zeros(n, device=device))
    feature_logits = nn.Parameter(torch.zeros(n, f, device=device))
    optimizer = torch.optim.Adam([node_logits, feature_logits], lr=args.lr)

    history = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        losses = []
        node_values = []
        feature_values = []

        for _ in range(args.samples):
            nm = concrete_sample(node_logits, args.temperature)
            raw_fm = concrete_sample(feature_logits, args.temperature)
            fm = args.feature_floor + (1.0 - args.feature_floor) * raw_fm
            combined = nm.unsqueeze(1) * fm
            xm = x * combined

            prob, _ = full_forward(models, classifier, quality, xm, edge_index, pools, nm)
            ce = -torch.sum(target * torch.log(prob + 1e-8))
            losses.append(ce)
            node_values.append(nm)
            feature_values.append(fm)

        ce_loss = torch.stack(losses).mean()
        nm_mean = torch.stack(node_values).mean()
        fm_mean = torch.stack(feature_values).mean()
        normalized_feature_density = (fm_mean - args.feature_floor) / (1.0 - args.feature_floor)
        loss = (
            ce_loss
            + args.node_sparsity * nm_mean
            + args.feature_sparsity * normalized_feature_density
        )
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            nm_det = torch.sigmoid(node_logits)
            raw_fm_det = torch.sigmoid(feature_logits)
            fm_det = args.feature_floor + (1.0 - args.feature_floor) * raw_fm_det
            combined = nm_det.unsqueeze(1) * fm_det
            xm = x * combined
            p_det, _ = full_forward(models, classifier, quality, xm, edge_index, pools, nm_det)

        history.append({
            "step": step,
            "loss": float(loss),
            "CE": float(ce_loss),
            "node_mask": float(nm_det.mean()),
            "feature_mask": float(fm_det.mean()),
            "target_prob": float(p_det[stored_prediction]),
        })

    with torch.no_grad():
        node_scores = torch.sigmoid(node_logits)
        raw_feature_scores = torch.sigmoid(feature_logits)
        feature_scores = args.feature_floor + (1.0 - args.feature_floor) * raw_feature_scores
        combined_scores = node_scores.unsqueeze(1) * feature_scores
        final_x = x * combined_scores
        final_prob, _ = full_forward(models, classifier, quality, final_x, edge_index, pools, node_scores)
        final_prob = final_prob.squeeze(0)

    node_order = torch.argsort(node_scores, descending=True).cpu().tolist()
    pair_order = torch.argsort(combined_scores.reshape(-1), descending=True).cpu().tolist()
    feature_global = combined_scores.sum(dim=0)
    feature_order = torch.argsort(feature_global, descending=True).cpu().tolist()
    total = float(feature_global.sum()) + 1e-12
    final_prediction = int(float(final_prob[1]) >= 0.30)

    # Save one compact per-graph summary plus detailed CSVs.
    graph_dir = os.path.join(args.output_dir, f"graph_{graph_idx}")
    os.makedirs(graph_dir, exist_ok=True)

    node_csv = os.path.join(graph_dir, "node_scores.csv")
    pair_csv = os.path.join(graph_dir, "node_feature_scores.csv")
    feat_csv = os.path.join(graph_dir, "global_feature_scores.csv")
    summary_json = os.path.join(graph_dir, "summary.json")

    with open(node_csv, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["rank", "node_id", "product_name", "node_score"])
        for rank, i in enumerate(node_order, 1):
            w.writerow([rank, i, node_names.get(i, f"node_{i}"), float(node_scores[i])])

    with open(pair_csv, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["rank", "node_id", "product_name", "feature_id", "feature_name", "score"])
        for rank, flat in enumerate(pair_order, 1):
            i, j = divmod(flat, f)
            w.writerow([rank, i, node_names.get(i, f"node_{i}"), j, FEATURE_NAMES[j],
                        float(combined_scores[i, j])])

    with open(feat_csv, "w", newline="", encoding="utf-8") as f_out:
        w = csv.writer(f_out)
        w.writerow(["rank", "feature_id", "feature_name", "raw_score", "normalized_score"])
        for rank, j in enumerate(feature_order, 1):
            w.writerow([rank, j, FEATURE_NAMES[j], float(feature_global[j]),
                        float(feature_global[j] / total)])

    summary = {
        "method": "reconstructed MaGNet node + per-node-feature Binary-Concrete mask",
        "version": "v3_all",
        "graph_index_testing": graph_idx,
        "true_label": true_label,
        "num_nodes": n,
        "num_features": f,
        "features": FEATURE_NAMES,
        "quality_weights": quality.detach().cpu().tolist(),
        "stored_checkpoint_prediction": stored_prediction,
        "reconstructed_prediction": reconstructed_prediction,
        "reconstructed_probabilities": reconstructed_prob.detach().cpu().tolist(),
        "final_prediction": final_prediction,
        "final_probabilities": final_prob.detach().cpu().tolist(),
        "threshold": 0.30,
        "steps": args.steps,
        "learning_rate": args.lr,
        "node_sparsity": args.node_sparsity,
        "feature_sparsity": args.feature_sparsity,
        "feature_floor": args.feature_floor,
        "temperature": args.temperature,
        "samples": args.samples,
        "history": history,
    }
    with open(summary_json, "w", encoding="utf-8") as f_out:
        json.dump(summary, f_out, indent=4)

    # Return rows for whole-test-set aggregate files.
    group = "TP" if true_label == 1 and stored_prediction == 1 else \
            "TN" if true_label == 0 and stored_prediction == 0 else \
            "FP" if true_label == 0 and stored_prediction == 1 else "FN"

    node_rows = []
    for rank, i in enumerate(node_order, 1):
        node_rows.append([graph_idx, true_label, stored_prediction, group, rank, i,
                          node_names.get(i, f"node_{i}"), float(node_scores[i])])

    pair_rows = []
    for rank, flat in enumerate(pair_order, 1):
        i, j = divmod(flat, f)
        pair_rows.append([graph_idx, true_label, stored_prediction, group, rank, i,
                          node_names.get(i, f"node_{i}"), j, FEATURE_NAMES[j],
                          float(combined_scores[i, j])])

    feature_rows = []
    for rank, j in enumerate(feature_order, 1):
        feature_rows.append([graph_idx, true_label, stored_prediction, group, rank, j,
                             FEATURE_NAMES[j], float(feature_global[j]),
                             float(feature_global[j] / total)])

    result_row = [graph_idx, int(143 + graph_idx), true_label, stored_prediction, group,
                  reconstructed_prediction, final_prediction,
                  float(reconstructed_prob[1]), float(final_prob[1])]

    return result_row, node_rows, pair_rows, feature_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-graph", type=int, default=0,
                        help="First testing-set graph index (default: 0)")
    parser.add_argument("--end-graph", type=int, default=None,
                        help="Last testing-set graph index, inclusive (default: all graphs)")
    parser.add_argument("--checkpoint", default="supplychain_interpretation_checkpoint.pt")
    parser.add_argument("--mapping", default="product_node_mapping.csv")
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--node-sparsity", type=float, default=0.01)
    parser.add_argument("--feature-sparsity", type=float, default=0.01)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--feature-floor", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pool-seed", type=int, default=123)
    parser.add_argument("--output-dir", default="reconstructed_magnet_supplychain_nodefeature_v3_all")
    args = parser.parse_args()

    if not 0.0 <= args.feature_floor < 1.0:
        raise ValueError("--feature-floor must be in [0, 1).")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load everything shared by the 78 explanations only once.
    set_seed(args.seed)
    ckpt = load_checkpoint(args.checkpoint, device)
    print("Loaded Supply Chain interpretation checkpoint.")

    x_all = ckpt["testing_features"].float().to(device)
    y_all = ckpt["testing_labels"].long().to(device)
    edge_index = ckpt["edge_index"].long().to(device)
    quality = ckpt["quality_vec"].float().view(-1).to(device)
    stored_predictions = ckpt.get("test_predictions", None)
    if stored_predictions is None:
        raise RuntimeError("Checkpoint does not contain test_predictions.")

    num_graphs = x_all.size(0)
    start = args.start_graph
    end = num_graphs - 1 if args.end_graph is None else args.end_graph
    if start < 0 or end >= num_graphs or start > end:
        raise ValueError(f"Graph range must be within 0..{num_graphs-1} and start <= end.")

    n, f = x_all[0].shape
    if f != 4:
        raise RuntimeError(f"Expected 4 node features, found {f}")

    print("Testing graphs:", start, "to", end, "(", end - start + 1, "graphs )")
    print("Number of nodes:", n)
    print("Number of features:", f)
    print("edge_index shape:", tuple(edge_index.shape))
    print("Quality weights:", quality.detach().cpu().numpy())

    logical_edges, _ = build_logical_edges(edge_index)
    print("Logical undirected edges:", len(logical_edges))

    models = [SupplyFirst(4).to(device)]
    for _ in range(5):
        models.append(SupplyMiddle().to(device))

    states = ckpt["magnet_models"]
    if len(states) != 6:
        raise RuntimeError(f"Expected 6 MaGNet components, found {len(states)}")
    for model, state in zip(models, states):
        model.load_state_dict(clean_state_dict(state), strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)

    classifier = WeakLast(
        int(ckpt["classifier_input_size"]),
        int(ckpt["classifier_hidden_size"]),
        int(ckpt["classifier_output_size"]),
    ).to(device)
    classifier.load_state_dict(clean_state_dict(ckpt["final_classifier"]), strict=True)
    classifier.eval()
    for p in classifier.parameters():
        p.requires_grad_(False)

    pools = [make_pool_assignment(n, 7, args.pool_seed + j, device) for j in range(6)]
    print("Fixed pooling assignments created for all six components.")

    node_names = {i: f"node_{i}" for i in range(n)}
    if os.path.exists(args.mapping):
        mapping = pd.read_csv(args.mapping)
        if "product_name" in mapping.columns and "node_id" in mapping.columns:
            node_names = {int(r.node_id): str(r.product_name) for r in mapping.itertuples()}

    os.makedirs(args.output_dir, exist_ok=True)
    all_result_rows = []
    all_node_rows = []
    all_pair_rows = []
    all_feature_rows = []

    for graph_idx in range(start, end + 1):
        # Independent deterministic seed for each graph, so reruns reproduce its explanation.
        set_seed(args.seed + graph_idx)
        x = x_all[graph_idx].clone()
        true_label = int(y_all[graph_idx].item())
        stored_prediction = int(torch.as_tensor(stored_predictions[graph_idx]).item())

        print(f"\n[{graph_idx - start + 1}/{end - start + 1}] "
              f"Graph {graph_idx} (global {int(ckpt['train_end']) + graph_idx}) "
              f"| true={true_label} | prediction={stored_prediction}")

        result_row, node_rows, pair_rows, feature_rows = interpret_graph(
            graph_idx, x, true_label, models, classifier, quality, edge_index, pools,
            stored_prediction, node_names, args, device
        )
        all_result_rows.append(result_row)
        all_node_rows.extend(node_rows)
        all_pair_rows.extend(pair_rows)
        all_feature_rows.extend(feature_rows)

        print("  Explanation complete. Top node:", node_rows[0][6],
              "| top feature:", feature_rows[0][6])

    # Aggregate outputs for thesis-level analysis.
    def write_csv(path, header, rows):
        with open(path, "w", newline="", encoding="utf-8") as f_out:
            w = csv.writer(f_out)
            w.writerow(header)
            w.writerows(rows)

    write_csv(
        os.path.join(args.output_dir, "all_graphs_predictions.csv"),
        ["graph_idx", "global_graph_idx", "true_label", "prediction", "group",
         "reconstructed_prediction", "final_masked_prediction",
         "reconstructed_prob_1", "final_masked_prob_1"],
        all_result_rows,
    )
    write_csv(
        os.path.join(args.output_dir, "all_graphs_node_scores.csv"),
        ["graph_idx", "true_label", "prediction", "group", "rank", "node_id",
         "product_name", "node_score"],
        all_node_rows,
    )
    write_csv(
        os.path.join(args.output_dir, "all_graphs_node_feature_scores.csv"),
        ["graph_idx", "true_label", "prediction", "group", "rank", "node_id",
         "product_name", "feature_id", "feature_name", "score"],
        all_pair_rows,
    )
    write_csv(
        os.path.join(args.output_dir, "all_graphs_feature_scores.csv"),
        ["graph_idx", "true_label", "prediction", "group", "rank", "feature_id",
         "feature_name", "raw_score", "normalized_score"],
        all_feature_rows,
    )

    metadata = {
        "method": "reconstructed MaGNet node + per-node-feature Binary-Concrete mask",
        "version": "v3_all",
        "graphs_interpreted": len(all_result_rows),
        "graph_range": [start, end],
        "test_set_size": num_graphs,
        "threshold": 0.30,
        "steps": args.steps,
        "learning_rate": args.lr,
        "node_sparsity": args.node_sparsity,
        "feature_sparsity": args.feature_sparsity,
        "feature_floor": args.feature_floor,
        "temperature": args.temperature,
        "samples": args.samples,
        "seed": args.seed,
        "pool_seed": args.pool_seed,
        "features": FEATURE_NAMES,
        "num_nodes": n,
        "logical_undirected_edges": len(logical_edges),
        "quality_weights": quality.detach().cpu().tolist(),
    }
    with open(os.path.join(args.output_dir, "run_metadata.json"), "w", encoding="utf-8") as f_out:
        json.dump(metadata, f_out, indent=4)

    print("\n========================================")
    print("ALL-GRAPH INTERPRETATION FINISHED")
    print("========================================")
    print("Graphs interpreted:", len(all_result_rows))
    print("Output directory:", args.output_dir)
    print("Aggregate files:")
    print("  all_graphs_predictions.csv")
    print("  all_graphs_node_scores.csv")
    print("  all_graphs_node_feature_scores.csv")
    print("  all_graphs_feature_scores.csv")
    print("  run_metadata.json")


if __name__ == "__main__":
    main()
