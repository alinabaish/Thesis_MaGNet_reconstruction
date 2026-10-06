"""
Reconstructed MaGNet interpretation - Node + Feature Version 1.

Based directly on reconstructed_magnet_interpretation_nodewise_v3.py.

Main changes from v3:
1. Learns one mask per node.
2. Learns one mask per node-feature pair.
3. Node mask controls node participation and induced edges.
4. Feature mask controls which features of each node are retained.
5. Saves node importance.
6. Saves node-feature importance.
7. Saves optional edge importance.
8. Keeps the same fixed-pooling and six-component-compatible
   interpretation structure used by the current reconstruction.

This is a reconstruction, not a claim that this exact source
was released by the MaGNet authors.
"""

import argparse
import csv
import json
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_max_pool


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# MaGNet convolution
# ============================================================

try:
    from convlayer import MaGNetConv
except ImportError:
    from model.convlayer import MaGNetConv


class WeakerFirst(nn.Module):

    def __init__(self, num_nodes: int, num_features: int):
        super().__init__()

        self.num_nodes = num_nodes
        self.local_size = 7

        self.conv0 = nn.Conv1d(
            num_nodes,
            num_nodes,
            3,
            1,
            padding=1
        )

        self.conv1 = MaGNetConv(num_features, 64)
        self.conv2 = MaGNetConv(64, 128)
        self.conv3 = MaGNetConv(128, 64)

        self.lr1 = nn.Linear(7 * 64, 256)
        self.lr2 = nn.Linear(256, 32)
        self.lr3 = nn.Linear(32, 2)

    def make_pool_assignment(self, device):
        required = torch.arange(
            self.local_size,
            device=device
        )

        extra = torch.randint(
            0,
            self.local_size,
            (self.num_nodes - self.local_size,),
            device=device
        )

        assignment = torch.cat((required, extra))

        return assignment[
            torch.randperm(
                self.num_nodes,
                device=device
            )
        ]

    def forward_with_pool(
        self,
        x,
        edge_index,
        pool_assignment
    ):

        x = self.conv0(x)

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=False)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=False)

        x = self.conv3(x, edge_index)

        lap = x

        x = global_max_pool(
            x,
            pool_assignment
        )

        x = torch.flatten(x)

        x = F.relu(self.lr1(x))
        x = F.relu(self.lr2(x))
        x = self.lr3(x)

        return (
            F.log_softmax(
                x.reshape(1, 2),
                dim=1
            ),
            lap,
            x
        )

    def forward(self, x, edge_index):

        pool_assignment = self.make_pool_assignment(
            x.device
        )

        return self.forward_with_pool(
            x,
            edge_index,
            pool_assignment
        )


# ============================================================
# Loading
# ============================================================

def torch_load(path, device):

    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=False
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device
        )


def clean_state_dict(state):

    if isinstance(state, dict):

        for key in (
            "state_dict",
            "model_state_dict",
            "model"
        ):

            value = state.get(key)

            if isinstance(value, dict):
                state = value
                break

    if not isinstance(state, dict):
        raise ValueError(
            "Checkpoint does not contain a state dictionary."
        )

    cleaned = {}

    for key, value in state.items():

        if key.startswith("module."):
            key = key[7:]

        cleaned[key] = value

    return cleaned


def load_model(
    path,
    num_nodes,
    num_features,
    device
):

    model = WeakerFirst(
        num_nodes,
        num_features
    ).to(device)

    state = clean_state_dict(
        torch_load(path, device)
    )

    missing, unexpected = model.load_state_dict(
        state,
        strict=False
    )

    if missing or unexpected:

        raise RuntimeError(
            "Checkpoint/model mismatch.\n"
            f"Missing keys: {missing}\n"
            f"Unexpected keys: {unexpected}"
        )

    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def load_data(path, device):

    data = torch_load(
        path,
        device
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Expected explanation data to be a dictionary."
        )

    return data


def get_graph(
    data,
    graph_index,
    device
):

    testing = data["original_testing"]

    graph = torch.as_tensor(
        testing[graph_index],
        dtype=torch.float32,
        device=device
    )

    # Setting 1 stores original_testing as:
    # [features, nodes] = [10, 75]
    #
    # MaGNet uses:
    # [nodes, features] = [75, 10]

    if graph.shape[0] < graph.shape[1]:

        x = graph.T.contiguous()

    else:

        x = graph.contiguous()

    num_nodes = int(x.shape[0])
    num_features = int(x.shape[1])

    edge_index = torch.as_tensor(
        data["edge_index"],
        dtype=torch.long,
        device=device
    )

    return (
        x,
        edge_index,
        num_nodes,
        num_features
    )


# ============================================================
# Logical undirected edges
# ============================================================

def build_logical_edges(edge_index):

    pairs = []
    lookup = {}

    for k in range(edge_index.shape[1]):

        u = int(edge_index[0, k])
        v = int(edge_index[1, k])

        if u == v:
            continue

        pair = (
            min(u, v),
            max(u, v)
        )

        if pair not in lookup:

            lookup[pair] = len(pairs)
            pairs.append(pair)

    return pairs


# ============================================================
# Weighted MaGNet message passing
# ============================================================

def add_self_loops(
    edge_index,
    num_nodes
):

    loops = torch.arange(
        num_nodes,
        device=edge_index.device,
        dtype=edge_index.dtype
    )

    loop_index = torch.stack(
        (loops, loops),
        dim=0
    )

    return torch.cat(
        (
            edge_index,
            loop_index
        ),
        dim=1
    )


def weighted_magnet_conv(
    conv,
    x,
    edge_index,
    directed_edge_weight
):

    x = conv.lin(x)

    num_nodes = x.size(0)

    edge_with_loops = add_self_loops(
        edge_index,
        num_nodes
    )

    loop_weight = torch.ones(
        num_nodes,
        dtype=directed_edge_weight.dtype,
        device=directed_edge_weight.device
    )

    full_weight = torch.cat(
        (
            directed_edge_weight,
            loop_weight
        ),
        dim=0
    )

    src = edge_with_loops[0]
    dst = edge_with_loops[1]

    messages = x[src]

    messages = (
        messages
        * full_weight.unsqueeze(-1)
    )

    out = torch.zeros_like(x)

    out.index_add_(
        0,
        dst,
        messages
    )

    return out


# ============================================================
# Differentiable forward
# ============================================================

def interpretation_forward(
    model,
    x,
    edge_index,
    directed_edge_weight,
    pool_assignment
):

    x = model.conv0(x)

    x = weighted_magnet_conv(
        model.conv1,
        x,
        edge_index,
        directed_edge_weight
    )

    x = F.relu(x)

    x = weighted_magnet_conv(
        model.conv2,
        x,
        edge_index,
        directed_edge_weight
    )

    x = F.relu(x)

    x = weighted_magnet_conv(
        model.conv3,
        x,
        edge_index,
        directed_edge_weight
    )

    lap = x

    x = global_max_pool(
        x,
        pool_assignment
    )

    x = torch.flatten(x)

    x = F.relu(
        model.lr1(x)
    )

    x = F.relu(
        model.lr2(x)
    )

    x = model.lr3(x)

    return (
        F.softmax(
            x.reshape(1, 2),
            dim=1
        ),
        lap,
        x
    )


# ============================================================
# Concrete / Gumbel-sigmoid
# ============================================================

def concrete_sample(
    logits,
    temperature
):

    u = torch.rand_like(
        logits
    ).clamp_(
        1e-6,
        1.0 - 1e-6
    )

    logistic_noise = (
        torch.log(u)
        - torch.log1p(-u)
    )

    return torch.sigmoid(
        (logits + logistic_noise)
        / temperature
    )


# ============================================================
# Node + Feature Interpreter
# ============================================================

class NodeFeatureInterpreter:

    def __init__(
        self,
        model,
        edge_index,
        pool_assignment,
        steps=800,
        lr=0.03,
        node_sparsity=0.01,
        feature_sparsity=0.01,
        temperature=0.5,
        samples=4
    ):

        self.model = model
        self.edge_index = edge_index
        self.pool_assignment = pool_assignment

        self.steps = steps
        self.lr = lr

        self.node_sparsity = node_sparsity
        self.feature_sparsity = feature_sparsity

        self.temperature = temperature
        self.samples = samples

        print(
            f"Node sparsity = {node_sparsity}"
        )

        print(
            f"Feature sparsity = {feature_sparsity}"
        )

    # --------------------------------------------------------
    # Node mask -> edge mask
    # --------------------------------------------------------

    def node_to_edge_mask(
        self,
        node_mask
    ):

        src = self.edge_index[0]
        dst = self.edge_index[1]

        return (
            node_mask[src]
            * node_mask[dst]
        )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    def optimize(self, x):

        full_weights = torch.ones(
            self.edge_index.shape[1],
            dtype=x.dtype,
            device=x.device
        )

        # ----------------------------------------------------
        # Original prediction
        # ----------------------------------------------------

        with torch.no_grad():

            original_prob, _, _ = (
                interpretation_forward(
                    self.model,
                    x,
                    self.edge_index,
                    full_weights,
                    self.pool_assignment
                )
            )

            original_prob = (
                original_prob.squeeze(0)
            )

            target_class = int(
                torch.argmax(
                    original_prob
                ).item()
            )

        # ----------------------------------------------------
        # Learnable parameters
        # ----------------------------------------------------

        num_nodes = x.shape[0]
        num_features = x.shape[1]

        node_logits = nn.Parameter(
            torch.zeros(
                num_nodes,
                dtype=x.dtype,
                device=x.device
            )
        )

        feature_logits = nn.Parameter(
            torch.zeros(
                num_nodes,
                num_features,
                dtype=x.dtype,
                device=x.device
            )
        )

        optimizer = torch.optim.Adam(
            [
                node_logits,
                feature_logits
            ],
            lr=self.lr
        )

        history = []

        # ----------------------------------------------------
        # Optimization loop
        # ----------------------------------------------------

        for step in range(
            1,
            self.steps + 1
        ):

            optimizer.zero_grad()

            ce_values = []
            node_mask_values = []
            feature_mask_values = []

            for _ in range(
                self.samples
            ):

                # --------------------------------------------
                # Node mask
                # --------------------------------------------

                node_mask = concrete_sample(
                    node_logits,
                    self.temperature
                )

                # --------------------------------------------
                # Feature mask
                # --------------------------------------------

                feature_mask = concrete_sample(
                    feature_logits,
                    self.temperature
                )

                # --------------------------------------------
                # Combined node-feature mask
                # --------------------------------------------

                combined_mask = (
                    node_mask.unsqueeze(-1)
                    * feature_mask
                )

                # --------------------------------------------
                # Mask node features
                # --------------------------------------------

                x_masked = (
                    x
                    * combined_mask
                )

                # --------------------------------------------
                # Mask graph edges using node mask
                # --------------------------------------------

                edge_weight = (
                    self.node_to_edge_mask(
                        node_mask
                    )
                )

                # --------------------------------------------
                # Forward
                # --------------------------------------------

                masked_prob, _, _ = (
                    interpretation_forward(
                        self.model,
                        x_masked,
                        self.edge_index,
                        edge_weight,
                        self.pool_assignment
                    )
                )

                masked_prob = (
                    masked_prob.squeeze(0)
                )

                # --------------------------------------------
                # Cross entropy
                # --------------------------------------------

                ce = -torch.sum(
                    original_prob.detach()
                    * torch.log(
                        masked_prob + 1e-8
                    )
                )

                ce_values.append(ce)

                node_mask_values.append(
                    node_mask
                )

                feature_mask_values.append(
                    feature_mask
                )

            # ------------------------------------------------
            # Average samples
            # ------------------------------------------------

            ce_loss = torch.stack(
                ce_values
            ).mean()

            sampled_node_mask = (
                torch.stack(
                    node_mask_values
                ).mean(dim=0)
            )

            sampled_feature_mask = (
                torch.stack(
                    feature_mask_values
                ).mean(dim=0)
            )

            # ------------------------------------------------
            # Sparsity
            # ------------------------------------------------

            node_sparsity_loss = (
                sampled_node_mask.mean()
            )

            feature_sparsity_loss = (
                sampled_feature_mask.mean()
            )

            sparsity_loss = (
                self.node_sparsity
                * node_sparsity_loss
                +
                self.feature_sparsity
                * feature_sparsity_loss
            )

            loss = (
                ce_loss
                + sparsity_loss
            )

            loss.backward()

            optimizer.step()

            # ------------------------------------------------
            # Deterministic evaluation
            # ------------------------------------------------

            with torch.no_grad():

                deterministic_node_mask = (
                    torch.sigmoid(
                        node_logits
                    )
                )

                deterministic_feature_mask = (
                    torch.sigmoid(
                        feature_logits
                    )
                )

                deterministic_combined_mask = (
                    deterministic_node_mask.unsqueeze(-1)
                    * deterministic_feature_mask
                )

                deterministic_edges = (
                    self.node_to_edge_mask(
                        deterministic_node_mask
                    )
                )

                deterministic_x = (
                    x
                    * deterministic_combined_mask
                )

                deterministic_prob, _, _ = (
                    interpretation_forward(
                        self.model,
                        deterministic_x,
                        self.edge_index,
                        deterministic_edges,
                        self.pool_assignment
                    )
                )

                deterministic_prob = (
                    deterministic_prob.squeeze(0)
                )

            history.append(
                {
                    "step": step,
                    "loss": float(
                        loss.item()
                    ),
                    "CE": float(
                        ce_loss.item()
                    ),
                    "node_sparsity_loss": float(
                        node_sparsity_loss.item()
                    ),
                    "feature_sparsity_loss": float(
                        feature_sparsity_loss.item()
                    ),
                    "mean_node_mask": float(
                        deterministic_node_mask.mean().item()
                    ),
                    "mean_feature_mask": float(
                        deterministic_feature_mask.mean().item()
                    ),
                    "target_prob": float(
                        deterministic_prob[
                            target_class
                        ].item()
                    )
                }
            )

            if (
                step == 1
                or step % 50 == 0
                or step == self.steps
            ):

                print(
                    f"Step {step:4d}/{self.steps} | "
                    f"loss={loss.item():.6f} | "
                    f"CE={ce_loss.item():.6f} | "
                    f"node_mask="
                    f"{deterministic_node_mask.mean().item():.6f} | "
                    f"feature_mask="
                    f"{deterministic_feature_mask.mean().item():.6f} | "
                    f"target_prob="
                    f"{deterministic_prob[target_class].item():.6f}"
                )

        # ----------------------------------------------------
        # Final masks
        # ----------------------------------------------------

        with torch.no_grad():

            final_node_mask = torch.sigmoid(
                node_logits
            )

            final_feature_mask = torch.sigmoid(
                feature_logits
            )

            final_combined_mask = (
                final_node_mask.unsqueeze(-1)
                * final_feature_mask
            )

            final_edges = (
                self.node_to_edge_mask(
                    final_node_mask
                )
            )

            final_x = (
                x
                * final_combined_mask
            )

            final_prob, _, _ = (
                interpretation_forward(
                    self.model,
                    final_x,
                    self.edge_index,
                    final_edges,
                    self.pool_assignment
                )
            )

            final_prob = (
                final_prob.squeeze(0)
            )

            final_prediction = int(
                torch.argmax(
                    final_prob
                ).item()
            )

        return {
            "target_class": target_class,

            "original_probability":
                original_prob.detach(),

            "original_prediction":
                target_class,

            "node_mask":
                final_node_mask.detach(),

            "feature_mask":
                final_feature_mask.detach(),

            "combined_mask":
                final_combined_mask.detach(),

            "directed_edge_mask":
                final_edges.detach(),

            "final_probability":
                final_prob.detach(),

            "final_prediction":
                final_prediction,

            "history":
                history
        }


# ============================================================
# Ranking helpers
# ============================================================

def ranked_nodes(scores):

    values = (
        scores
        .detach()
        .cpu()
        .numpy()
    )

    return sorted(
        [
            (
                i,
                float(v)
            )
            for i, v in enumerate(values)
        ],
        key=lambda z: z[1],
        reverse=True
    )


def ranked_features(
    feature_scores
):

    values = (
        feature_scores
        .detach()
        .cpu()
        .numpy()
    )

    return sorted(
        [
            (
                i,
                float(v)
            )
            for i, v in enumerate(values)
        ],
        key=lambda z: z[1],
        reverse=True
    )


# ============================================================
# Save node scores
# ============================================================

def save_node_csv(
    path,
    node_scores,
    ground_truth
):

    ranking = ranked_nodes(
        node_scores
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "rank",
                "node",
                "importance_score",
                "is_true_important"
            ]
        )

        for rank, (
            node,
            score
        ) in enumerate(
            ranking,
            1
        ):

            writer.writerow(
                [
                    rank,
                    node,
                    f"{score:.10f}",
                    (
                        "yes"
                        if node in ground_truth
                        else "no"
                    )
                ]
            )


# ============================================================
# Save node-feature scores
# ============================================================

def save_node_feature_csv(
    path,
    combined_mask
):

    values = (
        combined_mask
        .detach()
        .cpu()
        .numpy()
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        num_nodes = values.shape[0]
        num_features = values.shape[1]

        header = [
            "node"
        ]

        for feature in range(
            num_features
        ):
            header.append(
                f"feature_{feature}"
            )

        header.append(
            "node_importance_max"
        )

        header.append(
            "node_importance_mean"
        )

        writer.writerow(
            header
        )

        for node in range(
            num_nodes
        ):

            feature_values = (
                values[node]
            )

            writer.writerow(
                [
                    node,
                    *[
                        f"{v:.10f}"
                        for v in feature_values
                    ],
                    f"{feature_values.max():.10f}",
                    f"{feature_values.mean():.10f}"
                ]
            )


# ============================================================
# Save global feature scores
# ============================================================

def save_global_feature_csv(
    path,
    combined_mask
):

    values = (
        combined_mask
        .detach()
        .cpu()
        .numpy()
    )

    global_scores = (
        values.mean(axis=0)
    )

    ranking = sorted(
        [
            (
                feature,
                float(score)
            )
            for feature, score
            in enumerate(
                global_scores
            )
        ],
        key=lambda z: z[1],
        reverse=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "rank",
                "feature",
                "importance_score"
            ]
        )

        for rank, (
            feature,
            score
        ) in enumerate(
            ranking,
            1
        ):

            writer.writerow(
                [
                    rank,
                    feature,
                    f"{score:.10f}"
                ]
            )


# ============================================================
# Save edge scores
# ============================================================

def save_edge_csv(
    path,
    edge_index,
    node_scores
):

    src = (
        edge_index[0]
        .detach()
        .cpu()
        .numpy()
    )

    dst = (
        edge_index[1]
        .detach()
        .cpu()
        .numpy()
    )

    scores = (
        node_scores
        .detach()
        .cpu()
        .numpy()
    )

    rows = []
    seen = set()

    for u, v in zip(
        src,
        dst
    ):

        u = int(u)
        v = int(v)

        if u == v:
            continue

        pair = (
            min(u, v),
            max(u, v)
        )

        if pair in seen:
            continue

        seen.add(pair)

        score = (
            scores[pair[0]]
            * scores[pair[1]]
        )

        rows.append(
            (
                pair[0],
                pair[1],
                float(score)
            )
        )

    rows.sort(
        key=lambda row: row[2],
        reverse=True
    )

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(
            [
                "rank",
                "node_u",
                "node_v",
                "importance_score"
            ]
        )

        for rank, (
            u,
            v,
            score
        ) in enumerate(
            rows,
            1
        ):

            writer.writerow(
                [
                    rank,
                    u,
                    v,
                    f"{score:.10f}"
                ]
            )


# ============================================================
# Fixed pooling
# ============================================================

def make_fixed_pool_for_script(
    num_nodes,
    local_size,
    seed,
    device
):

    generator = torch.Generator(
        device=device
    )

    generator.manual_seed(
        seed
    )

    required = torch.arange(
        local_size,
        device=device
    )

    extra = torch.randint(
        0,
        local_size,
        (
            num_nodes
            - local_size,
        ),
        generator=generator,
        device=device
    )

    assignment = torch.cat(
        (
            required,
            extra
        )
    )

    permutation = torch.randperm(
        num_nodes,
        generator=generator,
        device=device
    )

    return assignment[
        permutation
    ]


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        default="magnet_explanation_data.pt"
    )

    parser.add_argument(
        "--checkpoint",
        default="magnet_component_0.pt"
    )

    parser.add_argument(
        "--graph",
        type=int,
        default=None,
        help="Interpret one testing graph only; omit to interpret all testing graphs."
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=800
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=0.03
    )

    parser.add_argument(
        "--node-sparsity",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--feature-sparsity",
        type=float,
        default=0.01
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.5
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=4
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--pool-seed",
        type=int,
        default=123
    )

    parser.add_argument(
        "--output-dir",
        default="reconstructed_magnet_nodefeature_v1"
    )

    args = parser.parse_args()

    set_seed(
        args.seed
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "Using device:",
        device
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    data = load_data(
        args.data,
        device
    )

    print(
        "Loaded explanation data."
    )

    print(
        "Available keys:",
        list(data.keys())
    )

    # --------------------------------------------------------
    # Determine which testing graphs to interpret
    # --------------------------------------------------------

    original_testing = data["original_testing"]
    num_testing_graphs = len(original_testing)

    if args.graph is None:
        graph_indices = list(range(num_testing_graphs))
    else:
        if not (0 <= args.graph < num_testing_graphs):
            raise IndexError(
                f"graph index {args.graph} is outside the testing set "
                f"[0, {num_testing_graphs - 1}]."
            )
        graph_indices = [args.graph]

    print("Testing graphs available:", num_testing_graphs)
    print("Graphs to interpret:", len(graph_indices))

    # Load model architecture once from the first testing graph.
    _, _, model_num_nodes, model_num_features = get_graph(
        data, graph_indices[0], device
    )

    model = load_model(
        args.checkpoint,
        model_num_nodes,
        model_num_features,
        device
    )

    print(
        "Loaded MaGNet checkpoint:",
        args.checkpoint
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # Accumulate recovery metrics across graphs.
    all_recovery = {k: [] for k in (10, 20, 30, 40, 50) if k <= model_num_nodes}
    all_summaries = []

    def process_graph(graph_index):
        (
            x,
            edge_index,
            num_nodes,
            num_features
        ) = get_graph(
            data,
            graph_index,
            device
        )

        print("\n" + "=" * 60)
        print(f"INTERPRETING SETTING-1 TEST GRAPH {graph_index + 1}/{num_testing_graphs} (index {graph_index})")
        print("=" * 60)
        print("Number of nodes:", num_nodes)
        print("Number of features:", num_features)
        print("X shape:", tuple(x.shape))
        print("edge_index shape:", tuple(edge_index.shape))

        logical_edges = build_logical_edges(edge_index)
        print("Logical undirected edges:", len(logical_edges))

        # The checkpoint architecture is fixed for Setting 1.
        pool_assignment = make_fixed_pool_for_script(
            num_nodes, model.local_size, args.pool_seed, device
        )

        full_weights = torch.ones(
            edge_index.shape[1], dtype=x.dtype, device=device
        )

        with torch.no_grad():
            original_prob, _, _ = interpretation_forward(
                model, x, edge_index, full_weights, pool_assignment
            )
            original_prob = original_prob.squeeze(0)
            original_prediction = int(torch.argmax(original_prob).item())

        print("Original prediction:", original_prediction)
        print("Original probabilities:", original_prob.cpu().numpy())

        print("Starting Node + Feature interpretation...")

        interpreter = NodeFeatureInterpreter(
            model=model, edge_index=edge_index, pool_assignment=pool_assignment,
            steps=args.steps, lr=args.lr,
            node_sparsity=args.node_sparsity,
            feature_sparsity=args.feature_sparsity,
            temperature=args.temperature, samples=args.samples
        )

        result = interpreter.optimize(x)
        node_scores = result["node_mask"]
        node_ranking = ranked_nodes(node_scores)
        combined_scores = result["combined_mask"]

        ground_truth = set(range(min(20, num_nodes)))

        print("\nTOP 20 NODES")
        for rank, (node, score) in enumerate(node_ranking[:20], 1):
            marker = " *" if node in ground_truth else ""
            print(f"{rank:3d}. node {node:2d} score={score:.9f}{marker}")
        print("* = Setting-1 ground-truth important node")

        global_feature_scores = combined_scores.mean(dim=0)
        feature_ranking = ranked_features(global_feature_scores)
        print("\nGLOBAL FEATURE IMPORTANCE")
        for rank, (feature, score) in enumerate(feature_ranking, 1):
            print(f"{rank:3d}. feature {feature:2d} score={score:.9f}")

        feature_matrix = combined_scores.detach().cpu().numpy()
        node_feature_pairs = [
            (node, feature, float(feature_matrix[node, feature]))
            for node in range(num_nodes)
            for feature in range(num_features)
        ]
        node_feature_pairs.sort(key=lambda z: z[2], reverse=True)
        print("\nTOP NODE-FEATURE PAIRS")
        for rank, (node, feature, score) in enumerate(node_feature_pairs[:20], 1):
            print(f"{rank:3d}. node={node:2d}, feature={feature:2d}, score={score:.9f}")

        print("\nFINAL MASKED GRAPH")
        print("Prediction:", result["final_prediction"])
        print("Probabilities:", result["final_probability"].cpu().numpy())

        node_path = os.path.join(args.output_dir, f"graph_{graph_index}_node_scores.csv")
        node_feature_path = os.path.join(args.output_dir, f"graph_{graph_index}_node_feature_scores.csv")
        global_feature_path = os.path.join(args.output_dir, f"graph_{graph_index}_global_feature_scores.csv")
        edge_path = os.path.join(args.output_dir, f"graph_{graph_index}_edge_scores.csv")
        summary_path = os.path.join(args.output_dir, f"graph_{graph_index}_summary.json")

        save_node_csv(node_path, node_scores, ground_truth)
        save_node_feature_csv(node_feature_path, combined_scores)
        save_global_feature_csv(global_feature_path, combined_scores)
        save_edge_csv(edge_path, edge_index, node_scores)

        recovery_rows = []
        for k in (10, 20, 30, 40, 50):
            if k > num_nodes:
                continue
            selected = {node for node, _ in node_ranking[:k]}
            recovered = len(selected.intersection(ground_truth))
            recall = recovered / len(ground_truth)
            precision = recovered / k
            row = {
                "top_k": k, "recovered": recovered,
                "total_important": len(ground_truth),
                "recall": recall, "precision": precision
            }
            recovery_rows.append(row)
            all_recovery.setdefault(k, []).append(row)

        summary = {
            "method": "node-feature induced-subgraph Binary-Concrete mask",
            "description": "Reconstruction based on v3 with separate node and node-feature masks.",
            "graph_index": graph_index,
            "num_nodes": num_nodes, "num_features": num_features,
            "directed_edges": int(edge_index.shape[1]),
            "logical_undirected_edges": len(logical_edges),
            "checkpoint": args.checkpoint, "seed": args.seed, "pool_seed": args.pool_seed,
            "steps": args.steps, "learning_rate": args.lr,
            "node_sparsity": args.node_sparsity, "feature_sparsity": args.feature_sparsity,
            "temperature": args.temperature, "samples_per_step": args.samples,
            "original_prediction": original_prediction,
            "original_probabilities": original_prob.cpu().tolist(),
            "final_prediction": result["final_prediction"],
            "final_probabilities": result["final_probability"].cpu().tolist(),
            "mean_node_importance": float(node_scores.mean().item()),
            "max_node_importance": float(node_scores.max().item()),
            "mean_feature_importance": float(result["feature_mask"].mean().item()),
            "recovery": recovery_rows,
            "important_nodes_used_only_for_evaluation": sorted(ground_truth),
            "history": result["history"]
        }

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)

        all_summaries.append(summary)
        return summary

    for graph_index in graph_indices:
        process_graph(graph_index)

    # --------------------------------------------------------
    # Aggregate results across all interpreted graphs
    # --------------------------------------------------------
    aggregate = {
        "num_graphs": len(all_summaries),
        "graph_indices": graph_indices,
        "mean_precision": {},
        "mean_recall": {},
        "mean_recovered": {}
    }

    print("\n" + "=" * 60)
    print("OVERALL SETTING-1 INTERPRETATION RESULTS")
    print("=" * 60)
    print("Graphs interpreted:", len(all_summaries))

    for k in sorted(all_recovery):
        rows = all_recovery[k]
        if not rows:
            continue
        mean_p = float(np.mean([r["precision"] for r in rows]))
        mean_r = float(np.mean([r["recall"] for r in rows]))
        mean_rec = float(np.mean([r["recovered"] for r in rows]))
        aggregate["mean_precision"][str(k)] = mean_p
        aggregate["mean_recall"][str(k)] = mean_r
        aggregate["mean_recovered"][str(k)] = mean_rec
        print(f"Top-{k}: mean recovered={mean_rec:.4f}, mean precision={mean_p:.4f}, mean recall={mean_r:.4f}")

    aggregate_path = os.path.join(args.output_dir, "all_graphs_summary.json")
    with open(aggregate_path, "w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=4)

    print("\nAggregate results saved:", aggregate_path)
    print("Node + feature interpretation finished for all selected graphs.")


if __name__ == "__main__":
    main()