#!/usr/bin/env python
# coding: utf-8


import json
import torch
import numpy as np 
import os
import random 
from torch_geometric.data import Data
from torch import nn
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import inspect
from typing import Any, Dict, Optional, Callable, Optional
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.utils import to_dense_batch
from torch_geometric.nn import GCNConv,  GATConv, APPNP, TransformerConv
from torch.nn import Linear, Sequential, BatchNorm1d, Dropout
from convlayer import MaGNetConv
from pooling import global_max_pool
from norm_operation import PairNorm, MeanSubtractionNorm
from gpslayer import * 
import warnings

import torch
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

print(torch.cuda.is_available())


def realdata_analysis():

    # Use CPU when CUDA is unavailable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Compuate the output the probability logits, loss value, and feature embeddings
    def compute_loss(dataset):
        out = []
        feat = []
        emb = [] 
        for i in range(dataset.shape[0]):
            # extract the node features from dataset
            nodes_fea = dataset[i]
            nodes_fea = np.asarray(nodes_fea, dtype=np.float32)

            x = torch.tensor(nodes_fea, dtype=torch.float)
            
            # contrust computational graph
            data = Data(x=x, edge_index=edge_index)
            data = data.to(device)

            # model inference 
            output_model = model(data.x, data.edge_index)
            tmpout = output_model[0]

            # compute the model output for loss value, logits, and embeddings
            tmpfeat = output_model[1].cpu().detach().numpy()
            tmpemb = output_model[2].cpu().detach().numpy()
            out.append(tmpout)
            feat.append(tmpfeat)
            emb.append(tmpemb)
            
        # summarize the total probability logtis
        soft_prob = torch.concat(out)
        return [soft_prob, feat, emb]

    # Evaluate the model performance in terms of datasets
    def model_acc(dataset, setname):
        pred = []
        for i in range(dataset.shape[0]):
            # extract the node features from dataset
            nodes_fea = dataset[i]
            x = torch.tensor(nodes_fea, dtype=torch.float)
            
            # contrust computational graph
            data = Data(x=x, edge_index=edge_index)
            data = data.to(device)

            # predict the graph labels
            pred.append(model(data.x, data.edge_index)[0].argmax(dim=1)[0])
            
        # send the results to computational device
        pred = torch.Tensor(pred).to(device)

        # evaluate the model performance on various datasets
        if setname == "testing":
            correct = (pred == testing_label).sum()
        elif setname == "training": 
            correct = (pred == training_label).sum()


        # compute the accuracy
        acc = int(correct) / dataset.shape[0]
        return acc    

    # Compute the weighted loss error rate
    def weight_loss(dataset, weights):
        pred = []

        # model prediction on graph labels
        for i in range(dataset.shape[0]):
            nodes_fea = dataset[i]
            x = torch.tensor(nodes_fea, dtype=torch.float)
            #data = Data(x=x, edge_index=sub_edge_index)
            data = Data(x=x, edge_index=edge_index)
            data = data.to(device)
            pred.append(model(data.x, data.edge_index)[0].argmax(dim=1)[0])
        pred = torch.Tensor(pred).to(device)
        
        # compute the weighted error rate for classifier 
        err_rate = (weights.to(device)*(pred != training_label)).sum()/weights.sum()

        return err_rate

    # Evalute the stabilized quality of the classifier 
    def quality_update(err_rate):
        
        #negative logit function
        alpha = torch.log(0.01+err_rate/(1-err_rate))
        return torch.max(-alpha,torch.tensor(0.05))

    # Update the weights for classifiers 
    def weight_update(err_rate, dataset, weights):
        alpha = quality_update(err_rate)

        for i in range(dataset.shape[0]):
            nodes_fea = dataset[i]
            x = torch.tensor(nodes_fea, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)
            data = data.to(device)

            # model prediction and weights updating via exponential rule
            curr_pred = torch.Tensor(model(data.x, data.edge_index)[0].argmax(dim=1)[0]).to(device) 
            weights[i] = torch.exp(alpha*(training_label[i] != curr_pred) )

        # normalize the weights for all samples
        weights = nn.functional.normalize(weights,p=2, dim=0)
            
        return weights

    # Evaluate the final model performance
    def test_acc(model):
    
        model.eval()
    
        with torch.no_grad():
    
            outputs = model(test_emb)
    
            predictions = (
                (outputs > 0.3)
                .long()
                .view(-1)
            )
    
        accuracy = accuracy_score(
            testing_label.cpu().numpy(),
            predictions.cpu().numpy()
        )
    
        return accuracy


    # ============================================================
    # IMPORT SUPPLY CHAIN DATA
    # ============================================================
    
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    features_path = os.path.join(
        BASE_DIR,
        "supplychain_features.csv"
    )
    
    labels_path = os.path.join(
        BASE_DIR,
        "supplychain_labels.csv"
    )
    
    edges_path = os.path.join(
        BASE_DIR,
        "supplychain_edges.csv"
    )
    
    mapping_path = os.path.join(
        BASE_DIR,
        "product_node_mapping.csv"
    )
    
    
    features_df = pd.read_csv(features_path)
    labels_df = pd.read_csv(labels_path)
    edges_df = pd.read_csv(edges_path)
    mapping_df = pd.read_csv(mapping_path)
    
    
    feature_arrays = []


    for node_id in range(41):
        node_features = features_df[
            [
                f"product_{node_id}_production",
                f"product_{node_id}_sales_order",
                f"product_{node_id}_delivery",
                f"product_{node_id}_factory_issue"
            ]
        ].to_numpy(dtype=np.float32)
    
        feature_arrays.append(node_features)
    
    # Construct the complete original dataset only once
    original_features = np.stack(
        feature_arrays,
        axis=1
    )
    
    # Keep dataset as the working copy
    dataset = original_features.copy()
        
    src = edges_df["source"].to_numpy(dtype=np.int64)
    dst = edges_df["target"].to_numpy(dtype=np.int64)
    
    
    edge_index = torch.tensor(
        np.vstack([
            np.concatenate([src, dst]),
            np.concatenate([dst, src])
        ]),
        dtype=torch.long
    )
    
    print("\n===== SUPPLY CHAIN GRAPH =====")
    print("Number of nodes:", len(mapping_df))
    print("Unique edges:", len(edges_df))
    print("edge_index shape:", edge_index.shape)
    print("Dataset shape:", dataset.shape)
    print("Labels shape:", labels_df.shape)
    
    feature_dates = pd.to_datetime(features_df["Date"])
    label_dates = pd.to_datetime(labels_df["date"])
    
    assert feature_dates.equals(label_dates)
    
    labels = labels_df["label"].to_numpy(dtype=np.int64)
    
    assert dataset.shape == (221, 41, 4)
    assert len(labels) == 221
    assert len(dataset) == len(labels)
    original_labels = labels.copy()
    print("Original labels length:", len(original_labels))
    
    # ============================================================
    # SAVE ORIGINAL CONSTRUCTED DATASET
    # ============================================================
    
    original_rows = []
    
    for graph_index in range(len(original_features)):
        for node_id in range(original_features.shape[1]):
    
            original_rows.append({
                "graph_index": graph_index,
                "date": features_df.iloc[graph_index]["Date"],
                "node_id": node_id,
                "production": original_features[graph_index, node_id, 0],
                "sales_order": original_features[graph_index, node_id, 1],
                "delivery": original_features[graph_index, node_id, 2],
                "factory_issue": original_features[graph_index, node_id, 3],
                "label": labels[graph_index]
            })
    
    original_dataset_df = pd.DataFrame(original_rows)
    
    original_dataset_df.to_csv(
        "supplychain_original_dataset.csv",
        index=False
    )
    
    print(
        "Saved original dataset:",
        original_dataset_df.shape
    )
        
    
    nodes_fea = dataset[0]
    
    x = torch.tensor(
        nodes_fea,
        dtype=torch.float
    )
    
    data = Data(
        x=x,
        edge_index=edge_index
    )
    
    train_end = int(0.65 * len(dataset))
    
    training = dataset[:train_end]
    testing = dataset[train_end:]
    
    training_label = torch.tensor(
        labels[:train_end],
        dtype=torch.long
    ).to(device)
    
    testing_label = torch.tensor(
        labels[train_end:],
        dtype=torch.long
    ).to(device)
    
    print("Training graphs:", len(training))
    print("Testing graphs:", len(testing))


    # Construct the first class of MaGNet model 
    class Weaker_First(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # build convolutional layers
            self.conv1 = MaGNetConv(data.num_node_features, 64)
            self.conv2 = MaGNetConv(64, 128)
            self.conv3 = MaGNetConv(128, 64)
            self.local_size = 7
            self.lr1 = nn.Linear(self.local_size*64, 256)
            self.lr2 = nn.Linear(256, 32)
            self.lr3 = nn.Linear(32, 2)
            #self.softmax = nn.Softmax(dim=1)

        def forward(self, x, edge_index):
            # model transformation and parameter updating rule
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv3(x, edge_index)
            lap = x
            
            # graph pooling 
            required_values = torch.tensor(list(range(self.local_size)))
            random_tensor = torch.randint(0, self.local_size, (data.num_nodes - self.local_size,))
            combined_tensor = torch.cat((required_values, random_tensor))
            shuffled_tensor = combined_tensor[torch.randperm(data.num_nodes)]
            shuffled_tensor = shuffled_tensor.to(device)
            x = global_max_pool(x,shuffled_tensor)

            # flatten and transformation
            x = torch.flatten(x)
            x = F.relu(self.lr1(x))
            x = F.relu(self.lr2(x))
            x = self.lr3(x)

            # return logits, feature and embeddings 
            return [F.log_softmax(x.reshape(1,len(x)), dim=1), lap, x]

    # Construct the middle class of MaGNet model 
    class Weaker_Middle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # build convolutional layer
            self.conv1 = MaGNetConv(64, 64)
            self.local_size = 7
            self.lr1 = nn.Linear(self.local_size*64, 128)
            self.lr2 = nn.Linear(128, 48)
            self.lr3 = nn.Linear(48, 2)
            #self.softmax = nn.Softmax(dim=1)

        def forward(self, x, edge_index):
            # model transformation and parameter updating rule
            x = self.conv1(x, edge_index)
            lap = x
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            
            # graph pooling 
            required_values = torch.tensor(list(range(self.local_size)))
            random_tensor = torch.randint(0, self.local_size, (data.num_nodes - self.local_size,))
            combined_tensor = torch.cat((required_values, random_tensor))
            shuffled_tensor = combined_tensor[torch.randperm(data.num_nodes)]
            shuffled_tensor = shuffled_tensor.to(device)  
            x = global_max_pool(x,shuffled_tensor)

            x = torch.flatten(x)
            x = F.relu(self.lr1(x))
            x = F.relu(self.lr2(x))
            x = self.lr3(x)
            
            # output the logits and embeddings
            return [F.log_softmax(x.reshape(1,len(x)), dim=1), lap, x]

    class WeakLast(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(WeakLast, self).__init__()
            # build MLP layers with sigmoid function
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            # model transformation and parameter updating rule
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            x = self.sigmoid(x)
            return x


    # transfer to the computational devices 
    layer_num = 6
    models = [] 
    latent_rep = []
    quality_vec = []
    torch.manual_seed(1)
    model = Weaker_First().to(device)

    # initilize adam optimizer 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
    model.train()
    iter_num = 200
    iter_num_first = 75


    for epoch in range(iter_num_first):

        print(
            f"\rFirst MaGNet layer - Epoch {epoch + 1}/{iter_num_first}",
            end="",
            flush=True
        )
    
        # compute the logits
        train_loss_compute = compute_loss(training)
        training_softprob = train_loss_compute[0]
    
        optimizer.zero_grad()
    
        # compute the loss value
        # use training loss for learning
        training_loss = F.nll_loss(
            training_softprob,
            training_label
        )
    
        training_loss.backward()
        optimizer.step()
    
        # evaluate the training accuracy
        train_acc = model_acc(
            training,
            "training"
        )
    
    print()

    # compute the embeddings of the first class of MaGNet model
    curr_feat = train_loss_compute[1]
    curr_latent = np.array(train_loss_compute[2])

    # initilize the weights, and error rates
    models.append(model)
    latent_rep.append(curr_latent)
    weights = torch.ones(training.shape[0])
    weights = nn.functional.normalize(weights,p=2, dim=0)
    err_rate = weight_loss(training, weights)
    quality_vec.append(quality_update(err_rate))
    
    original_training_features = original_features[:train_end].copy()
    original_testing_features = original_features[train_end:].copy()

    # model training on the middle class of MaGNet model
    for _ in range(layer_num-1):
        
        nodes_fea = np.asarray(nodes_fea, dtype=np.float32)        
        x = torch.from_numpy(nodes_fea)

        model = Weaker_Middle().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-5)
        model.train()
        training = np.array(curr_feat)
        nodes_fea = training[0, :, :]
        x = torch.from_numpy(
            np.asarray(nodes_fea, dtype=np.float32)
        )
        data = Data(x=x, edge_index=edge_index)
        
        for epoch in range(iter_num):

            print(
                f"\rMiddle MaGNet layer - Epoch {epoch + 1}/{iter_num}",
                end="",
                flush=True
            )
        
            train_loss_compute = compute_loss(training)
            training_softprob = train_loss_compute[0]
        
            optimizer.zero_grad()
        
            training_loss = F.nll_loss(
                training_softprob,
                training_label
            )
        
            training_loss.backward()
            optimizer.step()
        
            train_acc = model_acc(
                training,
                "training"
            )
        
            curr_feat = train_loss_compute[1]
        
        print()
        # compute and store the  weights, and error rates
        models.append(model)
        curr_latent = np.array(train_loss_compute[2])
        latent_rep.append(curr_latent)
        err_rate = weight_loss(training, weights)
        quality_vec.append(quality_update(err_rate))
        weights = weight_update(err_rate, training, weights)
        
    
    # Compute the embeddings of the MaGNet model via the aggregation 
    final_emb = torch.zeros(
        np.array(latent_rep[0]).shape[0], 
        np.array(latent_rep[0]).shape[1]
    ).to(device)
    
    
    for i in range(np.array(latent_rep[0]).shape[0]):
        for j in range(layer_num):
            final_emb[i,:] += quality_vec[j]*(
                torch.tensor(latent_rep[j])[i,:]
            ).to(device)

    # batching the final embeddings
    train_tensor = training_label.float().view(-1, 1)
    
    train_dataset = TensorDataset(
        final_emb,
        train_tensor
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True
    )
    
    layer_num = 6
    test_latent_rep = []

    # compute the testing embeddings of MaGNet model
    for j in range(layer_num):
        model = models[j]
        test_loss_compute = compute_loss(testing)
        curr_latent = np.array(test_loss_compute[2])
        test_latent_rep.append(curr_latent)
        curr_feat = np.array(test_loss_compute[1])
        testing = curr_feat
    test_emb = torch.zeros(np.array(test_latent_rep[0]).shape[0], np.array(test_latent_rep[0]).shape[1]).to(device)
    for i in range(np.array(test_latent_rep[0]).shape[0]):
        for j in range(layer_num):
            test_emb[i,:] += quality_vec[j]*(torch.tensor(test_latent_rep[j])[i,:]).to(device)

    # embedding updating via non-linear transformation (optinal)  
    input_size = final_emb.shape[1]
    hidden_size = 64
    output_size = 1
    mask_thres = 0.5

    model = WeakLast(input_size, hidden_size, output_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    # model training on the last class of MaGNet model 
    num_epochs = 5
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()


    # ============================================================
    # SAVE EVERYTHING REQUIRED FOR INTERPRETATION
    # ============================================================

    test_res = test_acc(model)


    # ------------------------------------------------------------
    # Final test predictions
    # ------------------------------------------------------------

    model.eval()

    with torch.no_grad():
        test_outputs = model(test_emb)

        test_predictions = (
            (test_outputs > 0.3)
            .long()
            .view(-1)
        )
    print("Prediction threshold: 0.30")
    print("Number predicted as 1:", int(test_predictions.sum().item()))
    print("Number predicted as 0:", int((test_predictions == 0).sum().item()))
        
    test_predictions_np = (
        test_predictions
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    
    test_labels_np = (
        original_labels[train_end:]
        .reshape(-1)
    )
    
    test_accuracy = accuracy_score(
        test_labels_np,
        test_predictions_np
    )
    
    test_precision = precision_score(
        test_labels_np,
        test_predictions_np,
        zero_division=0
    )
    
    test_recall = recall_score(
        test_labels_np,
        test_predictions_np,
        zero_division=0
    )
    
    print("Test accuracy:", test_accuracy)
    print("Test precision:", test_precision)
    print("Test recall:", test_recall)

    # ------------------------------------------------------------
    # Convert quality weights to plain CPU tensors
    # ------------------------------------------------------------

    quality_weights = torch.stack([
        q.detach().cpu()
        if torch.is_tensor(q)
        else torch.tensor(q, dtype=torch.float32)
        for q in quality_vec
    ])

    # ============================================================
    # 1. HUMAN-READABLE TRAINING FEATURES
    # ============================================================

    training_features_rows = []

    for graph_idx in range(train_end):

        date = features_df.iloc[graph_idx]["Date"]

        for node_idx in range(dataset.shape[1]):

            training_features_rows.append({
                "graph_index": graph_idx,
                "date": date,
                "node_id": node_idx,
                "production": float(
                    dataset[graph_idx, node_idx, 0]
                ),
                "sales_order": float(
                    dataset[graph_idx, node_idx, 1]
                ),
                "delivery": float(
                    dataset[graph_idx, node_idx, 2]
                ),
                "factory_issue": float(
                    dataset[graph_idx, node_idx, 3]
                )
            })

    training_features_df = pd.DataFrame(
        training_features_rows
    )

    training_features_df.to_csv(
        "training_features.csv",
        index=False
    )

    # ============================================================
    # 2. HUMAN-READABLE TESTING FEATURES
    # ============================================================

    testing_features_rows = []

    for graph_idx in range(train_end, len(dataset)):

        date = features_df.iloc[graph_idx]["Date"]

        for node_idx in range(dataset.shape[1]):

            testing_features_rows.append({
                "graph_index": graph_idx,
                "date": date,
                "node_id": node_idx,
                "production": float(
                    dataset[graph_idx, node_idx, 0]
                ),
                "sales_order": float(
                    dataset[graph_idx, node_idx, 1]
                ),
                "delivery": float(
                    dataset[graph_idx, node_idx, 2]
                ),
                "factory_issue": float(
                    dataset[graph_idx, node_idx, 3]
                )
            })

    testing_features_df = pd.DataFrame(
        testing_features_rows
    )

    testing_features_df.to_csv(
        "testing_features.csv",
        index=False
    )

    # ============================================================
    # 3. HUMAN-READABLE TRAINING LABELS
    # ============================================================

    training_labels_df = pd.DataFrame({
    "graph_index": np.arange(train_end),
    "date": features_df.iloc[:train_end]["Date"].values,
    "label": training_label.detach().cpu().numpy().reshape(-1)
    })

    training_labels_df.to_csv(
        "training_labels.csv",
        index=False
    )

    # ============================================================
    # 4. HUMAN-READABLE TESTING LABELS
    # ============================================================

    testing_labels_df = pd.DataFrame({
    "graph_index": np.arange(train_end, len(original_features)),
    "date": features_df.iloc[train_end:]["Date"].values,
    "label": testing_label.detach().cpu().numpy().reshape(-1)
    })

    testing_labels_df.to_csv(
        "testing_labels.csv",
        index=False
    )
    
    
    # ============================================================
    # 5. HUMAN-READABLE TRAINING EMBEDDINGS
    # ============================================================
    
    training_embeddings = (
        final_emb.detach()
        .cpu()
        .numpy()
    )
    
    print(
        "Training embeddings shape:",
        training_embeddings.shape
    )
    
    training_embeddings_df = pd.DataFrame(
        training_embeddings,
        columns=[
            f"embedding_{i}"
            for i in range(training_embeddings.shape[1])
        ]
    )
    
    training_embeddings_df.insert(
        0,
        "graph_index",
        np.arange(train_end)
    )
    
    training_embeddings_df.insert(
        1,
        "date",
        features_df.iloc[:train_end]["Date"].to_numpy()
    )
    
    training_embeddings_df.insert(
        2,
        "label",
        original_labels[:train_end]
    )
    
    training_embeddings_df.to_csv(
        "training_embeddings.csv",
        index=False
    )

    # ============================================================
    # 6. TESTING FINAL EMBEDDINGS
    # ============================================================

    testing_embeddings = (
        test_emb.detach()
        .cpu()
        .numpy()
    )

    testing_embeddings_df = pd.DataFrame(
        testing_embeddings,
        columns=[
            f"embedding_{i}"
            for i in range(testing_embeddings.shape[1])
        ]
    )

    testing_embeddings_df.insert(
        0,
        "graph_index",
        np.arange(train_end, len(original_features))
    )

    testing_embeddings_df.insert(
        1,
        "date",
        features_df.iloc[train_end:]["Date"].to_numpy()
    )

    testing_embeddings_df.insert(
        2,
        "true_label",
        original_labels[train_end:]
    )

    testing_embeddings_df.to_csv(
        "testing_embeddings.csv",
        index=False
    )

    # ============================================================
    # 7. TEST PREDICTIONS
    # ============================================================
    
    test_predictions_np = (
        test_predictions
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    
    test_probabilities_np = (
        test_outputs
        .detach()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    
    
    testing_predictions_df = pd.DataFrame({
        "graph_index":
            np.arange(train_end, len(original_features)),
    
        "date":
            features_df.iloc[train_end:]["Date"].to_numpy(),
    
        "true_label":
            original_labels[train_end:],
    
        "prediction":
            test_predictions_np,
    
        "probability":
            test_probabilities_np
    })
    
    testing_predictions_df.to_csv(
        "testing_predictions.csv",
        index=False
    )

    # ============================================================
    # 8. MAGNET QUALITY WEIGHTS
    # ============================================================

    quality_weights_numpy = (
        quality_weights
        .numpy()
        .reshape(-1)
    )

    quality_weights_df = pd.DataFrame({
        "layer": np.arange(
            len(quality_weights_numpy)
        ),
        "quality_weight":
            quality_weights_numpy
    })

    quality_weights_df.to_csv(
        "magnet_quality_weights.csv",
        index=False
    )

    # ============================================================
    # 9. HUMAN-READABLE METADATA
    # ============================================================

    metadata = {
        "dataset": {
            "number_of_graphs": int(len(dataset)),
            "number_of_nodes": int(dataset.shape[1]),
            "node_feature_dimension": int(dataset.shape[2]),
            "number_of_edges": int(len(edges_df)),
            "edge_index_shape":
                list(edge_index.shape)
        },

        "features": [
            "production",
            "sales_order",
            "delivery",
            "factory_issue"
        ],

        "split": {
            "method": "chronological",
            "train_ratio": 0.65,
            "train_end": int(train_end),
            "training_graphs": int(len(training)),
            "testing_graphs": int(len(testing))
        },

        "model": {
            "model": "MaGNet",
            "layer_num": int(layer_num),
            "first_layer_hidden_dimensions": [
                64,
                128,
                64
            ],
            "middle_layer_hidden_dimension": 64,
            "local_size": 7,

            "classifier": {
                "input_size": int(input_size),
                "hidden_size": int(hidden_size),
                "output_size": int(output_size)
            }
        },

        "results": {
            "test_accuracy": float(test_res)
        },

        "files": {
            "training_features":
                "training_features.csv",

            "testing_features":
                "testing_features.csv",

            "training_labels":
                "training_labels.csv",

            "testing_labels":
                "testing_labels.csv",

            "training_embeddings":
                "training_embeddings.csv",

            "testing_embeddings":
                "testing_embeddings.csv",

            "testing_predictions":
                "testing_predictions.csv",

            "quality_weights":
                "magnet_quality_weights.csv",

            "interpretation_checkpoint":
                "supplychain_interpretation_checkpoint.pt"
        }
    }

    with open(
        "supplychain_metadata.json",
        "w"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=4
        )

    # ============================================================
    # 10. PYTORCH CHECKPOINT FOR INTERPRETATION
    # ============================================================

    interpretation_checkpoint = {

        # --------------------------------------------------------
        # Trained MaGNet models
        # --------------------------------------------------------

        "magnet_models": [
            m.state_dict()
            for m in models
        ],

        "final_classifier":
            model.state_dict(),

        # --------------------------------------------------------
        # Model architecture
        # --------------------------------------------------------

        "layer_num":
            layer_num,

        "node_feature_dim":
            4,

        "hidden_dims":
            [64, 128, 64],

        "middle_hidden_dim":
            64,

        "local_size":
            7,

        "classifier_input_size":
            int(input_size),

        "classifier_hidden_size":
            int(hidden_size),

        "classifier_output_size":
            int(output_size),

        # --------------------------------------------------------
        # Graph structure
        # --------------------------------------------------------

        "edge_index":
            edge_index.cpu(),

        # --------------------------------------------------------
        # Original graph data
        # --------------------------------------------------------

        "training_features": torch.tensor(
            original_features[:train_end],
            dtype=torch.float32
        ),
        
        "testing_features": torch.tensor(
            original_features[train_end:],
            dtype=torch.float32
        ),

        "training_labels":
            training_label.cpu(),

        "testing_labels":
            testing_label.cpu(),

        # --------------------------------------------------------
        # Split information
        # --------------------------------------------------------

        "train_end":
            int(train_end),

        "num_training_graphs":
            int(len(training)),

        "num_testing_graphs":
            int(len(testing)),

        # --------------------------------------------------------
        # MaGNet intermediate representations
        # --------------------------------------------------------

        "training_latent_rep": [
            torch.tensor(
                latent,
                dtype=torch.float32
            )
            for latent in latent_rep
        ],

        "testing_latent_rep": [
            torch.tensor(
                latent,
                dtype=torch.float32
            )
            for latent in test_latent_rep
        ],

        "training_final_embedding":
            final_emb.detach().cpu(),

        "testing_final_embedding":
            test_emb.detach().cpu(),

        # --------------------------------------------------------
        # Node-level intermediate features
        # --------------------------------------------------------

        "testing_final_node_features":
            torch.tensor(
                curr_feat,
                dtype=torch.float32
            ),

        # --------------------------------------------------------
        # Quality weights
        # --------------------------------------------------------

        "quality_vec":
            quality_weights,

        # --------------------------------------------------------
        # Final classifier results
        # --------------------------------------------------------

        "test_outputs":
            test_outputs.detach().cpu(),

        "test_predictions":
            test_predictions.cpu(),

        "test_accuracy":
            float(test_res)
    }

    torch.save(
        interpretation_checkpoint,
        "supplychain_interpretation_checkpoint.pt"
    )

    # ============================================================
    # 11. SIMPLE JSON RESULT
    # ============================================================

    with open("proposed_supplychain.json", "w") as f:
        json.dump(
            {
                "test_accuracy": float(test_accuracy),
                "test_precision": float(test_precision),
                "test_recall": float(test_recall)
            },
            f,
            indent=4
        )

    # ============================================================
    # SUMMARY
    # ============================================================

    print()
    print("=" * 60)
    print("FILES SAVED")
    print("=" * 60)

    print("training_features.csv")
    print("testing_features.csv")
    print("training_labels.csv")
    print("testing_labels.csv")
    print("training_embeddings.csv")
    print("testing_embeddings.csv")
    print("testing_predictions.csv")
    print("magnet_quality_weights.csv")
    print("supplychain_metadata.json")
    print("proposed_supplychain.json")
    print("supplychain_interpretation_checkpoint.pt")

    print("=" * 60)
    print(
        "MaGNet testing accuracy: "
        f"{test_res:.3f}"
    )
    print("=" * 60)
    
if __name__ == "__main__":
    realdata_analysis()