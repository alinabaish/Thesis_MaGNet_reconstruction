#!/usr/bin/env python
# coding: utf-8

import json
import torch
import random
import numpy as np 
from torch_geometric.data import Data
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
from torch_geometric.nn import GCNConv, GATConv, APPNP, TransformerConv
from torch.nn import Linear, Sequential, BatchNorm1d, Dropout
from convlayer import MaGNetConv
from pooling import global_max_pool
from norm_operation import PairNorm, MeanSubtractionNorm
from gpslayer import * 
import warnings
from sklearn.metrics import accuracy_score, precision_score, recall_score

def simulation():

   # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Continuing with the script...")
    else:
        print("CUDA is not available. Running on CPU.")

    # Compuate the output the probability logits, loss value, and feature embeddings
    def compute_loss(dataset):
        out = []
        feat = []
        emb = [] 
        
        for i in range(dataset.shape[0]):
            # extract the node features from dataset
            nodes_fea = [tmp for tmp in np.transpose(dataset[i,:,:])]
            x = torch.tensor(nodes_fea, dtype=torch.float)
            
            # contrust computational graph
            data = Data(x=x, edge_index=edge_index)
            data = data.to(device)
            
            # model inference 
            output_model = model(data.x, data.edge_index)

            # compute the model output for loss value, logits, and embeddings
            tmpout = output_model[0]
            tmpfeat = np.transpose(output_model[1].cpu().detach().numpy())
            tmpemb = output_model[2].cpu().detach().numpy()

            # store the model output for loss value, logits, and embeddings
            #out.append(tmpout[0])
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
            nodes_fea = [tmp for tmp in np.transpose(dataset[i,:,:])]
            x = torch.tensor(nodes_fea, dtype=torch.float)
            #data = Data(x=x, edge_index=sub_edge_index)

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
        else:
            correct = (pred == valid_label).sum()

        # compute the accuracy
        acc = int(correct) / dataset.shape[0]

        return acc    


    # Compute the weighted loss error rate
    def weight_loss(dataset, weights):
        pred = []

        # model prediction on graph labels
        for i in range(dataset.shape[0]):
            nodes_fea = [tmp for tmp in np.transpose(dataset[i,:,:])]
            x = torch.tensor(nodes_fea, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)
            data = data.to(device)
            pred.append(model(data.x, data.edge_index)[0].argmax(dim=1)[0])
            
        pred = torch.Tensor(pred).to(device)

        # compute the weighted error rate for classifier 
        err_rate = (weights*(pred != training_label)).sum()/weights.sum()

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
            nodes_fea = [tmp for tmp in np.transpose(dataset[i,:,:])]
            x = torch.tensor(nodes_fea, dtype=torch.float)
            data = Data(x=x, edge_index=edge_index)
            data = data.to(device)

            # model prediction and weights updating via exponential rule
            curr_pred = torch.Tensor(model(data.x, data.edge_index)[0].argmax(dim=1)[0]).to(device) 
            weights[i] = torch.exp(alpha*(training_label[i] != curr_pred) )

        # normalize the weigths for all samples
        weights = nn.functional.normalize(weights,p=2, dim=0)
        return weights

        # Evaluate the final model performance
    def test_acc(model):
    
        predictions = []
    
        # build evaluation dataset and batch
        X_test_tensor = test_emb.clone().detach()
        test_dataset = TensorDataset(X_test_tensor)
        test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
        # freeze the model parameter and perform inference
        with torch.no_grad():
            for features in test_loader:
                features = features[0]
                outputs = model(features)
    
                # Apply threshold to convert logits to class labels
                predicted = (outputs > 0.5).float()
    
                predictions.extend(predicted.tolist())
    
        # Flatten the list
        predictions = [pred[0] for pred in predictions]
    
        # True labels
        true_labels = testing_label.cpu().tolist()
    
        # Calculate metrics
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(
            true_labels,
            predictions,
            zero_division=0
        )
        recall = recall_score(
            true_labels,
            predictions,
            zero_division=0
        )
    
        # Print results
        print("\n===== MaGNet Testing Results =====")
        print("Accuracy:  {:.4f}".format(accuracy))
        print("Precision: {:.4f}".format(precision))
        print("Recall:    {:.4f}".format(recall))
    
        return accuracy


    node_size = 75
    use_node = 20
    p = 10
    n = 100
    np.random.seed(2)
    random.seed(2)
    label = []

    for i in range(n):
        
        # generate the node features and graph labels
        mean = np.repeat(0, p)
        cov1 = np.random.uniform(0.1,0.1,p*p)
        cov1 = np.reshape(cov1, (p,p))
        np.fill_diagonal(cov1,1)
        x1 = np.random.multivariate_normal(mean, cov1, size=use_node)
        y = (np.mean(x1)>=0)+0
        label.append(y)
        x2 = np.random.uniform(0,1,(node_size-use_node)*p)
        x2 = np.reshape(x2, ((node_size-use_node),p))
        if i ==0: 
            x = np.transpose(np.concatenate((x1, x2), axis=0))
        else:
            x = np.dstack((x,np.transpose(np.concatenate((x1, x2), axis=0))))

    # generate the training and testing datasets
    label = np.array(label)
    data_lfp = np.transpose(x, (2,0,1))
    indices = np.random.permutation(data_lfp.shape[0])
    training_idx, test_idx = indices[:int(data_lfp.shape[0]/1.5)], indices[int(data_lfp.shape[0]/1.5):], 
    training,  testing = data_lfp[training_idx,:,:], data_lfp[test_idx,:,:]
    
    # Preserve the ORIGINAL node-feature graphs for interpretability
    original_training = training.copy()
    original_testing = testing.copy()
    
    device = torch.device('cpu')
    torch.use_deterministic_algorithms(True)
    warnings.filterwarnings('ignore')

    # generate the training and testing graph labels
    training_label = torch.tensor(label[training_idx], dtype=torch.long).to(device)
    testing_label = torch.tensor(label[test_idx], dtype=torch.long).to(device)

    # generate the fractional adj matrix
    adj_mat = np.ones(shape=(data_lfp.shape[2],data_lfp.shape[2]))
    for i in range(data_lfp.shape[2]):
        for j in range(data_lfp.shape[2]):
            corr_total = 0 
            for k in range(len(data_lfp[:,0,i])):
                i_channel = data_lfp[k,:,i]
                j_channel = data_lfp[k,:,j]
                corr_total += np.corrcoef(i_channel,j_channel)[0,1]
            corr_i_j = corr_total/len(data_lfp[:,0,i])
            adj_mat[i,j] = np.abs(corr_i_j)
            
    # generate the hard adj matrix 
    thres = np.quantile(adj_mat,0.5)
    adj_mat[np.where(adj_mat<thres)] = 0
    adj_mat[np.where(adj_mat==1)] = 0
    adj_mat[np.where(adj_mat>0)] = 1
    adj_mat = torch.tensor(adj_mat)

    # generate the computation graph structure
    edge_index = adj_mat.nonzero().t().contiguous()
    nodes_fea = np.array([tmp for tmp in np.transpose(data_lfp[0,:,:])])
    x = torch.tensor(nodes_fea, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)


    # Construct the first class of MaGNet model 
    class Weaker_First(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # build convolutional layers
            self.conv0 = nn.Conv1d(data.num_nodes, data.num_nodes, 3, 1,padding=1)
            self.conv1 = MaGNetConv(data.num_node_features, 64)
            self.conv2 = MaGNetConv(64, 128)
            self.conv3 = MaGNetConv(128, 64)
            self.local_size = 7
            self.lr1 = nn.Linear(self.local_size*64, 256)
            self.lr2 = nn.Linear(256, 32)
            self.lr3 = nn.Linear(32, 2)
            #self.softmax = nn.Softmax(dim=1)

        def forward(self, x, edge_index):
            #x, edge_index = data.x, data.edge_index
            #y = self.conv1(x, edge_index)
            
            # model transformation and parameter updating rule
            x = self.conv0(x)
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            x = self.conv3(x, edge_index)
            lap = x
            #batch = torch.randint(0, self.local_size, (data.num_nodes,))
            
            # graph pooling indexing 
            required_values = torch.tensor(list(range(self.local_size)))
            random_tensor = torch.randint(0, self.local_size, (data.num_nodes - self.local_size,))
            combined_tensor = torch.cat((required_values, random_tensor))
            shuffled_tensor = combined_tensor[torch.randperm(data.num_nodes)]
            x = global_max_pool(x,shuffled_tensor)
            x = torch.flatten(x)
            x = F.relu(self.lr1(x))
            x = F.relu(self.lr2(x))
            x = self.lr3(x)

            # output the logits and embeddings
            return [F.log_softmax(x.reshape(1,len(x)), dim=1), lap, x]
            #return [F.log_softmax(x.reshape(1,len(x)), dim=1), x]
            #return  F.log_softmax(x, dim=1)

    # Construct the middle class of MaGNet model 
    class Weaker_Middle(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # build convolutional layers
            self.conv1 = MaGNetConv(64, 64)
            self.local_size = 7
            self.lr1 = nn.Linear(self.local_size*64, 128)
            self.lr2 = nn.Linear(128, 48)
            self.lr3 = nn.Linear(48, 2)
            #self.softmax = nn.Softmax(dim=1)
            
        def forward(self, x, edge_index):
            #x, edge_index = data.x, data.edge_index
            
            # model transformation and parameter updating rule
            x = self.conv1(x, edge_index)
            lap = x
            x = F.relu(x)
            x = F.dropout(x, training=self.training)
            #batch = torch.randint(0, self.local_size, (data.num_nodes,))

            # graph pooling
            required_values = torch.tensor(list(range(self.local_size)))
            random_tensor = torch.randint(0, self.local_size, (data.num_nodes - self.local_size,))
            combined_tensor = torch.cat((required_values, random_tensor))
            shuffled_tensor = combined_tensor[torch.randperm(data.num_nodes)]
            x = global_max_pool(x,shuffled_tensor)
            x = torch.flatten(x)
            x = F.relu(self.lr1(x))
            x = F.relu(self.lr2(x))
            x = self.lr3(x)

            # output the logits and embeddings
            return [F.log_softmax(x.reshape(1,len(x)), dim=1), lap, x]

    # Construct the last class of MaGNet model 
    class WeakLast(nn.Module):
        def __init__(self, input_size, hidden_size, output_size):
            super(WeakLast, self).__init__()
            # build MLP layers with sigmoid function
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, output_size)
            self.sigmoid = nn.Sigmoid()
            
        # model transformation and parameter updating rule
        def forward(self, x):
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
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=5e-5)
    model.train()
    iter_num = 200
    iter_num_first = 40

    for epoch in range(iter_num_first):

        # compute the logits
        train_loss_compute = compute_loss(training)
        training_softprob = train_loss_compute[0]
        #testing_softprob = compute_loss(testing)[0]

        optimizer.zero_grad()
        
        # compute the loss value 
        training_loss = F.nll_loss(training_softprob, training_label) 
        #testing_loss = F.nll_loss(testing_softprob, testing_label) 
        
        #losses.append(training_loss)    
        #use training loss for learning
        training_loss.backward()
        optimizer.step()
        # evaluate the training accuracy
        train_acc = model_acc(training, "training")
        #acc.append(train_acc)

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

    # model training on the middle class of MaGNet model
    for _ in range(layer_num-1):
        
        model = Weaker_Middle().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=5e-5)
        model.train()
        training = np.array(curr_feat)
        nodes_fea = [tmp for tmp in np.transpose(training[0,:,:])]
        x = torch.tensor(nodes_fea, dtype=torch.float)
        data = Data(x=x, edge_index=edge_index)

        for epoch in range(iter_num):

            train_loss_compute = compute_loss(training)
            training_softprob = train_loss_compute[0]
            #testing_softprob = compute_loss(testing)[0]

            optimizer.zero_grad()

            training_loss = F.nll_loss(training_softprob, training_label) 
            training_loss.backward()
            optimizer.step()

            train_acc = model_acc(training, "training")
            #acc.append(train_acc)

            curr_feat = train_loss_compute[1]

        # compute and store the  weights, and error rates
        models.append(model)
        curr_latent = np.array(train_loss_compute[2])
        latent_rep.append(curr_latent)
        err_rate = weight_loss(training, weights)
        quality_vec.append(quality_update(err_rate))
        weights = weight_update(err_rate, training, weights)

    # Compute the embeddings of the MaGNet model via the aggregation 
    final_emb = torch.zeros(np.array(latent_rep[0]).shape[0], np.array(latent_rep[0]).shape[1])
    for i in range(np.array(latent_rep[0]).shape[0]):
        for j in range(layer_num):
            final_emb[i,:] += quality_vec[j]*np.array(latent_rep[j])[i,:]
            
    # batching the dataset 
    train_tensor = torch.tensor(training_label, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(final_emb, train_tensor)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

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
    test_emb = torch.zeros(np.array(test_latent_rep[0]).shape[0], np.array(test_latent_rep[0]).shape[1])
    for i in range(np.array(test_latent_rep[0]).shape[0]):
        for j in range(layer_num):
            test_emb[i,:] += quality_vec[j]*np.array(test_latent_rep[j])[i,:]

    # embedding updating via non-linear transformation (optinal)  
    input_size = final_emb.shape[1]
    hidden_size = 64
    output_size = 1
    model = WeakLast(input_size, hidden_size, output_size).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    # model training on the last class of MaGNet model 
    num_epochs = 1
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # ==========================================================
    # SAVE TRAINED MAGNET MODELS FOR INTERPRETABILITY ANALYSIS
    # ==========================================================

    print("\nSaving trained MaGNet models and explanation data...")

    # Save all six trained MaGNet component models
    for i, component_model in enumerate(models):
        torch.save(
            component_model.state_dict(),
            f"magnet_component_{i}.pt"
        )

    # Save the final WeakLast classifier
    torch.save(
        model.state_dict(),
        "magnet_final_classifier.pt"
    )

    # Save all data and parameters needed for later explanation
    explanation_data = {
        # Graph structure
        "edge_index": edge_index.cpu(),

        # MaGNet component aggregation weights
        "quality_vec": [
            float(q.detach().cpu())
            for q in quality_vec
        ],

        # Dataset information
        "original_training": original_training,
        "original_testing": original_testing,

        # Labels
        "training_label": training_label.cpu(),
        "testing_label": testing_label.cpu(),

        # Original experiment parameters
        "node_size": node_size,
        "use_node": use_node,
        "num_features": p,
        "layer_num": layer_num,

        # Final classifier dimensions
        "input_size": input_size,
        "hidden_size": hidden_size,
        "output_size": output_size
    }

    torch.save(
        explanation_data,
        "magnet_explanation_data.pt"
    )

    print("All trained models and explanation data saved successfully.")

    # Save MaGNet component quality weights for interpretability
    quality_values = [float(q.detach().cpu()) for q in quality_vec]

    print("\nMaGNet component quality weights:")
    for i, q in enumerate(quality_values):
        print(f"Component {i+1}: {q:.6f}")

    with open('MaGNet_quality_weights_setting1.json', 'w') as f:
        json.dump(quality_values, f)

    # Save testing results of MaGNet model to a JSON file
    test_res = test_acc(model)
    print('MaGNet testing accuracy: {:.3f}'.format(test_res))
    
    
    # Stop here — do not run PNGAT, APPNP, MSGCN, GTN, or GPST
    return
    
    
if __name__ == '__main__':
        simulation()