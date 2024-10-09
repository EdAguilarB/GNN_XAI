import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks import BaseNetwork
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, GATConv
import argparse


class GAT(BaseNetwork):

    def __init__(self, opt: argparse.Namespace, n_node_features:int, n_edge_features:int, **kwargs):

        # Call the constructor of the parent class (BaseNetwork)
        super().__init__(opt=opt, n_node_features=n_node_features, n_edge_features=n_edge_features, **kwargs)

        
        # Set class variables
        self._name = "GAT"
        gat_params = ['heads']
        for param in gat_params:
            setattr(self, param, self.kwargs.pop(param))

        # linear expansion of node features
        self.linear = nn.Linear(n_node_features, self.embedding_dim)
        self.batch_norm = nn.BatchNorm1d(self.embedding_dim)

        #Convolutions
        self.conv_layers = nn.ModuleList([])
        self.norm_layers = nn.ModuleList([])
        for _ in range(self.n_convolutions):
            self.conv_layers.append(GATConv(in_channels=self.embedding_dim, 
                                            out_channels=self.embedding_dim,
                                            heads=self.heads, 
                                            concat=False,
                                            edge_dim=n_edge_features,
                                            dropout=self.dropout,
                                            ))
            self.norm_layers.append(nn.BatchNorm1d(self.embedding_dim))


        #graph embedding is the concatenation of the global mean and max pooling, thus 2*embedding_dim
        graph_embedding = self.embedding_dim*2

        self.dropout_layer = nn.Dropout(p=self.dropout)

        #Readout layers
        self.readout = nn.ModuleList([])

        for _ in range(self.readout_layers-1):
            reduced_dim = int(graph_embedding/2)
            self.readout.append(nn.Sequential(nn.Linear(graph_embedding, reduced_dim), 
                                              nn.BatchNorm1d(reduced_dim)))
            graph_embedding = reduced_dim

        #Final readout layer
        self.output_layer = nn.Linear(graph_embedding, self._n_classes)
        
        self._make_loss()
        self._make_optimizer(opt.optimizer)
        self._make_scheduler(scheduler=opt.scheduler,)
        


    def forward(self, x=None, edge_index=None, batch_index=None, edge_attr=None):


        x = F.leaky_relu(self.batch_norm(self.linear(x)))

        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            x = conv_layer(x, edge_index, edge_attr)
            x = norm_layer(x)

        x = torch.cat([gmp(x, batch_index), 
                            gap(x, batch_index)], dim=1)
        
        for layer in self.readout:
            x = F.leaky_relu(layer(x))
        
        x = self.output_layer(x)

        return x