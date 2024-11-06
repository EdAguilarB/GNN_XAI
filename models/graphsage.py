import torch
import torch.nn as nn
import torch.nn.functional as F
from models.networks import BaseNetwork
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, global_add_pool as gsp, SAGEConv
import argparse

class graphsage(BaseNetwork):

    def __init__(self, opt: argparse.Namespace, n_node_features:int, n_edge_features:int, **kwargs):

        # Call the constructor of the parent class (BaseNetwork)
        super().__init__(opt=opt, n_node_features=n_node_features, n_edge_features=n_edge_features, **kwargs)

        # Set class variables
        self._name = "graphsage"
        graph_sample_params = ['aggr']
        for param in graph_sample_params:
            setattr(self, param, self.kwargs.pop(param))


        # linear expansion of node features
        self.linear = nn.Linear(n_node_features, self.embedding_dim)
        self.batch_norm = nn.BatchNorm1d(self.embedding_dim)


        #Convolutions
        self.conv_layers = nn.ModuleList([])
        self.norm_layers = nn.ModuleList([])

        for _ in range(self.n_convolutions):
            self.conv_layers.append(SAGEConv(in_channels=self.embedding_dim, 
                                            out_channels=self.embedding_dim,
                                            aggr=self.aggr
                                            ))
            self.norm_layers.append(nn.BatchNorm1d(self.embedding_dim))

        #graph embedding is the concatenation of the global mean and max pooling, thus 2*embedding_dim
        graph_embedding = self.graph_embedding

        self.dropout_layer = nn.Dropout(p=self.dropout)

        #Readout layers
        self.readout = nn.ModuleList([])

        for _ in range(self.readout_layers-1):
            reduced_dim = int(graph_embedding/2)
            self.readout.append(nn.Sequential(nn.Linear(graph_embedding, reduced_dim), 
                                              nn.BatchNorm1d(reduced_dim),
                                              ))
            graph_embedding = reduced_dim

        #Final readout layer
        self.output_layer = nn.Linear(graph_embedding, self.n_classes)
        
        
        self._make_loss()
        self._make_optimizer(opt.optimizer)
        self._make_scheduler(scheduler=opt.scheduler,)
        

    def forward(self,x, edge_index,  batch_index=None, edge_attr=None):

        x = F.leaky_relu(self.batch_norm(self.linear(x)))

        for conv_layer, norm_layer in zip(self.conv_layers, self.norm_layers):
            x = F.leaky_relu(norm_layer(conv_layer(x, edge_index)))

        if self.pooling == 'mean':
            x = gap(x, batch_index)
        elif self.pooling == 'max':
            x = gmp(x, batch_index)
        elif self.pooling == 'sum':
            x = gsp(x, batch_index)
        elif self.pooling == 'mean/max':
            x = torch.cat([gmp(x, batch_index), 
                                gap(x, batch_index)], dim=1)
        
        x = self.dropout_layer(x)
        
        for layer in self.readout:
            x = F.leaky_relu(layer(x))

        x = self.output_layer(x)

        if self.n_classes == 1:
            x = x.float().squeeze()

        return x