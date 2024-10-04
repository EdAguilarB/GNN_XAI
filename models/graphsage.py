import torch
import torch.nn as nn
from models.networks import BaseNetwork
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp, SAGEConv
import argparse
from tqdm import tqdm
from icecream import ic

class graphsage(BaseNetwork):

    def __init__(self, opt: argparse.Namespace, n_node_features:int,):
        super().__init__(opt=opt, n_node_features=n_node_features,)

        self._name = "graphsage"

        # linear expansion of node features
        self.linear = nn.Linear(n_node_features, self.embedding_dim)

        #Convolutions
        self.conv_layers = nn.ModuleList([])
        for _ in range(self.n_convolutions):
            self.conv_layers.append(SAGEConv(in_channels=self.embedding_dim, 
                                            out_channels=self.embedding_dim,
                                            ))

        #graph embedding is the concatenation of the global mean and max pooling, thus 2*embedding_dim
        graph_embedding = self.embedding_dim*2

        #Readout layers
        self.readout = nn.ModuleList([])

        for _ in range(self.readout_layers-1):
            reduced_dim = int(graph_embedding/2)
            self.readout.append(nn.Sequential(nn.Linear(graph_embedding, reduced_dim), 
                                              nn.ReLU()))
            graph_embedding = reduced_dim

        #Final readout layer
        self.readout.append(nn.Linear(graph_embedding, self._n_classes))
        
        
        self._make_loss()
        self._make_optimizer(opt.optimizer, opt.lr)
        self._make_scheduler(scheduler=opt.scheduler, step_size = opt.step_size, gamma = opt.gamma, min_lr=opt.min_lr)
        

    def forward(self,x=None, edge_index=None,  batch_index=None, edge_attr=None):

        x = self.linear(x)
        x = nn.ReLU()(x)

        for i in range(self.n_convolutions):
            x = self.conv_layers[i](x=x, edge_index=edge_index)
            x = nn.ReLU()(x)

        x = torch.cat([gmp(x, batch_index), 
                       gap(x, batch_index)], dim=1)
        
        for i in range(self.readout_layers):
            x = self.readout[i](x)

    
        return x