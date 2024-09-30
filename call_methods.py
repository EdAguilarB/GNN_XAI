import argparse
from copy import copy, deepcopy
from torch_geometric.loader import DataLoader



def make_network(network_name: str, opt: argparse.Namespace, n_node_features: int, n_edge_features: int = None):
    if network_name == "GCN":
        from models.gcn import GCN
        return GCN(opt=opt, n_node_features=n_node_features)
    elif network_name == "GAT":
        from models.gat import GAT
        return GAT(opt=opt, n_node_features=n_node_features, n_edge_features=n_edge_features)
    elif network_name == "graphsage":
        from models.graphsage import graphsage
        return graphsage(opt=opt, n_node_features=n_node_features,)
    else:
        raise ValueError(f"Network {network_name} not implemented")
    