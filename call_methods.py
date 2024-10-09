import argparse

def make_network(network_name: str, 
                 opt: argparse.Namespace, 
                 n_node_features: int, 
                 n_edge_features: int,
                 **kwargs):
    
    if network_name == "GCN":
        from models.gcn import GCN
        return GCN(opt=opt, n_node_features=n_node_features, n_edge_features = n_edge_features, **kwargs)
    
    elif network_name == "GAT":
        from models.gat import GAT
        return GAT(opt=opt, n_node_features=n_node_features, n_edge_features = n_edge_features, **kwargs)
    
    elif network_name == "graphsage":
        from models.graphsage import graphsage
        return graphsage(opt=opt, n_node_features=n_node_features, n_edge_features = n_edge_features, **kwargs)
    
    else:
        raise ValueError(f"Network {network_name} not implemented")
    