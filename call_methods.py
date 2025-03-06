import argparse

import torch


def make_network(
    network_name: str,
    n_node_features: int,
    n_edge_features: int,
    n_classes: int,
    problem_type: str,
    global_seed: int,
    optimizer: str,
    scheduler: str,
    **kwargs,
):

    if network_name.lower() == "GCN".lower():
        from models.gcn import GCN

        return GCN(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_classes=n_classes,
            problem_type=problem_type,
            global_seed=global_seed,
            **kwargs,
        )

    elif network_name.lower() == "GAT".lower():
        from models.gat import GAT

        return GAT(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_classes=n_classes,
            problem_type=problem_type,
            global_seed=global_seed,
            **kwargs,
        )

    elif network_name.lower() == "graphsage":
        from models.graphsage import graphsage

        return graphsage(
            n_node_features=n_node_features,
            n_edge_features=n_edge_features,
            n_classes=n_classes,
            problem_type=problem_type,
            global_seed=global_seed,
            **kwargs,
        )

    else:
        raise ValueError(f"Network {network_name} not implemented")


def make_loss(problem_type: str):
    if problem_type.lower() == "classification":
        return torch.nn.CrossEntropyLoss()
    elif problem_type.lower() == "regression":
        return torch.nn.MSELoss()
    else:
        raise ValueError(f"Problem type {problem_type} not supported")


def make_optimizer(optimizer, model, lr):
    if optimizer.lower() == "Adam".lower():
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer.lower() == "SGD".lower():
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif optimizer.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    else:
        raise NotImplementedError(f"Optimizer type {optimizer} not implemented")
    return optimizer


def make_scheduler(scheduler, optimizer, step_size, gamma, min_lr):
    if scheduler.lower() == "StepLR".lower():
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma, min_lr=min_lr
        )
    elif scheduler.lower() == "ReduceLROnPlateau".lower():
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.1, patience=10, verbose=True, min_lr=min_lr
        )
    else:
        raise NotImplementedError(f"Scheduler type {scheduler} not implemented")
    return scheduler
