import torch
from ray import tune
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

from call_methods import (make_loss, make_network, make_optimizer,
                          make_scheduler)
from data.mol_instance import molecular_graph
from utils.model_utils import eval_network, train_network


def train_model_ray(
    config,
    filename,
    root,
    mol_cols,
    set_col,
    target_variable,
    id_col,
    problem_type,
    n_classes,
    global_seed,
    network_name,
    optimizer,
    scheduler,
):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    mols = molecular_graph(
        filename=filename,
        root=f"../../../../../../../{root}",
        mol_cols=mol_cols,
        set_col=set_col,
        target_variable=target_variable,
        id_col=id_col,
    )

    # Split dataset indices
    train_indices = [i for i, s in enumerate(mols.set) if s != "train"]
    val_indices = [i for i, s in enumerate(mols.set) if s != "val"]

    # Define stratification target once based on the problem type
    stratified = mols.y[train_indices] if problem_type == "classification" else None

    # Further split train_indices into train and validation indices
    if len(val_indices) == 0:
        train_indices, test_indices = train_test_split(
            train_indices,
            test_size=0.2,
            random_state=global_seed,
            stratify=stratified,
        )

        # Re-define stratified target for remaining train_indices if classification
        stratified = mols.y[train_indices] if problem_type == "classification" else None

        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=0.2,
            random_state=global_seed,
            stratify=stratified,
        )

    else:
        train_indices, test_indices = train_test_split(
            train_indices,
            test_size=0.2,
            random_state=global_seed,
            stratify=stratified,
        )

    # Create datasets
    train_dataset = mols[train_indices]
    val_dataset = mols[val_indices]
    test_dataset = mols[test_indices]

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=int(config["batch_size"]), shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=int(config["batch_size"]), shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=int(config["batch_size"]), shuffle=False
    )

    # Initialize the model with hyperparameters from config
    model = make_network(
        network_name=network_name,
        n_node_features=mols.num_node_features,
        n_edge_features=mols.num_edge_features,
        n_classes=n_classes,
        problem_type=problem_type,
        global_seed=global_seed,
        **config,
    ).to(device)

    loss_fn = make_loss(problem_type)
    optimizer = make_optimizer(optimizer, model, config["lr"])
    scheduler = make_scheduler(
        scheduler, optimizer, config["step_size"], config["gamma"], config["min_lr"]
    )

    best_val_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(1, config["epochs"] + 1):
        # Training step
        train_loss = train_network(model, train_loader, device, loss_fn, optimizer)
        val_loss = eval_network(model, val_loader, device, loss_fn)
        test_loss = eval_network(model, test_loader, device, loss_fn)

        scheduler.step(val_loss)

        if epoch % 5 == 0:
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                tune.report(test_loss=test_loss, epoch=epoch)
                early_stopping_counter = 0

            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config["early_stopping_patience"]:
                    print(f"Early stopping at epoch {epoch}")
                    break
