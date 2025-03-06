import json
import os
import sys
import warnings
from copy import deepcopy
from pathlib import Path

import torch
from icecream import ic
from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

from call_methods import make_network
from data.mol_instance import molecular_graph
from options.base_options import BaseOptions
from utils.model_utils import eval_network, network_report, train_network


def train_model(opt):

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    mols = molecular_graph(opt=opt, filename=opt.filename, root=opt.root)

    json_file = (
        Path(opt.exp_name)
        / opt.filename[:-4]
        / opt.network_name
        / "results_hyp_opt"
        / "best_hyperparameters.json"
    )

    with open(json_file) as f:
        hyp = json.load(f)

    print(f"Using hyperparameters: {hyp}")

    train_indices = [i for i, s in enumerate(mols.set) if s == "train"]
    stratified = mols.y[train_indices] if opt.problem_type == "classification" else None
    val_indices = [i for i, s in enumerate(mols.set) if s == "val"]
    if len(val_indices) == 0:
        train_indices, val_indices = train_test_split(
            train_indices,
            test_size=0.2,
            random_state=opt.global_seed,
            stratify=stratified,
        )
    test_indices = [i for i, s in enumerate(mols.set) if s == "test"]

    train_dataset = mols[train_indices]
    val_dataset = mols[val_indices]
    test_dataset = mols[test_indices]

    batch_size = hyp.pop("batch_size")
    epochs = hyp.pop("epochs")
    early_stop_patience = hyp.pop("early_stopping_patience")

    # Make the dataloaders
    train_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_set = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Make the network
    model = make_network(
        network_name=opt.network_name,
        opt=opt,
        n_node_features=mols.num_node_features,
        n_edge_features=mols.num_edge_features,
        **hyp,
    ).to(device)

    if model.kwargs:
        unused_params = list(model.kwargs.keys())
        warnings.warn(
            f"Not all hyperparameters have been used: {unused_params}", UserWarning
        )

    train_list, val_list, test_list, lr_list = [], [], [], []

    best_val_loss = float("inf")
    early_stopping_counter = 0

    for epoch in range(1, epochs + 1):

        lr = model.scheduler.optimizer.param_groups[0]["lr"]

        train_loss = train_network(model=model, train_loader=train_set, device=device)

        val_loss = eval_network(model=model, loader=val_set, device=device)

        test_loss = eval_network(model=model, loader=test_set, device=device)

        model.scheduler.step(val_loss)

        print(
            "Epoch {:03d} | LR: {:.5f} | Train loss: {:.3f} | Val loss: {:.3f} | Test loss: {:.3f}".format(
                epoch, lr, train_loss, val_loss, test_loss
            )
        )

        train_list.append(train_loss)
        val_list.append(val_loss)
        test_list.append(test_loss)
        lr_list.append(lr)

        if epoch % 5 == 0:

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = deepcopy(model.state_dict())
                early_stopping_counter = 0
                print(f"Best model saved at epoch {epoch:03d}")
                print(f"Validation loss: {best_val_loss:.3f}")

            else:
                early_stopping_counter += 1
                print(f"Early stopping counter: {early_stopping_counter}")
                if early_stopping_counter >= early_stop_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

    model.load_state_dict(best_model_state)
    network_report(
        exp_name=Path(opt.exp_name) / mols.filename[:-4],
        loaders=(train_set, val_set, test_set),
        loss_lists=(train_list, val_list, test_list, lr_list),
        save_all=True,
        model=model,
    )


if __name__ == "__main__":
    # Ensure 'opt' is properly initialized with all necessary attributes
    opt = BaseOptions()
    opt = opt.parse()
    train_model(opt)
