import json
import os
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)

import random

from icecream import ic

from options.base_options import BaseOptions
from utils.XAI_utils import (calculate_attributions, calculate_XAI_metrics,
                             create_XAI_report, find_hot_spots,
                             find_substructures, get_attrs_atoms,
                             get_smarts_mols)


def run_XAI(
    global_seed,
    exp_name,
    filename,
    network_name,
    XAI_algorithm,
    XAI_mode,
    XAI_attrs_mode,
):

    random.seed(global_seed)

    exp_path = Path(exp_name) / filename[:-4] / network_name

    # Load the dataset and model from experiment

    exp_dir = exp_path / "results_model"

    train_loader = DataLoader(
        torch.load(exp_dir / "train_loader.pth").dataset.shuffle()[:10]
    )
    test_loader = DataLoader(torch.load(exp_dir / "test_loader.pth").dataset.shuffle())

    log_dir = exp_path / "results_XAI" / XAI_algorithm
    log_dir.mkdir(parents=True, exist_ok=True)

    mol_plot_dir = log_dir / "mols"
    mol_plot_dir.mkdir(parents=True, exist_ok=True)

    # Calculate the attributions for the training and test set
    data_train = calculate_attributions(
        exp_path=exp_path, loader=train_loader, XAI_algorithm=XAI_algorithm
    )
    data_test = calculate_attributions(
        exp_path=exp_path, loader=test_loader, XAI_algorithm=XAI_algorithm, set="test"
    )

    # Calculate the best threshold for the test set by using the training set
    mol_attrs_train, max_min_vals, max_val = get_attrs_atoms(
        data=data_train, XAI_attrs_mode=XAI_attrs_mode
    )
    train_smiles = get_smarts_mols(train_loader, XAI_mode)
    mol_attrs, _, _ = get_attrs_atoms(
        data=data_test,
        XAI_attrs_mode=XAI_attrs_mode,
        max_min_vals=max_min_vals,
        max_atom_val=max_val,
    )
    test_smiles = get_smarts_mols(test_loader, XAI_mode)

    if XAI_mode == "evaluate":
        threshold_results = {}
        for threshold in range(0, 100, 5):
            accuracies_train = []
            mol_attrs_train = {
                key: mol_attrs_train[key]
                for key in train_smiles
                if key in mol_attrs_train
            }
            results_train = calculate_XAI_metrics(
                experiment_dir=exp_path,
                mol_attrs=mol_attrs_train,
                mol_smiles=train_smiles,
                threshold=threshold / 100,
                save_results=False,
            )
            for _, value in results_train.items():
                accuracies_train.append(value["Accuracy"])
            mean_accuracy = sum(accuracies_train) / len(accuracies_train)
            threshold_results[threshold / 100] = mean_accuracy
        best_threshold = max(threshold_results, key=threshold_results.get)
        print(
            f"The best threshold is {best_threshold} with an accuracy of {threshold_results[best_threshold]}"
        )

        mol_attrs = {key: mol_attrs[key] for key in test_smiles if key in mol_attrs}
        # Calculate the metrics for the test set using the best threshold found
        results_test = calculate_XAI_metrics(
            experiment_dir=exp_path,
            mol_attrs=mol_attrs,
            mol_smiles=test_smiles,
            threshold=best_threshold,
            save_results=True,
            logdir=log_dir,
            XAI_attrs_mode=XAI_attrs_mode,
        )

        create_XAI_report(
            best_threshold, threshold_results, results_test, log_dir, XAI_attrs_mode
        )

    find_hot_spots(
        mol_attrs,
        test_smiles,
        XAI_attrs_mode,
        threshold=0.5,
        save_results=True,
        logdir=log_dir / "find_hot_spots",
    )

    find_substructures(
        mol_attrs,
        test_smiles,
        XAI_attrs_mode,
        threshold=0.5,
        save_results=True,
        logdir=f"{log_dir}/find_patterns",
    )


if __name__ == "__main__":
    # Ensure 'opt' is properly initialized with all necessary attributes
    opt = BaseOptions()
    opt = opt.parse()
    run_XAI(opt)
