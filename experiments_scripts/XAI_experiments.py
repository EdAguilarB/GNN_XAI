import json
import os
import sys

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


def run_XAI(opt):

    random.seed(opt.global_seed)

    # Load the dataset and model from experiment
    exp_dir = f"{opt.exp_name}/{opt.filename[:-4]}/{opt.network_name}/results_model"
    train_loader = DataLoader(
        torch.load("{}/train_loader.pth".format(exp_dir)).dataset.shuffle()[:10]
    )
    test_loader = DataLoader(
        torch.load("{}/test_loader.pth".format(exp_dir)).dataset.shuffle()
    )

    log_dir = f"{opt.exp_name}/{opt.filename[:-4]}/{opt.network_name}/results_XAI/{opt.XAI_algorithm}"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(f"{log_dir}/mols", exist_ok=True)

    # Calculate the attributions and generate JSON files if they do not exist for both the training and test set
    # if os.path.exists(f'{log_dir}/node_masks_train.json'):
    #    with open(f'{log_dir}/node_masks_train.json') as f:
    #        data_train = json.load(f)
    # else:
    data_train = calculate_attributions(
        opt=opt, loader=train_loader, XAI_algorithm=opt.XAI_algorithm
    )
    #    with open(f'{log_dir}/node_masks_train.json', 'w') as f:
    #        json.dump(data_train, f, indent=4)
    # if os.path.exists(f'{log_dir}/node_masks_test.json'):
    #    with open(f'{log_dir}/node_masks_test.json') as f:
    #        data_test = json.load(f)
    # else:
    data_test = calculate_attributions(
        opt=opt, loader=test_loader, XAI_algorithm=opt.XAI_algorithm, set="test"
    )
    #    with open(f'{log_dir}/node_masks_test.json', 'w') as f:
    #        json.dump(data_test, f, indent=4)

    # Calculate the best threshold for the test set by using the training set
    mol_attrs_train, max_min_vals, max_val = get_attrs_atoms(data_train, opt)
    train_smiles = get_smarts_mols(train_loader, opt)
    mol_attrs, _, _ = get_attrs_atoms(data_test, opt, max_min_vals, max_val)
    test_smiles = get_smarts_mols(test_loader, opt)

    if opt.XAI_mode == "evaluate":
        threshold_results = {}
        for threshold in range(0, 100, 5):
            accuracies_train = []
            mol_attrs_train = {
                key: mol_attrs_train[key]
                for key in train_smiles
                if key in mol_attrs_train
            }
            results_train = calculate_XAI_metrics(
                opt,
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
        ic(threshold_results)
        print(
            f"The best threshold is {best_threshold} with an accuracy of {threshold_results[best_threshold]}"
        )

        mol_attrs = {key: mol_attrs[key] for key in test_smiles if key in mol_attrs}
        # Calculate the metrics for the test set using the best threshold found
        results_test = calculate_XAI_metrics(
            opt,
            mol_attrs=mol_attrs,
            mol_smiles=test_smiles,
            threshold=best_threshold,
            save_results=True,
            logdir=f"{log_dir}",
        )

        create_XAI_report(opt, best_threshold, threshold_results, results_test, log_dir)

    elif opt.XAI_mode == "get_importance":
        find_hot_spots(
            opt,
            mol_attrs,
            test_smiles,
            threshold=0.5,
            save_results=True,
            logdir=f"{log_dir}/find_hot_spots",
        )
        find_substructures(
            opt,
            mol_attrs,
            test_smiles,
            threshold=0.5,
            save_results=True,
            logdir=f"{log_dir}/find_patterns",
        )


if __name__ == "__main__":
    # Ensure 'opt' is properly initialized with all necessary attributes
    opt = BaseOptions()
    opt = opt.parse()
    run_XAI(opt)
