import ast
import json
import os
import statistics
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from icecream import ic
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Draw import SimilarityMaps
from rdkit.Chem.PandasTools import AddMoleculeColumnToFrame
from scipy.stats import rankdata
from torch_geometric.explain import CaptumExplainer, Explainer, GNNExplainer
from torch_geometric.explain.metric import groundtruth_metrics
from tqdm import tqdm

from utils.plot_utils import (create_mol_plot2,
                              draw_molecule_with_similarity_map)


def calculate_attributions(opt, loader, XAI_algorithm="all", set="train"):
    exp_dir = f"{opt.exp_name}/{opt.filename[:-4]}/{opt.network_name}/results_model"

    device = torch.device("cpu")

    # Load the trained model
    model = torch.load(
        "{}/model.pth".format(exp_dir), map_location=torch.device("cpu")
    ).to(device)

    # Define available algorithms and their names
    algorithms_dict = {
        "GNNExplainer": GNNExplainer(),
        "IntegratedGradients": CaptumExplainer("IntegratedGradients"),
        "Saliency": CaptumExplainer("Saliency"),
        "InputXGradient": CaptumExplainer("InputXGradient"),
        "Deconvolution": CaptumExplainer("Deconvolution"),
        #'ShapleyValueSampling': CaptumExplainer('ShapleyValueSampling'),
        "GuidedBackprop": CaptumExplainer("GuidedBackprop"),
    }

    # Select algorithms based on XAI_algorithm parameter
    if XAI_algorithm == "all":
        algorithms = list(algorithms_dict.values())
        algorithms_names = list(algorithms_dict.keys())
    elif XAI_algorithm in algorithms_dict:
        algorithms = [algorithms_dict[XAI_algorithm]]
        algorithms_names = [XAI_algorithm]
    else:
        raise ValueError(
            f"XAI_algorithm '{XAI_algorithm}' is not recognized. Available options are: {list(algorithms_dict.keys())} and 'all'."
        )

    node_masks = {}

    # Determine the problem type for the model
    problem_type = (
        "multiclass_classification"
        if model.problem_type == "classification"
        else "regression"
    )

    for name, algorithm in zip(algorithms_names, algorithms):

        print(f"Runing {name} analysis")

        algo_dict = {}
        algo_path = (
            f"{opt.exp_name}/{opt.filename[:-4]}/{opt.network_name}/results_XAI/{name}"
        )
        # Reload the model for each algorithm
        # Initialize the explainer
        explainer = Explainer(
            model=model,
            algorithm=algorithm,
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode=problem_type,
                task_level="graph",
                return_type="raw",
            ),
        )

        if os.path.exists(f"{algo_path}/node_masks_{set}_raw.json"):
            print("Loading previously calculated node masks")
            with open(f"{algo_path}/node_masks_{set}_raw.json") as f:
                attrs_mols = json.load(f)
        else:
            attrs_mols = None
            os.makedirs(algo_path, exist_ok=True)

        for mol in tqdm(loader):
            model = torch.load(
                f"{exp_dir}/model.pth", map_location=torch.device("cpu")
            ).to(device)
            mol.to(device)
            smiles = mol.smiles[0]
            smarts = mol.smarts[0]
            idx = mol.idx[0]

            # Generate explanations
            if attrs_mols is not None:
                if idx in attrs_mols:
                    node_mask = torch.tensor(attrs_mols[idx])
                else:
                    explanation = explainer(
                        x=mol.x,
                        edge_index=mol.edge_index,
                    )
                    node_mask = explanation.node_mask
            else:
                explanation = explainer(
                    x=mol.x,
                    edge_index=mol.edge_index,
                )
                node_mask = explanation.node_mask

            # Store individual algorithm results
            mol_key = idx
            if mol_key not in node_masks:
                node_masks[mol_key] = {}
            node_masks[mol_key][name] = node_mask.detach().numpy().tolist()
            algo_dict[mol_key] = node_mask.detach().numpy().tolist()

        with open(f"{algo_path}/node_masks_{set}_raw.json", "w") as f:
            json.dump(algo_dict, f, indent=4)

    return node_masks


def get_attrs_atoms(data, opt, max_min_vals=None, max_atom_val=None):

    if not max_min_vals:
        max_min_vals = {}
        for outer_key, outer_value in data.items():
            for inner_key, inner_value in outer_value.items():
                max_val = max(max(inner_value))
                min_val = min(min(inner_value))

                if inner_key not in max_min_vals:
                    max_min_vals[inner_key] = {}
                    max_min_vals[inner_key]["max"] = max_val
                    max_min_vals[inner_key]["min"] = min_val

                elif max_val > max_min_vals[inner_key]["max"]:
                    max_min_vals[inner_key]["max"] = max_val

                elif min_val < max_min_vals[inner_key]["min"]:
                    max_min_vals[inner_key]["min"] = min_val

    norm_max = 0

    mol_attrs = {}

    for (
        outer_key,
        outer_value,
    ) in data.items():  # over dictionary containing the mol id and the attributions

        dir, mag = None, None  # initialize variables to store the attributions

        for (
            inner_key,
            inner_value,
        ) in (
            outer_value.items()
        ):  # over the dictionary containing the different XAI algorithms and their attributions

            max_val = max(
                max_min_vals[inner_key]["max"], abs(max_min_vals[inner_key]["min"])
            )
            value = np.array(
                inner_value
            )  # convert the list of attributions to a numpy array
            value /= max_val  # normalize the attributions to vals between -1 and 1 (0 and 1 for the GNNExplainer and Saliency)

            if inner_key == "GNNExplainer" or inner_key == "Saliency":
                if not isinstance(mag, np.ndarray):
                    mag = value  # if mag is None, assign the value to mag
                else:
                    mag += value  # if mag is not None, add the value to mag
            else:
                if not isinstance(dir, np.ndarray):
                    dir = value  # if dir is None, assign the value to dir
                else:
                    dir += value  # if dir is not None, add the value to dir

        if dir is not None:  # for cases where only positive values are present

            if opt.XAI_attrs_mode == "directional":
                # this will get the direction that each feature is pointing to. Then,
                # get the absolute values of importance and add them to the attributions
                # of the GNNExplainer and Saliency. Lastly, multiply the attributions by
                # the sign in the original attributions to get the direction of the attributions

                sign = np.sign(dir)  # get the sign of the attributions
                dir = np.abs(dir)  # get the magnitud of the attributions

                mag = (
                    mag if mag is not None else 0
                ) + dir  # add the magnitud of the attributions to the attributions of the GNNExplainer and Saliency
                mag = (
                    mag * sign
                )  # multiply the attributions by the sign to get the direction of the attributions

            elif opt.XAI_attrs_mode == "absolute":
                dir = np.where(dir < 0, 0, dir)  # set the negative attributions to 0
                # dir = np.abs(dir)
                mag = (
                    mag if mag is not None else 0
                ) + dir  # add the magnitud of the attributions to the attributions of the GNNExplainer and Saliency

        # TODO - check if the sum is the best way to combine the attributions
        # TODO - separate the scores in different families of node features eg. atom type, atom degree, etc.
        mag = np.sum(
            mag, axis=1
        )  # sum over the columns to get the overall importance of each atom

        if max_atom_val is None:
            if np.max(np.abs(mag)) > norm_max:
                norm_max = np.max(np.abs(mag))

        # mag /= np.max(np.abs(mag)) # normalize the importance values from 0 to 1

        mol_attrs[outer_key] = (
            mag  # key contains the mol id, value contains the importance of each atom
        )

    if max_atom_val is None:
        max_atom_val = norm_max
    for key, value in mol_attrs.items():
        mol_attrs[key] = value / max_atom_val

    return mol_attrs, max_min_vals, max_atom_val


def get_smarts_mols(loader, opt):

    mol_smarts = {}

    if opt.XAI_mode == "evaluate":
        for mol in loader:
            if isinstance(mol.smarts[0], list):
                mol_smarts[mol.idx[0]] = {
                    "smiles": mol.smiles[0],
                    "smarts": mol.smarts[0],
                }
            else:
                continue
    elif opt.XAI_mode == "get_importance":
        for mol in loader:
            mol_smarts[mol.idx[0]] = {"smiles": mol.smiles[0]}

    return mol_smarts


def calculate_XAI_metrics(
    opt, mol_attrs, mol_smiles, threshold=0.5, save_results=True, logdir=None
):
    # Calculate the metrics for the XAI

    if logdir:
        exp_dir = f"{opt.exp_name}/{opt.filename[:-4]}/{opt.network_name}/results_model"
        preds = pd.read_csv(f"{exp_dir}/predictions_test_set.csv", index_col=0)
        preds["index"] = preds["index"].astype(str)
        os.makedirs(
            os.path.join(logdir, "mols", "low_acc", "correct_pred"), exist_ok=True
        )
        os.makedirs(
            os.path.join(logdir, "mols", "low_acc", "incorrect_pred", "false_negative"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(logdir, "mols", "low_acc", "incorrect_pred", "false_positive"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(logdir, "mols", "high_acc", "correct_pred"), exist_ok=True
        )
        os.makedirs(
            os.path.join(
                logdir, "mols", "high_acc", "incorrect_pred", "false_negative"
            ),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(
                logdir, "mols", "high_acc", "incorrect_pred", "false_positive"
            ),
            exist_ok=True,
        )

    acc_mols = {}

    for idx, attrs in mol_attrs.items():
        pred_imp = []

        smarts_str = mol_smiles[idx]["smarts"]
        smiles = mol_smiles[idx]["smiles"]
        smarts_list = smarts_str

        mol = Chem.MolFromSmiles(smiles)

        indexes = set()

        for smarts in smarts_list:

            pattern = Chem.MolFromSmarts(smarts)
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                indexes.update(set().union(*matches))

        assert mol.GetNumAtoms() == len(
            attrs
        ), f"Num atoms: {mol.GetNumAtoms()} does not match the number of attributes: {len(attrs)}"

        acc = 0

        p = len(indexes)
        n = mol.GetNumAtoms() - p

        ground_truth = np.zeros(mol.GetNumAtoms())
        ground_truth[list(indexes)] = 1

        if p == 0 or n == 0:
            # Cannot compute AUROC if there are no positive or negative samples
            auroc = None
        else:
            # scores ascending ranking
            scores = [score for score in attrs]

            # Compute ranks, handling ties by assigning average ranks
            ranks = rankdata(scores, method="average")

            # compute ranks of positive samples
            sum_positive_ranks = sum([ranks[i] for i in indexes])

            # calculate auroc
            auroc = (sum_positive_ranks - p * (p + 1) / 2) / (p * n)

        for i, attr in enumerate(attrs):
            if (attr >= threshold and i in indexes) or (
                attr < threshold and i not in indexes
            ):
                acc += 1
            if attr >= threshold:
                pred_imp.append(i)

        acc = acc / len(attrs)
        acc_mols[idx] = {"Accuracy": acc, "Auroc": auroc}

        # accuracy, recall, precision, f1, auroc2 = groundtruth_metrics(pred_mask=torch.tensor(attrs), target_mask=torch.tensor(ground_truth), threshold=threshold)

        if save_results:

            if acc < 0.6:
                print(f"Mol {idx} being plot because of low accuracy")
                acc_dir = "low_acc"
            elif acc > 0.9:
                print(f"Mol {idx} being plot because of high accuracy")
                acc_dir = "high_acc"
            else:
                print(f"Mol {idx} not being plot")
                continue

            mol_y = preds.loc[preds["index"] == idx]
            y_true = mol_y.iloc[0, 0]
            y_pred = mol_y.iloc[0, 1]

            if y_true == y_pred:
                pred_dir = "correct_pred"
            else:
                if y_true == 1 or y_true == 2:
                    pred_dir = "incorrect_pred/false_negative"
                elif y_pred == 1 or y_pred == 2:
                    pred_dir = "incorrect_pred/false_positive"

            mol_plot_dir = f"{logdir}/mols/{acc_dir}/{pred_dir}/{idx}.png"
            create_mol_plot2(smiles, pred_imp, indexes, mol_plot_dir)

    if save_results:
        with open(f"{logdir}/metrics_{opt.XAI_attrs_mode}.json", "w") as f:
            json.dump(acc_mols, f, indent=4)

    return acc_mols


def create_XAI_report(opt, best_threshold, threshold_results, results_test, exp_dir):
    report_lines = []

    # Experiment Summary
    report_lines.append(f"Experiment Report: {opt.network_name} Evaluation\n")
    report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Model: {opt.network_name}")
    report_lines.append(f"Dataset: {opt.filename}")
    report_lines.append(f"Random Seed: {opt.global_seed}\n")

    # Threshold Optimization
    report_lines.append("Threshold Optimization:")
    report_lines.append(f"- Thresholds Tested: 0.00 to 0.95 (step size: 0.05)")
    report_lines.append(f"- Best Threshold: {best_threshold}")
    report_lines.append(
        f"- Mean Training Accuracy at Best Threshold: {threshold_results[best_threshold]:.2f}\n"
    )

    # Test Set Evaluation
    accuracies_test = [value["Accuracy"] for value in results_test.values()]
    mean_accuracy_test = sum(accuracies_test) / len(accuracies_test)
    if len(accuracies_test) > 1:
        std_accuracy_test = statistics.stdev(accuracies_test)
    else:
        std_accuracy_test = 0.0  # Standard deviation is zero if only one data point

    auroc_scores_test = [
        value["Auroc"] for value in results_test.values() if value["Auroc"] is not None
    ]
    if len(auroc_scores_test) > 1:
        mean_auroc_test = sum(auroc_scores_test) / len(auroc_scores_test)
        std_auroc_test = statistics.stdev(auroc_scores_test)

    else:
        mean_auroc_test = 0.0
        std_auroc_test = 0.0

    report_lines.append(f"Test Set Evaluation at Threshold {best_threshold}:")
    report_lines.append(
        f"- Mean Accuracy: {mean_accuracy_test:.2f} (±{std_accuracy_test:.2f})"
    )
    report_lines.append(
        f"- Mean AUROC: {mean_auroc_test:.2f} (±{std_auroc_test:.2f})\n"
    )

    # Write report to file
    report_path = os.path.join(exp_dir, f"experiment_report_{opt.XAI_attrs_mode}.txt")
    with open(report_path, "w") as report_file:
        report_file.write("\n".join(report_lines))

    print(f"Experiment report saved to {report_path}")


def find_hot_spots(
    opt, mol_attrs, mol_smiles, threshold=0.5, save_results=True, logdir=None
):
    """
    Find the most frequent SMARTS patterns identified as important based on attributions.

    Args:
        opt: Configuration options for experiment and filenames.
        mol_attrs: Dictionary with molecule IDs as keys and atom-level importance scores as values.
        mol_smiles: Dictionary with molecule IDs as keys and SMILES strings as values.
        threshold: Float value to classify important atoms.
        save_results: Boolean to save the frequent SMARTS results.
        logdir: Directory to save results.

    Returns:
        frequent_smarts: Dictionary of SMARTS patterns and their frequencies.
    """
    if logdir:
        os.makedirs(logdir, exist_ok=True)

    # Dictionary to count SMARTS occurrences
    smarts_counts = {}

    for idx, attrs in mol_attrs.items():
        smiles = mol_smiles[idx]["smiles"]
        mol = Chem.MolFromSmiles(smiles)

        assert mol.GetNumAtoms() == len(
            attrs
        ), f"Num atoms: {mol.GetNumAtoms()} does not match the number of attributes: {len(attrs)}"

        # Identify important atoms based on threshold
        important_atoms = [i for i, attr in enumerate(attrs) if attr >= threshold]

        if not important_atoms:
            continue  # Skip molecules without important atoms

        # Generate all substructures (SMARTS) for the molecule
        for atom_idx in important_atoms:
            for radius in range(1, 3):  # Explore increasing neighborhood sizes
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx, True)

                if not env:
                    continue

                atoms = set()
                for bond_idx in env:
                    bond = mol.GetBondWithIdx(bond_idx)
                    atoms.add(bond.GetBeginAtomIdx())
                    atoms.add(bond.GetEndAtomIdx())

                smarts = Chem.MolFragmentToSmarts(mol, atomsToUse=atoms)

                if smarts:
                    smarts_counts[smarts] = smarts_counts.get(smarts, 0) + 1

    # Sort SMARTS patterns by frequency
    frequent_smarts = dict(
        sorted(smarts_counts.items(), key=lambda x: x[1], reverse=True)
    )

    # Save results if required
    if save_results and logdir:
        df = pd.DataFrame(
            list(frequent_smarts.items()), columns=["smarts", "Frequency"]
        )

        df = reduce_df(df)
        df.to_csv(
            os.path.join(logdir, f"frequent_smarts_{opt.XAI_attrs_mode}.csv"),
            index=False,
        )

    return frequent_smarts


def find_substructures(
    opt, mol_attrs, mol_smiles, threshold=0.5, save_results=True, logdir=None
):
    """
    Find the most frequent SMARTS patterns identified as important based on connected fragments of important atoms.

    Args:
        opt: Configuration options for experiment and filenames.
        mol_attrs: Dictionary with molecule IDs as keys and atom-level importance scores as values.
        mol_smiles: Dictionary with molecule IDs as keys and SMILES strings as values.
        threshold: Float value to classify important atoms.
        save_results: Boolean to save the frequent SMARTS results.
        logdir: Directory to save results.

    Returns:
        frequent_smarts: Dictionary of SMARTS patterns and their frequencies.
    """
    if logdir:
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(os.path.join(logdir, "mols"), exist_ok=True)

    # Dictionary to count SMARTS occurrences
    smarts_counts = {}

    for idx, attrs in mol_attrs.items():
        smiles = mol_smiles[idx]["smiles"]
        mol = Chem.MolFromSmiles(smiles)

        assert mol.GetNumAtoms() == len(
            attrs
        ), f"Num atoms: {mol.GetNumAtoms()} does not match the number of attributes: {len(attrs)}"

        # weights = np.where(np.array(attrs) >= threshold, attrs, 0)
        draw_molecule_with_similarity_map(
            smiles,
            attrs,
            save_path=os.path.join(logdir, "mols", f"{idx}_importance.png"),
        )

        # Identify important atoms based on threshold
        important_atoms = [i for i, attr in enumerate(attrs) if attr >= threshold]

        if not important_atoms:
            continue  # Skip molecules without important atoms

        # Generate substructures based on connected fragments of important atoms
        atom_set = set(important_atoms)
        fragments = []

        # Identify connected fragments
        for bond in mol.GetBonds():
            begin, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if begin in atom_set and end in atom_set:
                # Merge connected fragments
                merged = False
                for frag in fragments:
                    if begin in frag or end in frag:
                        frag.update([begin, end])
                        merged = True
                        break
                if not merged:
                    fragments.append({begin, end})

        # Add any isolated important atoms as individual fragments
        # for atom in atom_set:
        #    if not any(atom in frag for frag in fragments):
        #        fragments.append({atom})

        # Generate SMARTS for each fragment

        if not fragments:
            continue

        for frag in fragments:

            smarts = Chem.MolFragmentToSmarts(
                mol,
                atomsToUse=list(frag),
            )

            if smarts:
                smarts_counts[smarts] = smarts_counts.get(smarts, 0) + 1

    # Sort SMARTS patterns by frequency
    frequent_smarts = dict(
        sorted(smarts_counts.items(), key=lambda x: x[1], reverse=True)
    )

    # Save results if required
    if save_results and logdir:
        df = pd.DataFrame(
            list(frequent_smarts.items()), columns=["smarts", "Frequency"]
        )
        df = reduce_df(df)

        df.to_csv(
            os.path.join(logdir, f"frequent_smarts_{opt.XAI_attrs_mode}.csv"),
            index=False,
        )

    return frequent_smarts


def are_smarts_equivalent(smarts1, smarts2):
    mol1 = Chem.MolFromSmarts(smarts1)
    mol2 = Chem.MolFromSmarts(smarts2)

    if mol1 is None or mol2 is None:
        raise ValueError("Invalid SMARTS string")

    # Check if mol1 matches mol2 and vice versa
    try:
        return mol1.HasSubstructMatch(mol2) and mol2.HasSubstructMatch(mol1)
    except:
        return False


def reduce_df(df):
    """
    Reduce the DataFrame by finding equivalent SMARTS strings and grouping them.
    """
    # This will store canonical (equivalent) SMARTS
    smarts_dict = {}

    # Loop over each SMARTS in the DataFrame
    for smart in df["smarts"]:
        matched = False

        # Iterate over existing SMARTS and check for equivalence
        for canonical in smarts_dict:
            if are_smarts_equivalent(smart, canonical):
                # If equivalent, increment count and mark matched
                smarts_dict[canonical] += df.loc[
                    df["smarts"] == smart, "Frequency"
                ].sum()
                matched = True
                break

        if not matched:
            # If no match found, add new canonical SMARTS to the dictionary
            smarts_dict[smart] = df.loc[df["smarts"] == smart, "Frequency"].sum()

    # Create a new DataFrame from the reduced SMARTS
    reduced_df = pd.DataFrame(
        list(smarts_dict.items()), columns=["smarts", "Frequency"]
    )

    return reduced_df
