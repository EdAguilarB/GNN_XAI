import torch
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
from tqdm import tqdm
import json
import os
import numpy as np
from rdkit import Chem
from scipy.stats import rankdata
from utils.plot_utils import create_mol_plot2
from datetime import datetime
import statistics
import ast
from icecream import ic

def calculate_attributions(opt, loader, XAI_algorithm='all'):
    exp_dir = f'{opt.exp_name}/{opt.filename[:-4]}/{opt.network_name}/results_model'

    # Load the trained model
    model = torch.load("{}/model.pth".format(exp_dir))

    # Define available algorithms and their names
    algorithms_dict = {
        'GNNExplainer': GNNExplainer(), 
        'IntegratedGradients': CaptumExplainer('IntegratedGradients'), 
        'Saliency': CaptumExplainer('Saliency'),
        'InputXGradient': CaptumExplainer('InputXGradient'),
        'Deconvolution': CaptumExplainer('Deconvolution'),
        #'ShapleyValueSampling': CaptumExplainer('ShapleyValueSampling'),
        'GuidedBackprop': CaptumExplainer('GuidedBackprop'),
    }

    # Select algorithms based on XAI_algorithm parameter
    if XAI_algorithm == 'all':
        algorithms = list(algorithms_dict.values())
        algorithms_names = list(algorithms_dict.keys())
    elif XAI_algorithm in algorithms_dict:
        algorithms = [algorithms_dict[XAI_algorithm]]
        algorithms_names = [XAI_algorithm]
    else:
        raise ValueError(f"XAI_algorithm '{XAI_algorithm}' is not recognized. Available options are: {list(algorithms_dict.keys())} and 'all'.")

    node_masks = {}

    # Determine the problem type for the model
    if model.problem_type == 'classification':
        problem_type = 'multiclass_classification'
    else:
        problem_type = 'regression'
        
    for mol in tqdm(loader):
        smiles = mol.smiles[0]
        smarts = mol.smarts[0]
        idx = mol.idx[0]
        algo_dict = {}

        if smarts == '':
            print(f'Mol {idx} does not have a ground-truth pattern associated. Skipping analysis.')
            continue

        for name, algorithm in zip(algorithms_names, algorithms):
            # Reload the model for each algorithm if necessary
            model = torch.load("{}/model.pth".format(exp_dir))
            # Initialize the explainer
            explainer = Explainer(
                model=model,
                algorithm=algorithm,
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode=problem_type,
                    task_level='graph',
                    return_type='raw',
                ),
            )

            # Generate explanations
            explanation = explainer(x=mol.x, edge_index=mol.edge_index)
            node_mask = explanation.node_mask

            # Store individual algorithm results
            algo_dict[name] = node_mask.detach().numpy().tolist()
        
        # Store the results in the same JSON structure as before
        node_masks[f'{idx}|{smiles}|{smarts}'] = algo_dict

    return node_masks


def get_attrs_atoms(data,):

    
    mol_attrs = {}

    for outer_key, outer_value in data.items(): # over dictionary containing the mol id and the attributions

        dir, mag = None, None # initialize variables to store the attributions

        for inner_key, inner_value in outer_value.items(): # over the dictionary containing the different XAI algorithms and their attributions
            value = np.array(inner_value) # convert the list of attributions to a numpy array
            value /= np.abs(np.max(value)) # normalize the attributions to vals between -1 and 1 (0 and 1 for the GNNExplainer and Saliency)
            if inner_key == 'GNNExplainer' or inner_key == 'Saliency':
                if not isinstance(mag, np.ndarray):
                    mag = value # if mag is None, assign the value to mag
                else:
                    mag += value # if mag is not None, add the value to mag
            else:
                if not isinstance(dir, np.ndarray):
                    dir = value # if dir is None, assign the value to dir
                else:
                    dir += value # if dir is not None, add the value to dir
        
        sign = np.sign(dir) # get the sign of the attributions
        dir = np.abs(dir) # get the magnitud of the attributions
        mag = dir + mag # add the magnitud of the attributions to the attributions of the GNNExplainer and Saliency
        # TODO - check if the sum is the best way to combine the attributions
        # TODO - separate the scores in different families of node features eg. atom type, atom degree, etc.
        mag = np.sum(mag, axis=1) # sum over the columns to get the overall importance of each atom
        mag /= np.max(mag) # normalize the importance values from 0 to 1

        mol_attrs[outer_key] = mag # key contains the mol id, smiles and smarts, value contains the importance of each atom

    return mol_attrs

def calculate_XAI_metrics(opt, mol_attrs, exp_dir, threshold = 0.5, save_results = True, logdir = None):
    # Calculate the metrics for the XAI

    acc_mols = {}
    
    for smiles, attrs in mol_attrs.items():
        pred_imp = []
        idx, smiles, smarts_str = smiles.split('|')

        smarts_list = ast.literal_eval(smarts_str)

        mol = Chem.MolFromSmiles(smiles) 

        indexes = set()

        for smarts in smarts_list:
            
            pattern = Chem.MolFromSmarts(smarts)
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                indexes.update(set().union(*matches))


        assert mol.GetNumAtoms() == len(attrs), f'Num atoms: {mol.GetNumAtoms()} does not match the number of attributes: {len(attrs)}'

        acc = 0

        p = len(indexes)
        n = mol.GetNumAtoms() - p

        if p == 0 or n ==0:
            # Cannot compute AUROC if there are no positive or negative samples
            auroc = None
        else:
            # scores ascending ranking
            scores = [score for score in attrs]

            # Compute ranks, handling ties by assigning average ranks
            ranks = rankdata(scores, method='average')

            # compute ranks of positive samples
            sum_positive_ranks = sum([ranks[i] for i in indexes])

            # calculate auroc
            auroc = (sum_positive_ranks - p*(p+1)/2) / (p*n)
            
        for i, attr in enumerate(attrs):
            if (attr >= threshold and i in indexes) or (attr < threshold and i not in indexes):
                acc += 1
            if attr >= threshold:
                pred_imp.append(i)

        acc = acc/len(attrs)
        acc_mols[idx] = {'Accuracy': acc, 'Auroc': auroc}
        

        if acc < .95 and save_results:
            print(idx)
            create_mol_plot2(smiles, pred_imp, indexes, f'{logdir}/mols/{idx}.png')
    
    if save_results:
        with open(f'{logdir}/metrics.json', 'w') as f:
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
    report_lines.append(f"- Mean Training Accuracy at Best Threshold: {threshold_results[best_threshold]:.2f}\n")

    # Test Set Evaluation
    accuracies_test = [value['Accuracy'] for value in results_test.values()]
    mean_accuracy_test = sum(accuracies_test) / len(accuracies_test)
    if len(accuracies_test) > 1:
        std_accuracy_test = statistics.stdev(accuracies_test)
    else:
        std_accuracy_test = 0.0  # Standard deviation is zero if only one data point

    auroc_scores_test = [value['Auroc'] for value in results_test.values() if value['Auroc'] is not None]
    if len(auroc_scores_test) >1:
        mean_auroc_test = sum(auroc_scores_test) / len(auroc_scores_test)
        std_auroc_test = statistics.stdev(auroc_scores_test)

    else:
        mean_auroc_test = 0.0
        std_auroc_test = 0.0

    report_lines.append(f"Test Set Evaluation at Threshold {best_threshold}:")
    report_lines.append(f"- Mean Accuracy: {mean_accuracy_test:.2f} (±{std_accuracy_test:.2f})")
    report_lines.append(f"- Mean AUROC: {mean_auroc_test:.2f} (±{std_auroc_test:.2f})\n")

    # Write report to file
    report_path = os.path.join(exp_dir, 'experiment_report.txt')
    with open(report_path, 'w') as report_file:
        report_file.write('\n'.join(report_lines))

    print(f"Experiment report saved to {report_path}")