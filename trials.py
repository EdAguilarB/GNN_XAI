import torch
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer, PGExplainer
from torch_geometric.loader import DataLoader
from options.base_options import BaseOptions
import json
from tqdm import tqdm
from icecream import ic


def run():

    # Parse the options
    opt = BaseOptions()
    opt = opt.parse()

    # Set the device
    device = torch.device("cpu")

    # Load the dataset and model from experiment
    exp_dir = '3MR/graphsage'
    train_loader = DataLoader(torch.load("{}/train_loader.pth".format(exp_dir)).dataset)
    test_loader = DataLoader(torch.load("{}/test_loader.pth".format(exp_dir)).dataset)
    model = torch.load("{}/model.pth".format(exp_dir))


    algorithms = [GNNExplainer(), 
                  CaptumExplainer('IntegratedGradients'), 
                  CaptumExplainer('Saliency'),
                  CaptumExplainer('InputXGradient'),
                  CaptumExplainer('Deconvolution'),
                  CaptumExplainer('ShapleyValueSampling'),
                  CaptumExplainer('GuidedBackprop'),
                  ]
    
    algorithms_names = ['GNNExplainer',
                        'IntegratedGradients',
                        'Saliency',
                        'InputXGradient',
                        'Deconvolution',
                        'ShapleyValueSampling',
                        'GuidedBackprop',
                        ]
    

    node_masks = {}
    
    for mol in test_loader:

        smiles = mol.smiles[0]
        algo_dict = {}

        for name, algorithm in zip(algorithms_names, algorithms):
            model = torch.load("{}/model.pth".format(exp_dir))
            # Initialize the explainer
            explainer = Explainer(
                model=model,
                algorithm=algorithm,
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='multiclass_classification',
                    task_level='graph',
                    return_type='raw',
                ),
            )

            explanation = explainer(x=mol.x, edge_index=mol.edge_index)
            node_mask = explanation.node_mask

            algo_dict[name] = node_mask.detach().numpy().tolist()
        
        node_masks[smiles] = algo_dict


    with open('node_masks_test.json', 'w') as f:
        json.dump(node_masks, f, indent=4)



    node_masks = {}
    
    for mol in train_loader:

        smiles = mol.smiles[0]
        algo_dict = {}

        for name, algorithm in zip(algorithms_names, algorithms):
            model = torch.load("{}/model.pth".format(exp_dir))
            # Initialize the explainer
            explainer = Explainer(
                model=model,
                algorithm=algorithm,
                explanation_type='model',
                node_mask_type='attributes',
                edge_mask_type='object',
                model_config=dict(
                    mode='multiclass_classification',
                    task_level='graph',
                    return_type='raw',
                ),
            )

            explanation = explainer(x=mol.x, edge_index=mol.edge_index)
            node_mask = explanation.node_mask

            algo_dict[name] = node_mask.detach().numpy().tolist()
        
        node_masks[smiles] = algo_dict


    with open('node_masks_train.json', 'w') as f:
        json.dump(node_masks, f, indent=4)
        

if __name__ == '__main__':
    run()