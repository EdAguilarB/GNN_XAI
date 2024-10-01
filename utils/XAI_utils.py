import torch
from torch_geometric.explain import Explainer, GNNExplainer, CaptumExplainer
from tqdm import tqdm
import json


def calculate_attributions(opt, loader):

    exp_dir = '3MR/graphsage'

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
    
    for mol in tqdm(loader):

        smiles = mol.smiles[0]
        smarts = mol.smarts[0]
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
        
        node_masks[smiles+'.'+smarts] = algo_dict


    with open('node_masks_test.json', 'w') as f:
        json.dump(node_masks, f, indent=4)
    
    return node_masks