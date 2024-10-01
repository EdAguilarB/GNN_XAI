import torch
from torch_geometric.loader import DataLoader
import json
import numpy as np
from rdkit import Chem
import random
from options.base_options import BaseOptions
from utils.XAI_utils import calculate_attributions
from icecream import ic


def run():

    # Parse the options
    opt = BaseOptions()
    opt = opt.parse()

    random.seed(opt.global_seed)

    # Load the dataset and model from experiment
    exp_dir = '3MR/graphsage'
    train_loader = DataLoader(torch.load("{}/train_loader.pth".format(exp_dir)).dataset[:50])
    test_loader = DataLoader(torch.load("{}/test_loader.pth".format(exp_dir)).dataset[:10])


    #calculate_attributions(opt, train_loader)

    with open('node_masks_test.json') as f:
        data = json.load(f)

    

    mol_attrs = {}
    
    for outer_key, outer_value in data.items():
        dir, mag = None, None
        for inner_key, inner_value in outer_value.items():
            value = np.array(inner_value)
            value /= np.abs(np.max(value))
            #ic(inner_key, value.shape)
            if inner_key == 'GNNExplainer' or inner_key == 'Saliency':
                if not isinstance(mag, np.ndarray):
                    mag = value
                else:
                    mag += value
            else:
                if not isinstance(dir, np.ndarray):
                    dir = value
                else:
                    dir += value
        
        sign = np.sign(dir)
        dir = np.abs(dir)
        mag = dir + mag
        mag = np.sum(mag, axis=1) # sum over the columns to get the overall importance of each atom
        mag /= np.max(mag) # normalize the importance values from 0 to 1
        #mag = mag * sign

        mag = np.where(mag < 0.5, 0, 1) # thresholding to map atoms to either important or not 

        mol_attrs[outer_key] = mag

    
    pattern = Chem.MolFromSmarts('[*]1[*][*]1')
    for smiles, attrs in mol_attrs.items():
        mol = Chem.MolFromSmiles(smiles) 
        matches = mol.GetSubstructMatches(pattern)

        acc = 0

        indexes = set().union(*matches)
        ic(indexes)


        for i, attr in enumerate(attrs):
            ic(i, attr)
            if attr == 1:
                if i in indexes:
                    acc += 1
                elif i not in indexes:
                    pass
            elif attr == 0:
                if i in indexes:
                    pass
                elif i not in indexes:
                    acc += 1
            
            #ic(acc)
        
        ic(acc)
        ic(len(attrs))
                    
        ic(smiles, acc/len(attrs))
        break



        

if __name__ == '__main__':
    run()