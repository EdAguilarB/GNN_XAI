import argparse
import os
import sys
import torch
import pandas as pd
from torch_geometric.data import Dataset

from icecream import ic


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class mol_graph_dataset(Dataset):


    def __init__(self, opt: argparse.Namespace, filename, root) -> None:

        self._opt = opt
        self.filename = filename
        self.mol_cols = opt.mol_cols
        self._root = root
        self._name = "BaseDataset"
        
        super().__init__(root = root)
        self.set = pd.read_csv(self.raw_paths[0])[opt.set_col]
        self.y = pd.read_csv(self.raw_paths[0])[opt.target_variable]
        

    @property
    def raw_file_names(self):
        return self.filename
    
    @property
    def processed_file_names(self):
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()
        molecules = [f'mol_{i}.pt' for i in list(self.data.index)]
        return molecules
    
    @property
    def _elem_list(self):
        elements = [
            'H',  # Hydrogen (1)
            'Li', # Lithium (3)
            'B',  # Boron (5)
            'C',  # Carbon (6)
            'N',  # Nitrogen (7)
            'O',  # Oxygen (8)
            'F',  # Fluorine (9)
            'Na', # Sodium (11)
            'Si', # Silicon (14)
            'P',  # Phosphorus (15)
            'S',  # Sulfur (16)
            'Cl', # Chlorine (17)
            'K',  # Potassium (19)
            'Se', # Selenium (34)
            'Br', # Bromine (35)
            'As', # Arsenic (33)
            'Ge', # Germanium (32)
            'Sn', # Tin (50)
            'Sb', # Antimony (51)
            'I',  # Iodine (53)
        ]
        return elements
        
    
    def download(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError
    
    def _get_node_feats(self):
        raise NotImplementedError
    
    def _get_edge_features(self):
        raise NotImplementedError
    
    def _print_dataset_info(self) -> None:
        """
        Prints the dataset info
        """
        print(f"{self._name} dataset has {len(self)} samples")

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):

        molecule = torch.load(os.path.join(self.processed_dir, 
                                f'mol_{idx}.pt')) 
        return molecule
    
    def _get_atom_chirality(self, CIP_dict, atom_idx):
        try:
            chirality = CIP_dict[atom_idx]
        except KeyError:
            chirality = 'No_Stereo_Center'

        return chirality
    
    def _one_h_e(self, x, allowable_set, ok_set=None):

        if x not in allowable_set:
            if ok_set is not None and x == ok_set:
                pass
            else:
                print(x)
        return list(map(lambda s: x == s, allowable_set))