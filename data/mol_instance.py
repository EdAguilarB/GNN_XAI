import argparse
import pandas as pd
import torch
from torch_geometric.data import  Data
import numpy as np 
from rdkit import Chem
import os
from tqdm import tqdm
import sys
from data.mol_graph import mol_graph_dataset
from sklearn.model_selection import StratifiedKFold

from icecream import ic

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class molecular_graph(mol_graph_dataset):

    def __init__(self, opt:argparse.Namespace, filename: str, root: str = None, include_fold = False) -> None:

        self._include_fold = include_fold

        if self._include_fold:
            try:
                file_folds = filename[:-4] + '_folds' + filename[-4:]
                pd.read_csv(os.path.join(root, 'raw', f'{file_folds}'))
            except:
                self.split_data(root, filename, opt.folds, opt.global_seed)
            filename = filename[:-4] + '_folds' + filename[-4:]

        super().__init__(opt = opt,)

        
    def process(self):

        # reads the csv file with smiles of molecules
        self.data = pd.read_csv(self.raw_paths[0]).reset_index()

        # iterates over the rows of the dataframe
        for index, mols in tqdm(self.data.iterrows(), total=self.data.shape[0]):

            # initialize the node features for the molecule
            node_feats_mols = None

            smiles_list = []
            smarts_list = []

            # iterates over the molecules that modulate the target variable
            for molecule in self.mol_cols:  

                #create a molecule object from the smiles string
                smiles = Chem.MolToSmiles(Chem.MolFromSmiles(mols[molecule]), canonical=True)
                smiles_list.append(smiles)
                smarts_list.append(mols[self._opt.smarts_col])

                mol = Chem.MolFromSmiles(smiles)

                node_feats = self._get_node_feats(mol,) 

                edge_attr, edge_index = self._get_edge_features(mol,)

                if node_feats_mols is None:
                    node_feats_mols = node_feats
                    edge_attr_mols = edge_attr
                    edge_index_mols = edge_index

                else:
                    node_feats_mols = torch.cat([node_feats_mols, node_feats], axis=0)
                    edge_attr_mols = torch.cat([edge_attr_mols, edge_attr], axis=0)
                    edge_index += max(edge_index_mols[0]) + 1
                    edge_index_mols = torch.cat([edge_index_mols, edge_index], axis=1)

            y = torch.tensor(mols[self._opt.target_variable]).reshape(1)
            set = mols[self._opt.set_col]

            if self._opt.id_col:
                id = mols[self._opt.id_col]
            else:
                id = index

            data = Data(x=node_feats_mols, 
                        edge_index=edge_index_mols, 
                        edge_attr=edge_attr_mols, 
                        y=y,
                        smiles = smiles,
                        idx = id,
                        set = set
                        ) 
            
            torch.save(data, 
                       os.path.join(self.processed_dir, 
                                    f'mol_{index}.pt'))
    

    def _get_node_feats(self, mol):

        all_node_feats = []
        #CIPtuples = dict(Chem.FindMolChiralCenters(mol, includeUnassigned=False))

        for atom in mol.GetAtoms():
            node_feats = []
            # Feature 1: Atomic number        
            node_feats += self._one_h_e(atom.GetSymbol(), self._elem_list)
            # Feature 2: Atom degree
            node_feats += self._one_h_e(atom.GetDegree(), [1, 2, 3, 4])
            # Feature 3: Formal Charge
            node_feats += self._one_h_e(atom.GetFormalCharge(), [-1, 0, 1])
            # Feature 4: Chirality
            node_feats += self._one_h_e(atom.GetChiralTag(), [Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.rdchem.ChiralType.CHI_UNSPECIFIED])
            # Feature 5: Num Hs
            node_feats += self._one_h_e(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
            # Feature 6: Hybridization
            node_feats += self._one_h_e(atom.GetHybridization(), [0,2,3,4])
            # Feature 7: Aromaticity
            node_feats += [atom.GetIsAromatic()]
            # Feature 8: In Ring
            node_feats += [atom.IsInRing()]

            # Append node features to matrix
            all_node_feats.append(node_feats)

        all_node_feats = np.asarray(all_node_feats, dtype=np.float32)
        return torch.tensor(all_node_feats, dtype=torch.float)
    

    def _get_labels(self, label):
        label = np.asarray([label])
        return torch.tensor(label, dtype=torch.float)
    
    
    def _get_edge_features(self, mol):

        all_edge_feats = []
        edge_indices = []

        for bond in mol.GetBonds():

            #list to save the edge features
            edge_feats = []

            # Feature 1: Bond type (as double)
            edge_feats += self._one_h_e(bond.GetBondType(), [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC])

            #feature 2: double bond stereochemistry
            edge_feats += self._one_h_e(bond.GetStereo(), [Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE], Chem.rdchem.BondStereo.STEREONONE)

            # Feature 3: Is in ring
            edge_feats.append(bond.IsInRing())

            # Append node features to matrix (twice, per direction)
            all_edge_feats += [edge_feats, edge_feats]

            # Append edge indices to list (twice, per direction)
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            #create adjacency list
            edge_indices += [[i, j], [j, i]]

        all_edge_feats = np.asarray(all_edge_feats)
        edge_indices = torch.tensor(edge_indices)
        edge_indices = edge_indices.t().to(torch.long).view(2, -1)

        return torch.tensor(all_edge_feats, dtype=torch.float), edge_indices
    

    def _create_folds(num_folds, df):
        """
        splits a dataset in a given quantity of folds

        Args:
        num_folds = number of folds to create
        df = dataframe to be splited

        Returns:
        dataset with new "folds" and "mini_folds" column with information of fold for each datapoint
        """

        # Calculate the number of data points in each fold
        fold_size = len(df) // num_folds
        remainder = len(df) % num_folds

        # Create a 'fold' column to store fold assignments
        fold_column = []

        # Assign folds
        for fold in range(1, num_folds + 1):
            fold_count = fold_size
            if fold <= remainder:
                fold_count += 1
            fold_column.extend([fold] * fold_count)

        # Assign the 'fold' column to the DataFrame
        df['fold'] = fold_column

        return df
    

    def split_data(self, root, filename, n_folds, random_seed):

        dataset = pd.read_csv(os.path.join(root, 'raw', f'{filename}'))
        dataset['category'] = dataset['%top'].apply(lambda m: 0 if m < 50 else 1)

        folds = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state=random_seed)

        test_idx = []

        for _, test in folds.split(np.zeros(len(dataset)), dataset['category']):
            test_idx.append(test)

        index_dict = {index: list_num for list_num, index_list in enumerate(test_idx) for index in index_list}

        dataset['fold'] = dataset.index.map(index_dict)

        filename = filename[:-4] + '_folds' + filename[-4:]

        dataset.to_csv(os.path.join(root, 'raw', filename))

        print('{}.csv file was saved in {}'.format(filename, os.path.join(root, 'raw')))