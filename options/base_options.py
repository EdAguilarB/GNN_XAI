import argparse
import os


class BaseOptions:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):

        ###########################################
        ########Options to run experiments#########
        ###########################################

        self.parser.add_argument(
            '--train', 
            type=self.str2bool,
            nargs='?', 
            const=True, 
            default=True, 
            help='Whether to train the GNN or not'
            )
        
        self.parser.add_argument(
            '--filename',
            type=str,
            default='3MR.csv',
            help='name of the csv file',
            )
        
        self.parser.add_argument(
            '--root',
            type=str,
            default='data/datasets/3MR',
            help='root directory of the dataset',
            )
        
        self.parser.add_argument(
            '--mol_cols',
            type=list,
            default=['SMILES'],
            help='Columns containing the SMILES of the molecules',
            )
        
        self.parser.add_argument(
            '--target_variable',
            type=str,
            default='label_full',
            help='Name of the column with the target variable',
            )
        
        self.parser.add_argument(
            '--set_col',
            type=str,
            default='splits',
            help='Name of the column with the set information',
            )
        

    



        self.initialized = True


    def parse(self):
        if not self.initialized:
            self.initialize()
        self._opt = self.parser.parse_args()

        return self._opt
    
    @staticmethod
    def str2bool(value):
        if isinstance(value, bool):
            return value
        if value.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif value.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')