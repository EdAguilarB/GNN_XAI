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
            '--exp_name',
            type=str,
            default='exp1',
            help='Name of the experiment',
            )

        self.parser.add_argument(
            '--train_GNN', 
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
            default='3MR',
            help='Name of the column with the target variable',
            )
        
        self.parser.add_argument(
            '--set_col',
            type=str,
            default='splits',
            help='Name of the column with the set information',
            )
        
        self.parser.add_argument(
            '--id_col',
            type=str,
            default='index',
            help='Column with the id of the molecules',
            )
        
        self.parser.add_argument(
            '--problem_type',
            type=str,
            default='classification',
            help='Type of problem to solve',
            )
        
        self.parser.add_argument(
            '--optimizer',
            type=str,
            default='Adam',
            help='Type of optimizer',
            )
        
        self.parser.add_argument(
            '--lr',
            type=float,
            default=0.0001,
            help='Learning rate',
            )
        
        self.parser.add_argument(
            '--scheduler',
            type=str,
            default='ReduceLROnPlateau',
            help='Type of scheduler',
            )
        
        self.parser.add_argument(
            '--step_size',
            type=int,
            default=7,
            help='Step size for the scheduler',
            )
        
        self.parser.add_argument(
            '--gamma',
            type=float,
            default=0.7,
            help='Factor for the scheduler',
            )
        
        self.parser.add_argument(
            '--min_lr',
            type=float,
            default=1e-08,
            help='Minimum learning rate for the scheduler',
            )
        
        self.parser.add_argument(
            '--n_classes',
            type=int,
            default=2,
            help='Number of classes in the target variable',
            )
        
        self.parser.add_argument(
            '--n_convolutions',
            type=int,
            default=3,
            help='Number of classes in the target variable',
            )  
        
        self.parser.add_argument(
            '--embedding_dim',
            type=int,
            default=128,
            help='Number of classes in the target variable',
            )  
        
        self.parser.add_argument(
            '--readout_layers',
            type=int,
            default=2,
            help='Number of classes in the target variable',
            )  
        
        self.parser.add_argument(
            '--improved',
            type=int,
            default=False,
            help='To include biases in GCNConv or not',
            )
        
        self.parser.add_argument(
            '--batch_size',
            type=float,
            default=64,
            help='Dropout rate',
            )
        
        self.parser.add_argument(
            '--epochs',
            type=int,
            default=100,
            help='Number of epochs',
            )

        ###########################################
        ########Options for explainability#########
        ###########################################

        self.parser.add_argument(
            '--algorithm',
            type=str,
            default='GNNExplainer',
            help='Algorithm to use for explainability',
            )
        


        self.parser.add_argument(
            '--global_seed',
            type=int,
            default=42,
            help='Seed for the random number generator',
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