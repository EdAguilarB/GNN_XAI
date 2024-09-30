import torch
import argparse
import torch.nn as nn
from torch_geometric.seed import seed_everything


class BaseNetwork(nn.Module):
    
    def __init__(self, opt: argparse.Namespace, n_node_features:int, n_edge_features:int = None):
        super().__init__()
        self._name = "BaseNetwork"
        self._opt = opt
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features  
        self._n_classes = opt.n_classes
        self.n_convolutions = opt.n_convolutions
        self.embedding_dim = opt.embedding_dim  
        self.readout_layers = opt.readout_layers
        self.problem_type = opt.problem_type
        self.num_classes = opt.n_classes
        self._seed_everything(opt.global_seed)

    def forward(self):
        raise NotImplementedError
    
    @property
    def name(self):
        return self._name
    
    def _make_loss(self,  mae=None):
        if self.problem_type == "classification":
            self.loss = nn.CrossEntropyLoss()
        elif self.problem_type == "regression" and mae is None:
            self.loss = nn.MSELoss()
        else:
            raise ValueError(f"Problem type {self.problem_type} not supported")
        
    def _make_optimizer(self, optimizer, lr):
        if optimizer == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, eps=1e-9)
        elif optimizer == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        elif optimizer == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.parameters(), lr=lr)
        else:
            raise NotImplementedError(f"Optimizer type {optimizer} not implemented")
        
    def _make_scheduler(self, scheduler, step_size, gamma, min_lr):
        if scheduler == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=step_size, gamma=gamma)
        elif scheduler == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=gamma)
        elif scheduler == "ReduceLROnPlateau":
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=gamma, patience=step_size, min_lr=min_lr)
        else:
            raise NotImplementedError(f"Scheduler type {scheduler} not implemented")
        
    def _seed_everything(self, seed):
        seed_everything(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
