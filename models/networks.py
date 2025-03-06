import torch
import torch.nn as nn
from torch_geometric.seed import seed_everything


class BaseNetwork(nn.Module):

    def __init__(
        self,
        n_node_features: int,
        n_edge_features,
        n_classes,
        problem_type,
        global_seed,
        **kwargs,
    ):

        super().__init__()

        self._name = "BaseNetwork"
        self.n_node_features = n_node_features
        self.n_edge_features = n_edge_features

        base_params = [
            "lr",
            "n_convolutions",
            "embedding_dim",
            "readout_layers",
            "dropout",
            "step_size",
            "gamma",
            "min_lr",
            "pooling",
        ]

        for param in base_params:
            setattr(self, param, kwargs.pop(param))

        self.kwargs = kwargs

        self.n_classes = n_classes

        self.problem_type = problem_type
        self.num_classes = n_classes
        self._seed_everything(global_seed)

        if self.pooling == "mean/max":
            self.graph_embedding = self.embedding_dim * 2
        else:
            self.graph_embedding = self.embedding_dim

    def forward(self):
        raise NotImplementedError

    @property
    def name(self):
        return self._name

    def _seed_everything(self, seed):
        seed_everything(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
