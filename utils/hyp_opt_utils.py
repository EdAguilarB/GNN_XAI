import torch
from torch_geometric.loader import DataLoader
from data.mol_instance import molecular_graph
from call_methods import make_network
from sklearn.model_selection import train_test_split
from utils.model_utils import train_network, eval_network
from ray import tune

def train_model_ray(config, opt):
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    mols = molecular_graph(opt=opt, filename=opt.filename, root=f'../../../../../../../{opt.root}')

    # Split dataset indices
    train_indices = [i for i, s in enumerate(mols.set) if s == 'train']

    # Further split train_indices into train and validation indices
    train_indices, test_indices = train_test_split(train_indices, test_size=0.2, random_state=opt.global_seed)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=opt.global_seed)

    # Create datasets
    train_dataset = mols[train_indices]
    val_dataset = mols[val_indices]
    test_dataset = mols[test_indices]

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=int(config['batch_size']), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=int(config['batch_size']), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=int(config['batch_size']), shuffle=False)


    # Initialize the model with hyperparameters from config
    model = make_network(
        network_name=opt.network_name,
        opt=opt,
        n_node_features=mols.num_node_features,
        n_edge_features=mols.num_edge_features,
        **config
    ).to(device)


    best_val_loss = float('inf')
    early_stopping_counter = 0


    for epoch in range(1, config['epochs'] + 1):
        # Training step
        train_loss = train_network(model, train_loader, device)
        val_loss = eval_network(model, val_loader, device)
        test_loss = eval_network(model, test_loader, device)

        model.scheduler.step(val_loss)

        if epoch % 5 == 0:
            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                tune.report(test_loss = test_loss, epoch = epoch)
                early_stopping_counter = 0

            else:
                early_stopping_counter += 1
                if early_stopping_counter >= config['early_stopping_patience']:
                    print(f'Early stopping at epoch {epoch}')
                    break


