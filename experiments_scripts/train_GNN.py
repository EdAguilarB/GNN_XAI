import torch
from torch_geometric.loader import DataLoader
from utils.model_utils import train_network, eval_network, network_report
from data.mol_instance import molecular_graph
from call_methods import make_network
from icecream import ic

def train_model(opt):

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    mols = molecular_graph(opt = opt, filename = opt.filename, root = opt.root)

    
    train_indices = [i for i, s in enumerate(mols.set) if s == 'train']
    test_indices = [i for i, s in enumerate(mols.set) if s == 'test']

    train_dataset = mols[train_indices]
    test_dataset = mols[test_indices]

    # Make the dataloaders
    train_set = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    test_set = DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False)

    # Make the network
    model = make_network(network_name=opt.network_name, 
                         opt=opt, 
                         n_node_features=mols.num_node_features, 
                         n_edge_features=mols.num_edge_features).to(device)
    
    train_list,  test_list = [], []

    for epoch in range(1, opt.epochs+1):

        train_loss = train_network(model = model, 
                                   train_loader=train_set, 
                                   device=device)
        
        
        test_loss = eval_network(model = model,
                                loader=test_set,
                                device=device)
        
        print('Epoch {:03d} | Train loss: {:.3f} | Test loss: {:.3f}'.format(epoch, train_loss, test_loss))

        train_list.append(train_loss)
        test_list.append(test_loss)

    network_report(
                exp_name=f'{opt.exp_name}/{mols.filename[:-4]}',
                loaders=(train_set, test_set),
                loss_lists=(train_list, test_list),
                save_all=True,
                model=model,
            )


