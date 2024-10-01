import torch
from torch_geometric.loader import DataLoader
from utils.model_utils import train_network, eval_network, network_report
from data.mol_instance import molecular_graph
from options.base_options import BaseOptions
from call_methods import make_network

def train_model():

    # Parse the options
    opt = BaseOptions()
    opt = opt.parse()

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the dataset
    mols = molecular_graph(opt = opt, filename = '3MR.csv', root = 'data/datasets/3MR')

    # Make the network
    model = make_network(network_name='graphsage', 
                         opt=opt, 
                         n_node_features=mols.num_node_features, 
                         n_edge_features=mols.num_edge_features).to(device)

    train_set = []
    test_set = []

    for mol in mols:
        if mol.set == 'train':
            train_set.append(mol)
        elif mol.set == 'test':
            test_set.append(mol)


    # Make the dataloaders
    train_set = DataLoader(train_set, batch_size=opt.batch_size, shuffle=True)
    test_set = DataLoader(test_set, batch_size=opt.batch_size, shuffle=False)
    
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
                exp_name=mols.filename[:-4],
                loaders=(train_set, test_set),
                loss_lists=(train_list, test_list),
                save_all=True,
                model=model,
            )


if __name__ == '__main__':
    train_model()