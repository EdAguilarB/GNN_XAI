from data.mol_instance import molecular_graph
from options.base_options import BaseOptions
from call_methods import make_network

def run():
    opt = BaseOptions()
    opt = opt.parse()

    mols = molecular_graph(opt = opt, filename = '3MR.csv', root = 'data/datasets/3MR')
    train_set = []
    test_set = []
    val_set = []    

    
    model = make_network('graphsage', opt, mols.num_node_features, mols.num_edge_features)
    print(model)

    print(model.forward(x = mols[0].x, edge_index = mols[0].edge_index, ))


if __name__ == '__main__':
    run()