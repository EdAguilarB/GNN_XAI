from data.mol_instance import molecular_graph
from options.base_options import BaseOptions


opt = BaseOptions()
opt = opt.parse()

mol = molecular_graph(opt = opt, filename = '3MR.csv', root = 'data/datasets/3MR')