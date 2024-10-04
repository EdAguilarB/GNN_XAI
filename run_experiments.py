from experiments_scripts.train_GNN import train_model
from options.base_options import BaseOptions

def run():
    opt = BaseOptions()
    opt = opt.parse()

    if opt.train_GNN:
        train_model(opt)


if __name__ == '__main__':
    run()