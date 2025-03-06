import os
from pathlib import Path

from experiments_scripts.hyp_opt import run_tune
from experiments_scripts.train_GNN import train_model
from experiments_scripts.XAI_experiments import run_XAI
from options.base_options import BaseOptions


def run():
    opt = BaseOptions()
    opt = opt.parse()

    parent_dir = Path(opt.exp_name) / opt.filename[:-4] / opt.network_name

    hyp_dir = parent_dir / "results_hyp_opt"
    GNN_dir = parent_dir / "results_model"
    XAI_dir = parent_dir / "results_XAI" / opt.XAI_algorithm

    # Run hyperparameter optimization if the best config file does not exist
    if not os.path.exists(f"{hyp_dir}/best_hyperparameters.json"):
        print(
            f"{hyp_dir}/best_hyperparameters.json does not exist. Running hyperparameter optimization."
        )
        run_tune(
            exp_name=opt.exp_name,
            filename=opt.filename,
            network_name=opt.network_name,
            root=opt.root,
            mol_cols=opt.mol_cols,
            set_col=opt.set_col,
            target_variable=opt.target_variable,
            id_col=opt.id_col,
            problem_type=opt.problem_type,
            n_classes=opt.n_classes,
            global_seed=opt.global_seed,
            optimizer=opt.optimizer,
            scheduler=opt.scheduler,
        )

    # Train the GNN if the model file does not exist
    if not os.path.exists(f"{GNN_dir}/model.pth"):
        print(f"{GNN_dir}/model.pth does not exist. Training the GNN model.")
        train_model(opt)

    # Run the XAI experiments if the node masks do not exist
    if not os.path.exists(f"{XAI_dir}/metrics_{opt.XAI_attrs_mode}.json"):
        print(
            f"{XAI_dir}/node_masks_train.json does not exist. Running XAI experiments."
        )
        run_XAI(opt)


if __name__ == "__main__":
    run()
