import json
import os
from pathlib import Path

import pandas as pd
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from data.mol_instance import molecular_graph
from options.base_options import BaseOptions
from utils.hyp_opt_utils import train_model_ray

configs = {
    # ALL MODEL HYPERPARAMETERS
    # base network params
    "lr": tune.choice([1e-2, 1e-3, 1e-4]),
    "n_convolutions": tune.choice([2, 3, 4]),
    "embedding_dim": tune.choice([32, 64, 128]),
    "readout_layers": tune.choice([1, 2, 3]),
    "dropout": tune.choice([0.0, 0.1]),
    "step_size": tune.choice([5, 10, 15]),
    "gamma": tune.choice([0.7, 0.5, 0.3]),
    "min_lr": tune.choice([1e-08, 0]),
    "pooling": tune.choice(["mean", "max", "sum", "mean/max"]),
    # training params
    "batch_size": tune.choice([32, 64]),
    "early_stopping_patience": tune.choice([1, 3, 5]),
    "epochs": 300,  # Keep epochs constant or adjust as needed
    # GCN SPECIFIC HYPERPARAMETERS
    "improved": tune.choice([True, False]),
    # GAT SPECIFIC HYPERPARAMETERS
    "heads": tune.choice([1, 2, 4]),
    # GraphSAGE SPECIFIC HYPERPARAMETERS
    "aggr": tune.choice(["mean", "max"]),
}

base_network_params = [
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
training_params = ["batch_size", "early_stopping_patience", "epochs"]
all_params = base_network_params + training_params


def run_tune(
    exp_name,
    filename,
    network_name,
    root,
    mol_cols,
    set_col,
    target_variable,
    id_col,
    problem_type,
    n_classes,
    global_seed,
    optimizer,
    scheduler,
):

    log_dir = Path(exp_name) / filename[:-4] / network_name / "results_hyp_opt"

    log_dir.mkdir(parents=True, exist_ok=True)

    _ = molecular_graph(
        filename=filename,
        root=root,
        mol_cols=mol_cols,
        set_col=set_col,
        target_variable=target_variable,
        id_col=id_col,
    )

    basic_datasets = ["benzene.csv", "3MR.csv"]

    if filename in basic_datasets:
        num_samples = 50
    else:
        num_samples = 30

    if network_name.lower() == "GCN".lower():
        GCN_params = all_params + ["improved"]
        config = {k: v for k, v in configs.items() if k in GCN_params}
    elif network_name.lower() == "GAT".lower():
        GAT_params = all_params + ["heads"]
        config = {k: v for k, v in configs.items() if k in GAT_params}
    elif network_name.lower() == "graphsage":
        GraphSAGE_params = all_params + ["aggr"]
        config = {k: v for k, v in configs.items() if k in GraphSAGE_params}
    else:
        raise ValueError(f"Network {network_name} not implemented")

    AshasScheduler = ASHAScheduler(
        time_attr="epoch",
        metric="test_loss",
        mode="min",
        max_t=300,
        grace_period=50,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        parameter_columns=[
            "lr",
            "n_convolutions",
            "embedding_dim",
            "readout_layers",
            "batch_size",
        ],
        metric_columns=["val_loss", "test_loss", "training_iteration"],
    )

    ray.init()
    print("Cluster resources:", ray.cluster_resources())

    result = tune.run(
        tune.with_parameters(
            train_model_ray,
            filename=filename,
            root=root,
            mol_cols=mol_cols,
            set_col=set_col,
            target_variable=target_variable,
            id_col=id_col,
            problem_type=problem_type,
            n_classes=n_classes,
            global_seed=global_seed,
            network_name=network_name,
            optimizer=optimizer,
            scheduler=scheduler,
        ),
        resources_per_trial={"cpu": 1, "gpu": 0.0},  # Adjust based on your resources
        config=config,
        num_samples=num_samples,  # Number of hyperparameter combinations to try
        scheduler=AshasScheduler,
        progress_reporter=reporter,
        storage_path=f"{log_dir}/ray_results",
        name="tune_hyperparameters",
    )

    best_trial = result.get_best_trial("test_loss", "min", "last")
    best_config = best_trial.config
    best_test_loss = best_trial.last_result["test_loss"]
    print("Best trial config: {}".format(best_config))
    print("Best trial final test loss: {:.4f}".format(best_test_loss))

    best_config = result.get_best_config(metric="test_loss", mode="min", scope="last")
    best_config_df = pd.DataFrame.from_dict(best_config, orient="index")
    best_config_df.to_csv(f"{log_dir}/best_config.csv", header=False)
    all_runs_df = result.results_df
    all_runs_df.to_csv(f"{log_dir}/all_runs.csv", index=False)
    best_tuned_config = {key: best_trial.config[key] for key in config.keys()}

    with open(f"{log_dir}/best_hyperparameters.json", "w") as f:
        json.dump(best_tuned_config, f, indent=4)


if __name__ == "__main__":
    # Ensure 'opt' is properly initialized with all necessary attributes
    opt = BaseOptions()
    opt = opt.parse()
    run_tune(opt)
