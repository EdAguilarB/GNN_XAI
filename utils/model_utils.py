import csv
import os
from datetime import date, datetime
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from icecream import ic
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from utils.plot_utils import (
    create_parity_plot,
    create_training_plot,
    plot_confusion_matrix,
)


def train_network(model, train_loader, device, loss_fn, optimizer):

    train_loss = 0
    model.train()

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(
            x=batch.x,
            edge_index=batch.edge_index,
            batch_index=batch.batch,
            edge_attr=batch.edge_attr,
        )
        if model.problem_type == "classification":
            loss = loss_fn(out, batch.y.long())
        elif model.problem_type == "regression":
            loss = torch.sqrt(loss_fn(out.squeeze(), batch.y.float()))

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch.num_graphs

    return train_loss / len(train_loader.dataset)


def eval_network(model, loader, device, loss_fn):
    model.eval()
    loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)
            if model.problem_type == "classification":
                loss += loss_fn(out, batch.y.long()).item() * batch.num_graphs
            elif model.problem_type == "regression":
                loss += (
                    torch.sqrt(loss_fn(out.squeeze(), batch.y.float())).item()
                    * batch.num_graphs
                )
    return loss / len(loader.dataset)


def calculate_metrics(
    y_true: np.ndarray,
    y_predicted: np.ndarray,
    task: str,
    num_classes: int,
    y_score: np.ndarray = None,
) -> dict:
    metrics = {}
    if task == "regression":
        metrics["R2"] = r2_score(y_true=y_true, y_pred=y_predicted)
        metrics["MAE"] = mean_absolute_error(y_true=y_true, y_pred=y_predicted)
        metrics["RMSE"] = sqrt(mean_absolute_error(y_true=y_true, y_pred=y_predicted))
        error = [(y_predicted[i] - y_true[i]) for i in range(len(y_true))]
        prctg_error = mean_absolute_percentage_error(y_true=y_true, y_pred=y_predicted)
        metrics["Mean Bias Error"] = np.mean(error)
        metrics["Mean Absolute Percentage Error"] = np.mean(prctg_error)
        metrics["Error Standard Deviation"] = np.std(error)

    elif task == "classification":
        y_true = np.array(y_true).astype(int)
        y_predicted = np.array(y_predicted).astype(int)
        metrics["Accuracy"] = accuracy_score(y_true, y_predicted)

        average_method = "binary" if num_classes == 2 else "macro"
        metrics["Precision"] = precision_score(
            y_true, y_predicted, average=average_method, zero_division=0
        )
        metrics["Recall"] = recall_score(
            y_true, y_predicted, average=average_method, zero_division=0
        )
        metrics["F1"] = f1_score(
            y_true, y_predicted, average=average_method, zero_division=0
        )

        if y_score is not None:
            if num_classes == 2:
                # Binary classification
                if y_score.ndim == 2:
                    y_score = y_score[
                        :, 1
                    ]  # Extract probabilities for positive class (class 1)
                metrics["AUROC"] = roc_auc_score(y_true, y_score)
            else:
                # Multiclass classification
                metrics["AUROC"] = roc_auc_score(y_true, y_score, multi_class="ovr")
        else:
            metrics["AUROC"] = None

    else:
        raise ValueError("Task must be 'regression' or 'classification'.")

    return metrics


def predict_network(model, loader):
    device = torch.device("cpu")
    model.to(device)
    model.eval()

    y_pred, y_true, idx, y_score = [], [], [], []

    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch, batch.edge_attr)

        out = out.cpu().detach()
        if model.problem_type == "classification":
            if out.dim() == 1 or out.size(1) == 1:
                # Binary classification with single output
                probs = torch.sigmoid(out)
                preds = (probs >= 0.5).long()
                y_score.append(probs.numpy().flatten())
            else:
                # Binary classification with two outputs or multiclass classification
                probs = torch.softmax(out, dim=1)
                preds = torch.argmax(probs, dim=1)
                y_score.append(probs.numpy())  # Shape: [batch_size, num_classes]
            y_pred.append(preds.numpy().flatten())
        else:
            # Regression problem
            out = out.numpy().flatten()
            y_pred.append(out)
            y_score = None  # No probabilities for regression

        y_true.append(batch.y.cpu().detach().numpy().flatten())
        idx.append(batch.idx)

    y_pred = np.concatenate(y_pred, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    idx = np.concatenate(idx, axis=0)
    if model.problem_type == "classification":
        y_score = np.concatenate(y_score, axis=0)
    else:
        y_score = None

    return y_pred, y_true, idx, y_score


def generate_embeddings(model, loader):
    """
    Generates a CSV file containing molecular embeddings from the trained model.

    Args:
        model (torch.nn.Module): Trained GNN model.
        loader (torch_geometric.loader.DataLoader): DataLoader containing molecules.
        save_path (str or Path, optional): Path to save the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing embeddings with instances as rows and embedding features as columns.
    """

    device = torch.device("cpu")  # Run on CPU for inference
    model.to(device)
    model.eval()  # Set model to evaluation mode

    embeddings_list = []
    indices_list = []

    # Iterate over batches in the DataLoader
    for batch in loader:
        batch = batch.to(device)

        # Compute embeddings using the model
        emb = model.return_embeddings(
            x=batch.x,
            edge_index=batch.edge_index,
            batch_index=batch.batch,
            edge_attr=batch.edge_attr,
        )

        # Convert embeddings to NumPy
        emb_np = emb.cpu().detach().numpy()
        embeddings_list.append(emb_np)

        # Store molecule indices
        indices_list.append(batch.idx)

    # Concatenate all embeddings and indices
    embeddings_array = np.concatenate(embeddings_list, axis=0)
    indices_array = np.concatenate(indices_list, axis=0)

    # Create a DataFrame
    df = pd.DataFrame(
        embeddings_array,
        columns=[f"emb_{i}" for i in range(1, model.graph_embedding + 1)],
    )
    df.insert(0, "Molecule_ID", indices_array)  # Add molecule ID column

    return df


def evaluate_and_log(model, loader, set_name, log_dir, file, results, embeddings):
    """
    Helper function to evaluate the model on a dataset and log the results.
    """
    y_pred, y_true, idx, y_score = predict_network(model, loader)

    new_embeddings = generate_embeddings(model, loader)
    new_embeddings["Set"] = set_name

    embeddings = pd.concat([embeddings, new_embeddings], ignore_index=True)

    # Store results
    new_results = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "idx": idx, "set": set_name}
    )
    results = pd.concat([results, new_results], ignore_index=True)

    # Compute metrics
    metrics = calculate_metrics(
        y_true=y_true,
        y_predicted=y_pred,
        task=model.problem_type,
        num_classes=model.num_classes,
        y_score=y_score,
    )

    # Log performance
    file.write(f"{set_name.capitalize()} set\n")
    file.write(f"Set size = {len(y_true)}\n")

    for name, value in metrics.items():
        file.write(f"{name} = {value:.3f}\n")

    file.write("***************\n")

    # Generate plots
    if model.problem_type == "classification":
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            title=f"{set_name.capitalize()} Set",
            save_path=log_dir / f"Confusion_matrix_{set_name}.png",
        )

    return results, y_true, y_pred, idx, embeddings


def network_report(
    exp_name: str,
    loaders: tuple,
    loss_lists: list,
    save_all: bool,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    optimizer: str,
    scheduler: str,
):
    """
    Generates a report for a trained neural network, logging its performance
    on train, validation, and test sets, and saving relevant data and plots.
    """

    # 1) Create the results directory
    log_dir = Path(exp_name) / model.name / "results_model"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2) Log run time
    today_str = date.today().strftime("%d-%b-%Y")
    time_str = datetime.now().strftime("%H:%M:%S")
    run_period = f"{today_str}, {time_str}\n"

    # 3) Extract loaders and dataset sizes
    train_loader, val_loader, test_loader = loaders
    N_train, N_val, N_test = (
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )
    N_tot = N_train + N_test

    # 4) Save loaders and model if needed
    if save_all:
        torch.save(train_loader, log_dir / "train_loader.pth")
        torch.save(val_loader, log_dir / "val_loader.pth")
        torch.save(model, log_dir / "model.pth")

    torch.save(test_loader, log_dir / "test_loader.pth")

    # 5) Save loss trends
    train_list, val_list, test_list, lr_list = loss_lists

    if train_list and test_list:
        csv_path = log_dir / "learning_process.csv"
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Epoch",
                    "Learning Rate",
                    f"Train_{loss_fn}",
                    f"Val_{loss_fn}",
                    f"Test_{loss_fn}",
                ]
            )
            for i in range(len(train_list)):
                writer.writerow(
                    [i + 1, lr_list[i], train_list[i], val_list[i], test_list[i]]
                )

        create_training_plot(df=csv_path, save_path=log_dir)

    # 6) Start writing performance report
    report_path = log_dir / "performance.txt"
    with open(report_path, "w") as file:
        file.write(run_period)
        file.write("---------------------------------------------------------\n")
        file.write("GNN TRAINING AND PERFORMANCE\n")
        file.write(f"Model = {model.name}\n")
        file.write(f"Loss function = {loss_fn}\n")
        file.write(f"Optimizer = {optimizer}\n")
        file.write(f"Scheduler = {scheduler}\n")
        file.write(f"Dataset Size = {N_tot}\n")
        file.write("***************\n")

        results = pd.DataFrame(columns=["y_true", "y_pred", "idx", "set"])
        embeddings = pd.DataFrame(
            columns=["Molecule_ID"]
            + ["Set"]
            + [f"emb_{i}" for i in range(1, model.graph_embedding + 1)]
        )

        # Evaluate train, validation, and test sets
        results, y_true_train, y_pred_train, idx_train, embeddings = evaluate_and_log(
            model, train_loader, "train", log_dir, file, results, embeddings
        )
        results, y_true_val, y_pred_val, idx_val, embeddings = evaluate_and_log(
            model, val_loader, "val", log_dir, file, results, embeddings
        )
        results, y_true_test, y_pred_test, idx_test, embeddings = evaluate_and_log(
            model, test_loader, "test", log_dir, file, results, embeddings
        )

        # Handle regression parity plots
        if model.problem_type == "regression":
            create_parity_plot(results, log_dir)

        # Save results
        results.to_csv(log_dir / "predictions.csv", index=False)
        embeddings.to_csv(log_dir / "embeddings.csv", index=False)

        # Handle mispredictions for classification
        if model.problem_type == "classification":
            mispredicted = [
                i for i in range(len(y_true_test)) if y_true_test[i] != y_pred_test[i]
            ]
            if mispredicted:
                file.write("Mispredicted instances in test set:\n")
                for index in mispredicted:
                    file.write(
                        f"Mispredicted index = {idx_test[index]}. Real Value: {y_true_test[index]}. Predicted Value: {y_pred_test[index]}.\n"
                    )

        file.write("---------------------------------------------------------\n")

    return f"Report saved in {log_dir}"
