import csv
import os
from datetime import date, datetime
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from icecream import ic
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_absolute_percentage_error, precision_score,
                             r2_score, recall_score, roc_auc_score)

from utils.plot_utils import (create_parity_plot, create_training_plot,
                              plot_confusion_matrix)


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


def predict_network(model, loader, return_emb=False):
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


def network_report(
    exp_name,
    loaders,
    loss_lists,
    save_all,
    model,
):

    # 1) Create a directory to store the results
    log_dir = exp_name / model.name / "results_model"
    log_dir.mkdir(parents=True, exist_ok=True)

    # 2) Time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)

    # 3) Unfold loaders and save loaders and model
    train_loader, val_loader, test_loader = loaders[0], loaders[1], loaders[2]
    N_train, N_val, N_test = (
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )
    N_tot = N_train + N_test

    if save_all == True:
        torch.save(train_loader, log_dir / "train_loader.pth")
        torch.save(val_loader, log_dir / "val_loader.pth")
        torch.save(model, log_dir / "model.pth")

    torch.save(test_loader, log_dir / "test_loader.pth")
    loss_function = str(model.loss)

    # 4) loss trend during training
    train_list = loss_lists[0]
    val_list = loss_lists[1]
    test_list = loss_lists[2]
    lr_list = loss_lists[3]

    if train_list is not None and test_list is not None:
        with open(log_dir / "learning_process.csv", "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(
                [
                    "Epoch",
                    "Learning Rate",
                    "Train_{}".format(loss_function),
                    "Val_{}".format(loss_function),
                    "Test_{}".format(loss_function),
                ]
            )
            for i in range(len(train_list)):
                writer.writerow(
                    [i + 1, lr_list[i], train_list[i], val_list[i], test_list[i]]
                )
        create_training_plot(
            df=log_dir / "learning_process.csv",
            save_path=log_dir,
        )

    # 5) Start writting report
    file1 = open(log_dir / "performance.txt", "w")
    file1.write(run_period)
    file1.write("---------------------------------------------------------\n")
    file1.write("GNN TRAINING AND PERFORMANCE\n")
    file1.write("Dataset Size = {}\n".format(N_tot))
    file1.write("***************\n")

    results = pd.DataFrame(columns=["y_true", "y_pred", "idx", "set"])

    y_pred, y_true, idx, y_score = predict_network(model, train_loader, True)
    results = pd.DataFrame(
        {"y_true": y_true, "y_pred": y_pred, "idx": idx, "set": "train"}
    )

    metrics = calculate_metrics(
        y_true=y_true,
        y_predicted=y_pred,
        task=model.problem_type,
        num_classes=model.num_classes,
        y_score=y_score,
    )

    file1.write("Training set\n")
    file1.write("Set size = {}\n".format(N_train))

    for name, value in metrics.items():
        file1.write("{} = {:.3f}\n".format(name, value))

    if model.problem_type == "classification":
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            title="Train Set",
            save_path=log_dir / "Confusion_matrix_train.png",
        )
        mispredicted = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
        if len(mispredicted) > 0:
            file1.write("Mispredicted instances in train set:\n")
            for index in mispredicted:
                file1.write(
                    "Mispredicted index = {}. Real Value: {}. Predicted Value: {}.\n".format(
                        idx[index], y_true[index], y_pred[index]
                    )
                )

    file1.write("***************\n")

    y_pred, y_true, idx, y_score = predict_network(model, val_loader, True)
    results = results.append(
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "idx": idx, "set": "val"})
    )

    metrics = calculate_metrics(
        y_true=y_true,
        y_predicted=y_pred,
        task=model.problem_type,
        num_classes=model.num_classes,
        y_score=y_score,
    )

    file1.write("Validation set\n")
    file1.write("Set size = {}\n".format(N_val))

    for name, value in metrics.items():
        file1.write("{} = {:.3f}\n".format(name, value))

    if model.problem_type == "classification":
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            title="Validation Set",
            save_path=log_dir / "Confusion_matrix_validation.png",
        )

        mispredicted = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
        if len(mispredicted) > 0:
            file1.write("Mispredicted instances in train set:\n")
            for index in mispredicted:
                file1.write(
                    "Mispredicted index = {}. Real Value: {}. Predicted Value: {}.\n".format(
                        idx[index], y_true[index], y_pred[index]
                    )
                )

    file1.write("***************\n")

    y_pred, y_true, idx, y_score = predict_network(model, test_loader, True)
    results = results.append(
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "idx": idx, "set": "test"})
    )

    if model.problem_type == "regression":
        create_parity_plot(results, log_dir)

    # plot_tsne_with_subsets(data_df=emb_all, feature_columns=[i for i in range(128)], color_column='ddG_exp', set_column='set', fig_name='tsne_emb_exp', save_path=log_dir)
    # emb_all.to_csv("{}/embeddings.csv".format(log_dir))

    results.to_csv("{}/predictions.csv".format(log_dir))

    file1.write("Test set\n")
    file1.write("Set size = {}\n".format(N_test))

    metrics = calculate_metrics(
        y_true=y_true,
        y_predicted=y_pred,
        task=model.problem_type,
        num_classes=model.num_classes,
        y_score=y_score,
    )

    for name, value in metrics.items():
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("---------------------------------------------------------\n")

    if model.problem_type == "classification":
        plot_confusion_matrix(
            y_true=y_true,
            y_pred=y_pred,
            title="Test Set",
            save_path=log_dir / "Confusion_matrix_test.png",
        )
        mispredicted = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
        if len(mispredicted) > 0:
            file1.write("Mispredicted instances in test set:\n")
            for index in mispredicted:
                file1.write(
                    "Mispredicted index = {}. Real Value: {}. Predicted Value: {}.\n".format(
                        idx[index], y_true[index], y_pred[index]
                    )
                )

    # create_st_parity_plot(real = y_true, predicted = y_pred, figure_name = 'outer_{}_inner_{}'.format(outer, inner), save_path = "{}".format(log_dir))

    file1.close()

    return "Report saved in {}".format(log_dir)
