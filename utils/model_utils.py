import torch
import os
import numpy as np
import pandas as pd
import csv
from datetime import date, datetime
from sklearn.metrics import r2_score, mean_absolute_error, \
    mean_squared_error, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from math import sqrt

from utils.plot_utils import create_training_plot
from icecream import ic



def train_network(model, train_loader, device):

    train_loss = 0
    model.train()

    for batch in train_loader:
        batch = batch.to(device)
        model.optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.batch)
        loss = model.loss(out, batch.y.long())
        loss.backward()
        model.optimizer.step()

        train_loss += loss.item() * batch.num_graphs

    return train_loss / len(train_loader.dataset)


def eval_network(model, loader, device):
    model.eval()
    loss = 0
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.batch)
        loss += model.loss(out, batch.y.long()).item() * batch.num_graphs
    return loss / len(loader.dataset)



def calculate_metrics(y_true:list, y_predicted: list,  task = 'regression'):

    if task == 'regression':
        r2 = r2_score(y_true=y_true, y_pred=y_predicted)
        mae = mean_absolute_error(y_true=y_true, y_pred=y_predicted)
        rmse = sqrt(mean_squared_error(y_true=y_true, y_pred=y_predicted))  
        error = [(y_predicted[i]-y_true[i]) for i in range(len(y_true))]
        prctg_error = [ abs(error[i] / y_true[i]) for i in range(len(error))]
        mbe = np.mean(error)
        mape = np.mean(prctg_error)
        error_std = np.std(error)
        metrics = [r2, mae, rmse, mbe, mape, error_std]
        metrics_names = ['R2', 'MAE', 'RMSE', 'Mean Bias Error', 'Mean Absolute Percentage Error', 'Error Standard Deviation']

    elif task == 'classification':
        accuracy = accuracy_score(y_true=y_true, y_pred=y_predicted)
        precision = precision_score(y_true=y_true, y_pred=y_predicted)
        recall = recall_score(y_true=y_true, y_pred=y_predicted)
        f1 = f1_score(y_true=y_true, y_pred=y_predicted)
        auroc = roc_auc_score(y_true=y_true, y_score=y_predicted)
        metrics = [accuracy, precision, recall, f1, auroc]
        metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUROC']

    return np.array(metrics), metrics_names


def predict_network(model, loader, return_emb = False):
    model.to('cpu')
    model.eval()

    y_pred, y_true, idx, embeddings = [], [], [], []

    for batch in loader:
        batch = batch.to('cpu')
        out = model(batch.x, batch.edge_index, batch.batch)

        out = torch.argmax(out, dim=1).cpu().detach().numpy()
        out = np.where(out == 1, 1, 0)

        y_pred.append(out)
        y_true.append(batch.y.cpu().detach().numpy())
        idx.append(batch.idx.cpu().detach().numpy())


    y_pred = np.concatenate(y_pred, axis=0).ravel()
    y_true = np.concatenate(y_true, axis=0).ravel()
    idx = np.concatenate(idx, axis=0).ravel()



    return y_pred, y_true, idx


def network_report(exp_name,
                   loaders,
                   loss_lists,
                   save_all,
                   model, 
                   ):


    #1) Create a directory to store the results
    log_dir = "{}/{}".format(exp_name,  model.name)
    os.makedirs(log_dir, exist_ok=True)

    #2) Time of the run
    today = date.today()
    today_str = str(today.strftime("%d-%b-%Y"))
    time = str(datetime.now())[11:]
    time = time[:8]
    run_period = "{}, {}\n".format(today_str, time)

    #3) Unfold loaders and save loaders and model
    train_loader,  test_loader = loaders[0],  loaders[1]
    N_train,  N_test = len(train_loader.dataset), len(test_loader.dataset)
    N_tot = N_train +  N_test     

    if save_all == True:
        torch.save(train_loader, "{}/train_loader.pth".format(log_dir))
        torch.save(model, "{}/model.pth".format(log_dir))


    torch.save(test_loader, "{}/test_loader.pth".format(log_dir)) 
    loss_function = str(model.loss)

    #4) loss trend during training
    train_list = loss_lists[0]
    test_list = loss_lists[1]
    if train_list is not None and test_list is not None:
        with open('{}/{}.csv'.format(log_dir, 'learning_process'), 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["Epoch", "Train_{}".format(loss_function), "Val_{}".format(loss_function), "Test_{}".format(loss_function)])
            for i in range(len(train_list)):
                writer.writerow([i+1, train_list[i],  test_list[i]])
        create_training_plot(df='{}/{}.csv'.format(log_dir, 'learning_process'), save_path='{}'.format(log_dir))


    #5) Start writting report
    file1 = open("{}/performance.txt".format(log_dir), "w")
    file1.write(run_period)
    file1.write("---------------------------------------------------------\n")
    file1.write("GNN TRAINING AND PERFORMANCE\n")
    file1.write("Dataset Size = {}\n".format(N_tot))
    file1.write("***************\n")


    y_pred, y_true, idx = predict_network(model, train_loader, True)

    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = model.problem_type)

    file1.write("Training set\n")
    file1.write("Set size = {}\n".format(N_train))

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))


    if model.problem_type == 'classification':
        #plot_confusion_matrix(y_true, y_pred, log_dir)
        mispredicted = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
        if len(mispredicted) > 0:
            file1.write("Mispredicted instances in train set:\n")
            for index in mispredicted:
                file1.write("Mispredicted index = {}. Real Value: {}. Predicted Value: {}.\n".format(idx[index], y_true[index], y_pred[index]))

    
    #file1.write("***************\n")
    #y_pred, y_true, idx, emb_val = predict_network(model, val_loader, True)
    #emb_val['set'] = 'val'
    #metrics, metrics_names = calculate_metrics(y_true, y_pred, task = 'r')

    #file1.write("Validation set\n")
    #file1.write("Set size = {}\n".format(N_val))

    #for name, value in zip(metrics_names, metrics):
    #    file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("***************\n")

    y_pred, y_true, idx = predict_network(model, test_loader, True)


    #plot_tsne_with_subsets(data_df=emb_all, feature_columns=[i for i in range(128)], color_column='ddG_exp', set_column='set', fig_name='tsne_emb_exp', save_path=log_dir)
    #emb_all.to_csv("{}/embeddings.csv".format(log_dir))

    pd.DataFrame({'y': y_true, 'ŷ': y_pred, 'index': idx}).to_csv("{}/predictions_test_set.csv".format(log_dir))

    file1.write("Test set\n")
    file1.write("Set size = {}\n".format(N_test))


    metrics, metrics_names = calculate_metrics(y_true, y_pred, task = model.problem_type)

    for name, value in zip(metrics_names, metrics):
        file1.write("{} = {:.3f}\n".format(name, value))

    file1.write("---------------------------------------------------------\n")

    if model.problem_type == 'classification':
        #plot_confusion_matrix(y_true, y_pred, log_dir)
        mispredicted = [i for i in range(len(y_true)) if y_true[i] != y_pred[i]]
        if len(mispredicted) > 0:
            file1.write("Mispredicted instances in test set:\n")
            for index in mispredicted:
                file1.write("Mispredicted index = {}. Real Value: {}. Predicted Value: {}.\n".format(idx[index], y_true[index], y_pred[index]))

    #create_st_parity_plot(real = y_true, predicted = y_pred, figure_name = 'outer_{}_inner_{}'.format(outer, inner), save_path = "{}".format(log_dir))




    file1.close()

    return 'Report saved in {}'.format(log_dir)