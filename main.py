import argparse
import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
import matplotlib.pyplot as plt
from predict import *
from tqdm import tqdm
import numpy as np

from dataset import ECGDataset
from resnet import resnet34
from utils import cal_f1s, cal_aucs, split_data, accuracy_score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/CPSC', help='Directory for data dir')
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--num-classes', type=int, default=int, help='Num of diagnostic classes')
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Num of workers to load data')
    parser.add_argument('--phase', type=str, default='train', help='Phase: train or test')
    parser.add_argument('--epochs', type=int, default=40, help='Training epochs')
    parser.add_argument('--resume', default=False, action='store_true', help='Resume')
    parser.add_argument('--use-gpu', default=True, action='store_true', help='Use GPU')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()


def train(dataloader, net, args, criterion, epoch, scheduler, optimizer, device):
    print('Training epoch %d:' % epoch)
    net.train()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    # scheduler.step()
    print(f'Avg Loss per epoch:{running_loss/(len(tqdm(dataloader)))}')
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1s = cal_f1s(y_trues, y_scores)
    avg_f1 = np.mean(f1s)
    #y_train_accuracy = y_scores > 0.5  # Ask Ariel if OK
    #accuracies = accuracy_score(y_trues, y_train_accuracy)
    #avg_accuracy = np.mean(accuracies)
    output_loss = running_loss/len(output_list)
    return output_loss, avg_f1#, avg_accuracy


def evaluate(dataloader, net, args, criterion, device):
    print('Validating...')
    net.eval()
    running_loss = 0
    output_list, labels_list = [], []
    for _, (data, labels) in enumerate(tqdm(dataloader)):
        data, labels = data.to(device), labels.to(device)
        output = net(data)
        loss = criterion(output, labels)
        running_loss += loss.item()
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        labels_list.append(labels.data.cpu().numpy())
    print(f'Avg Loss per epoch:{running_loss/(len(output_list))}')
    y_trues = np.vstack(labels_list)
    y_scores = np.vstack(output_list)
    f1s = cal_f1s(y_trues, y_scores)
    avg_f1 = np.mean(f1s)
    y_val_accuracy = y_scores > 0.5  # Ask Ariel if OK
    accuracies = accuracy_score(y_trues, y_val_accuracy)
    avg_accuracy = np.mean(accuracies)
    output_loss = running_loss/len(output_list)
    print('F1s:', f1s)
    print('Avg F1: %.4f' % avg_f1)
    if args.phase == 'train' and avg_f1 > args.best_metric:
        args.best_metric = avg_f1
        torch.save(net.state_dict(), args.model_path)
    else:
        aucs = cal_aucs(y_trues, y_scores)
        avg_auc = np.mean(aucs)
        print('AUCs:', aucs)
        print('Avg AUC: %.4f' % avg_auc)
    return output_loss, avg_f1, avg_accuracy


def calc_weights(df, dataloader, lambda_param=0.01):
    df = df.labels.iloc[:, 1:8]
    label_counts = df.sum()
    label_weights = np.array(1/label_counts)
    sample_weights = [0] * len(tqdm(dataloader))
    for idx, (data, label) in enumerate(tqdm(dataloader)):
        sample_weights[idx] = np.dot(label_weights, label)
    #regularization_term = lambda_param * sum(sample_weights ** 2)
    #sample_weights = [1/label_counts[i] for i in range(len(df.columns))]
    return sample_weights


def plot_metrics(all_metrics_dict):
    epoch_vec = np.arange(args.epochs)

    # F1
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(epoch_vec, all_metrics_dict["train_f1"], epoch_vec, all_metrics_dict["val_f1"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("F1 score")
    ax.set_title('F1 as a function of epochs')
    ax.legend(['train', 'val'])
    fig.savefig("Sampled_f1.png")

    # Accuracy
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(epoch_vec, all_metrics_dict["val_accuracy"])
    ax.set_xlabel("epoch")
    ax.set_title('Accuracy as a function of epochs')
    fig.savefig("Sampled_accuracy.png")

    # Loss
    fig, ax = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    ax.plot(epoch_vec, all_metrics_dict["train_loss"], epoch_vec, all_metrics_dict["val_loss"])
    ax.set_xlabel("epoch")
    ax.set_ylabel("Loss")
    ax.legend(['train', 'val'])
    ax.set_title('Loss as a function of epochs')
    fig.savefig("Sampled_loss.png")


if __name__ == "__main__":
    args = parse_args()
    args.best_metric = 0
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)

    if not args.model_path:
        args.model_path = f'models/resnet34_{database}_{args.leads}_{args.seed}.pth'

    if args.use_gpu and torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = 'cpu'
    
    if args.leads == 'all':
        leads = 'all'
        nleads = 12
    else:
        leads = args.leads.split(',')
        nleads = len(leads)
    label_csv = os.path.join(data_dir, 'labels.csv')
    train_folds, val_folds, test_folds = split_data(seed=args.seed)
    train_dataset = ECGDataset('train', data_dir, label_csv, train_folds, leads)
    train_weights = calc_weights(train_dataset, train_dataset)
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_weights, len(train_weights), replacement=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                              pin_memory=True, sampler=train_sampler)
    val_dataset = ECGDataset('val', data_dir, label_csv, val_folds, leads)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=True)
    test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True)
    net = resnet34(input_channels=nleads).to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)

    criterion = nn.BCEWithLogitsLoss()
    #train_accuracy = []
    val_accuracy = []
    train_loss = []
    val_loss = []
    train_f1 = []
    val_f1 = []
    if args.phase == 'train':
        if args.resume:
            net.load_state_dict(torch.load(args.model_path, map_location=device))
        for epoch in range(args.epochs):
            curr_train_loss, curr_train_f1 = train(train_loader, net, args, criterion, epoch,
                                                                        scheduler, optimizer, device)
            curr_val_loss, curr_val_f1, curr_val_accuracy = evaluate(val_loader, net, args, criterion, device)
            train_loss.append(curr_train_loss)
            val_loss.append(curr_val_loss)
            #train_accuracy.append(curr_train_accuracy)
            val_accuracy.append(curr_val_accuracy)
            train_f1.append(curr_train_f1)
            val_f1.append(curr_val_f1)
    else:
        net.load_state_dict(torch.load(args.model_path, map_location=device))
        evaluate(test_loader, net, args, criterion, device)
    all_metrics_dict = {'val_accuracy': val_accuracy, 'train_f1': train_f1,
                        'val_f1': val_f1, 'train_loss': train_loss, 'val_loss': val_loss}
    plot_metrics(all_metrics_dict)
