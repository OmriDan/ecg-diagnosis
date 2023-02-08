import argparse
import os
import pickle
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from resnet import resnet34
from dataset import ECGDataset
from utils import cal_scores, find_optimal_threshold, split_data
from preprocess import *

def split_data_classify(seed=42):
    folds = range(1, 11)
    return folds

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='classify_data', help='Directory to data dir') #change dir
    parser.add_argument('--leads', type=str, default='all', help='ECG leads to use')
    parser.add_argument('--seed', type=int, default=42, help='Seed to split data')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of workers to load data')
    parser.add_argument('--use-gpu', default=False, action='store_true', help='Use gpu')
    parser.add_argument('--model-path', type=str, default='', help='Path to saved model')
    return parser.parse_args()


def get_thresholds(val_loader, net, device, threshold_path):
    print('Finding optimal thresholds...')
    if os.path.exists(threshold_path):
        return pickle.load(open(threshold_path, 'rb'))
    output_list, label_list = [], []
    for _, (data, label) in enumerate(tqdm(val_loader)):
        data, labels = data.to(device), label.to(device)
        output = net(data)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    thresholds = []
    for i in range(y_trues.shape[1]):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        threshold = find_optimal_threshold(y_true, y_score)
        thresholds.append(threshold)
    # pickle.dump(thresholds, open(threshold_path, 'wb'))
    return thresholds


def apply_thresholds(test_loader, net, device, thresholds):
    output_list, label_list = [], []
    for _, (data, label) in enumerate(tqdm(test_loader)):
        data, labels = data.to(device), label.to(device)
        output = net(data)
        output = torch.sigmoid(output)
        output_list.append(output.data.cpu().numpy())
        label_list.append(labels.data.cpu().numpy())
    y_trues = np.vstack(label_list)
    y_scores = np.vstack(output_list)
    y_preds = []
    scores = [] 
    for i in range(len(thresholds)):
        y_true = y_trues[:, i]
        y_score = y_scores[:, i]
        y_pred = (y_score >= thresholds[i]).astype(int)
        scores.append(cal_scores(y_true, y_pred, y_score))
        y_preds.append(y_pred)
    y_preds = np.array(y_preds).transpose()
    scores = np.array(scores)

    return y_trues


if __name__ == "__main__":
    dx_dict = {
        '426783006': 'SNR', # Normal sinus rhythm
        '39732003': 'LAF', # left axis deviation
        '164934002': 'TWA',  # t wave abnormal
        '445118002': 'LAFB', #left anterior fascicular block
        '164889003': 'AF', # atrial fibrillation
        '713426002': 'IRBBB', # incomplete right bundle branch block
        '427084000': 'ST', # sinus tachycardia
    }
    classes = ['SNR', 'LAF', 'TWA', 'LAFB', 'AF', 'IRBBB', 'ST']
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/WFDB', help='Directory to dataset')
    args = parser.parse_args()
    data_dir = args.data_dir
    reference_csv = os.path.join(data_dir, 'reference.csv')
    label_csv = os.path.join(data_dir, 'labels.csv')
    gen_reference_csv(data_dir, reference_csv)
    gen_label_csv(label_csv, reference_csv, dx_dict, classes)


    args = parse_args()
    data_dir = os.path.normpath(args.data_dir)
    database = os.path.basename(data_dir)
    if not args.model_path:
        args.model_path = f'models/resnet34_{database}_{args.leads}_{args.seed}.pth'
    args.threshold_path = f'models/{database}-threshold.pkl'
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
    data_dir = args.data_dir
    label_csv = os.path.join(data_dir, 'labels.csv')
    
    net = resnet34(input_channels=nleads).to(device)
    net.load_state_dict(torch.load(args.model_path, map_location=device))
    net.eval()

    test_folds = split_data_classify(seed=args.seed)
    test_dataset = ECGDataset('test', data_dir, label_csv, test_folds, leads)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    thresholds = pickle.load(open(args.threshold_path, 'rb'))

    print('Results on test data:')
    y_trues=apply_thresholds(test_loader, net, device, thresholds)


