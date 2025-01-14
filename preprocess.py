import wfdb
import numpy as np
import pandas as pd

from glob import glob
import argparse
import os


def gen_reference_csv(data_dir, reference_csv):
    if not os.path.exists(reference_csv):
        recordpaths = glob(os.path.join(data_dir, '*.hea'))
        results = []
        for recordpath in recordpaths:
            patient_id = recordpath.split('/')[-1][:-4]
            _, meta_data = wfdb.rdsamp(recordpath[:-4])
            sample_rate = meta_data['fs']
            signal_len = meta_data['sig_len']
            age = meta_data['comments'][0]
            sex = meta_data['comments'][1]
            dx = meta_data['comments'][2]
            age = age[5:] if age.startswith('Age: ') else np.NaN
            sex = sex[5:] if sex.startswith('Sex: ') else 'Unknown'
            dx = dx[4:] if dx.startswith('Dx: ') else ''
            results.append([patient_id, sample_rate, signal_len, age, sex, dx])
        df = pd.DataFrame(data=results, columns=['patient_id', 'sample_rate', 'signal_len', 'age', 'sex', 'dx'])
        df.sort_values('patient_id').to_csv(reference_csv, index=None)


def gen_label_csv(label_csv, reference_csv, dx_dict, classes):
    if not os.path.exists(label_csv):
        results = []
        df_reference = pd.read_csv(reference_csv)
        for _, row in df_reference.iterrows():
            patient_id = row['patient_id']
            dxs = [dx_dict.get(code, '') for code in row['dx'].split(',')]
            labels = [0] * 7
            for idx, label in enumerate(classes):
                if label in dxs:
                    labels[idx] = 1
            results.append([patient_id] + labels)
        df = pd.DataFrame(data=results, columns=['patient_id'] + classes)
        n = len(df)
        folds = np.zeros(n, dtype=np.int8)
        for i in range(10):
            start = int(n * i / 10)
            end = int(n * (i + 1) / 10)
            folds[start:end] = i + 1
        df['fold'] = np.random.permutation(folds)
        columns = df.columns
        df['keep'] = df[classes].sum(axis=1)
        df = df[df['keep'] > 0]
        df[columns].to_csv(label_csv, index=None)

def init_preprocess(data_dir):
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
    reference_csv = os.path.join(data_dir, 'reference.csv')
    label_csv = os.path.join(data_dir, 'labels.csv')
    gen_reference_csv(data_dir, reference_csv)
    gen_label_csv(label_csv, reference_csv, dx_dict, classes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data/WFDB', help='Directory to dataset')
    args = parser.parse_args()
    data_dir = args.data_dir
    init_preprocess(data_dir)
