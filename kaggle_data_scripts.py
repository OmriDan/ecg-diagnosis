import physionet_challenge_utility_script as pc
import ecg_plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras_preprocessing.sequence import pad_sequences
from keras import layers
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization, Add
from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPool1D, ZeroPadding1D, LSTM, Bidirectional
from keras.models import Sequential, Model
from keras.utils import plot_model
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.layers import concatenate
from scipy import optimize
from scipy.io import loadmat
import os


if __name__ == "__main__":
    data_path = r"C:\Data_for_Physionet\Data\WFDB"
    gender, age, labels, ecg_filenames = pc.import_key_data(data_path)
    ecg_filenames = np.asarray(ecg_filenames)
    SNOMED_scored = pd.read_csv(r"C:\ecg-diagnosis\SNOMED_mappings_scored.csv", sep=";")
    SNOMED_unscored = pd.read_csv(r"C:\ecg-diagnosis\SNOMED_mappings_unscored.csv", sep=";")
    df_labels = pc.make_undefined_class(labels, SNOMED_unscored)
    y, snomed_classes = pc.onehot_encode(df_labels)
    pc.plot_classes(snomed_classes, SNOMED_scored,y)
    y_all_comb = pc.get_labels_for_all_combinations(y)
    print("Total number of unique combinations of diagnosis: {}".format(len(np.unique(y_all_comb))))
    folds = pc.split_data(labels, y_all_comb)
    pc.plot_all_folds(folds,y,snomed_classes)

