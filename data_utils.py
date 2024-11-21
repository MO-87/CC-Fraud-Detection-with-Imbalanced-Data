import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter


def under_sampling(X, y):
    ctr = Counter(y)

    rus = RandomUnderSampler(sampling_strategy={0: int(ctr[0] / 2)}, random_state=42)
    X_us, y_us = rus.fit_resample(X, y)

    return X_us, y_us


def over_sampling(X, y):

    ctr = Counter(y)

    # ros = RandomOverSampler(sampling_strategy={1: ctr[1] * 10}, random_state=42)
    # X_os, y_os = ros.fit_resample(X, y)

    smote_os = SMOTE(sampling_strategy={1: ctr[1] * 10}, random_state=42, k_neighbors=3)
    X_os, y_os = smote_os.fit_resample(X, y)

    return X_os, y_os


def data_loader(path):
    data = pd.read_csv(path)

    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    return X, y


def data_preprocessing(X_train, X_val):
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    return X_train, X_val, scaler
