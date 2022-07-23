import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import torch


def mean_absolute_percentage_error(y_true, y_pred):
    return torch.mean(torch.abs((y_true - y_pred) / y_true)).numpy()


def result_evaluate(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return mape, mae, rmse


def evaluate_with_normed_bp(bp, y_true, y_pred):
    if bp == 'SBP':
        y_pred = y_pred * 80 + 80
        y_true = y_true * 80 + 80
    else:
        y_pred = y_pred * 50 + 50
        y_true = y_true * 50 + 50
    return result_evaluate(y_true, y_pred)
