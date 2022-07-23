import os
import pandas as pd
import numpy as np
from scipy.signal import argrelextrema
import FE_lib
import csv

DATA_ROOT = r'G:\PythonPro\BP_prediction\data'
sequence_file_name = 'input_sequence.csv'
feature_file_name = 'input_feature.csv'
BP_file_name = 'target_BP.csv'
scope_file_name = 'feature_scope.csv'
selected_feature_list = [1, 2, 3, 4, 5, 7, 9, 10, 13, 15, 16, 17, 19, 28, 29, 30, 31, 54, 55, 56, 57]


def read_data(nrows=None):
    sequence_data = pd.read_csv(sequence_file, dtype=np.float32, header=None, nrows=nrows)
    feature_data = pd.read_csv(feature_file, dtype=np.float32, header=None, nrows=nrows)
    BP_data = pd.read_csv(BP_file, dtype=np.float32, names=['SBP', 'DBP'], header=None, nrows=nrows)
    return sequence_data, feature_data, BP_data


def data_clean(sequence_data, feature_data, bp_data):
    invalid_row_index = bp_data[(bp_data.SBP > 165) | (bp_data.DBP < 30)].index
    sequence_data = sequence_data.drop(invalid_row_index)
    feature_data = feature_data.drop(invalid_row_index)
    bp_data = bp_data.drop(invalid_row_index)
    # 只保留所需特征
    feature_data = feature_data.iloc[:, selected_feature_list]
    feature_data.columns = range(len(selected_feature_list))
    # 删除不合理feature的记录
    print(1)
    feature_data = feature_data.replace([np.inf, -np.inf], np.nan)
    feature_data = feature_data.dropna()
    sequence_data = sequence_data.loc[feature_data.index]
    bp_data = bp_data.loc[feature_data.index]

    # 删除不合理sequence的记录\
    print(2)
    sequence_data = sequence_data.replace([np.inf, -np.inf], np.nan)
    sequence_data = sequence_data.dropna()
    feature_data = feature_data.loc[sequence_data.index]
    bp_data = bp_data.loc[sequence_data.index]
    return sequence_data, feature_data, bp_data


def save_dataframe2file(df, file):
    df.to_csv(file, header=None, index=None)


def data_shuffle_split2multiplefiles(sequence_data, feature_data, bp_data, user_id='055'):
    # 打乱数据
    sample_num = bp_data.shape[0]
    permutation = np.random.permutation(sample_num)
    sequence_data = sequence_data.reset_index(drop=True).reindex(permutation)
    feature_data = feature_data.reset_index(drop=True).reindex(permutation)
    bp_data = bp_data.reset_index(drop=True).reindex(permutation)
    training_sample_num = sample_num * 3 // 5
    # 对feature做归一化至[0,1], 记录最值
    scope_records = pd.DataFrame(np.zeros([2, len(selected_feature_list)], dtype=np.float32),
                                 index=['min', 'max'])
    scope_records.loc['min'] = feature_data.iloc[:training_sample_num].min()
    scope_records.loc['max'] = feature_data.iloc[:training_sample_num].max()
    scope_records.to_csv(scope_file, index=True, header=False)

    for i in range(len(selected_feature_list)):
        feature_data.iloc[:, i] = feature_data.iloc[:, i].map(
            lambda x: (x - scope_records.loc['min', i]) / (scope_records.loc['max', i] - scope_records.loc['min', i]))

    save_dataframe2file(sequence_data.iloc[:training_sample_num],
                        os.path.join(DATA_ROOT, user_id, 'train_' + sequence_file_name))
    save_dataframe2file(feature_data.iloc[:training_sample_num],
                        os.path.join(DATA_ROOT, user_id, 'train_' + feature_file_name))
    save_dataframe2file(bp_data.iloc[:training_sample_num],
                        os.path.join(DATA_ROOT, user_id, 'train_' + BP_file_name))

    validation_sample_num = sample_num // 5
    save_dataframe2file(sequence_data.iloc[training_sample_num:training_sample_num + validation_sample_num],
                        os.path.join(DATA_ROOT, user_id, 'valid_' + sequence_file_name))
    save_dataframe2file(feature_data.iloc[training_sample_num:training_sample_num + validation_sample_num],
                        os.path.join(DATA_ROOT, user_id, 'valid_' + feature_file_name))
    save_dataframe2file(bp_data.iloc[training_sample_num:training_sample_num + validation_sample_num],
                        os.path.join(DATA_ROOT, user_id, 'valid_' + BP_file_name))

    test_sample_num = sample_num // 5
    save_dataframe2file(sequence_data.iloc[-test_sample_num:],
                        os.path.join(DATA_ROOT, user_id, 'test_' + sequence_file_name))
    save_dataframe2file(feature_data.iloc[-test_sample_num:],
                        os.path.join(DATA_ROOT, user_id, 'test_' + feature_file_name))
    save_dataframe2file(bp_data.iloc[-test_sample_num:],
                        os.path.join(DATA_ROOT, user_id, 'test_' + BP_file_name))

    save_dataframe2file(sequence_data.iloc[:-test_sample_num],
                        os.path.join(DATA_ROOT, user_id, 'except_test_' + sequence_file_name))
    save_dataframe2file(feature_data.iloc[:-test_sample_num],
                        os.path.join(DATA_ROOT, user_id, 'except_test_' + feature_file_name))
    save_dataframe2file(bp_data.iloc[:-test_sample_num],
                        os.path.join(DATA_ROOT, user_id, 'except_test_' + BP_file_name))


if __name__ == '__main__':
    users = ['055', '224']
    for user_id in users:
        print('start to handle user {}'.format(user_id))
        sequence_file = os.path.join(DATA_ROOT, user_id, sequence_file_name)
        feature_file = os.path.join(DATA_ROOT, user_id, feature_file_name)
        BP_file = os.path.join(DATA_ROOT, user_id, BP_file_name)
        scope_file = os.path.join(DATA_ROOT, user_id, scope_file_name)

        sequence_data, feature_data, BP_data = read_data()
        sequence_data, feature_data, BP_data = data_clean(sequence_data, feature_data, BP_data)
        # save_dataframe2file(sequence_data, sequence_file)
        # save_dataframe2file(feature_data, feature_file)
        # save_dataframe2file(BP_data, BP_file)
        data_shuffle_split2multiplefiles(sequence_data, feature_data, BP_data)
