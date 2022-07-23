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


def init_data_file(user_id='055'):
    sequence_file = open(os.path.join(DATA_ROOT, user_id, sequence_file_name), 'w', newline='')
    feature_file = open(os.path.join(DATA_ROOT, user_id, feature_file_name), 'w', newline='')
    BP_file = open(os.path.join(DATA_ROOT, user_id, BP_file_name), 'w', newline='')
    return sequence_file, feature_file, BP_file


# 读取文件
# def read_data_file(user_id='055', nrows=None):
#     data_file = os.path.join(DATA_ROOT, 'origin', user_id + '.csv')
#     raw_data = pd.read_csv(data_file, skiprows=2, names=['ABP', 'PLETH'], nrows=nrows, dtype=np.float32)
#     return raw_data
#

def save_row_to_file(row, file):
    csv_write = csv.writer(file)
    csv_write.writerow(row)


# 截取一个波谷，向前获取60s（6000条pleth和bp的记录）
def get_valid_samples(raw_data):
    min_peaks = argrelextrema(raw_data['PLETH'].values, np.less)[0]
    i = 0
    while min_peaks[i] < WINDOW_SIZE:
        i += 1
    while i < len(min_peaks) - 3:
        end = min_peaks[i] + 4
        start = end - WINDOW_SIZE
        final_cycle_length = (end - min_peaks[i - 2]) + 4
        i += 1
        pleth = raw_data.iloc[start:end, 1].values
        ABP = raw_data.iloc[start:end, 0].values
        # 检验该周期的脉搏波是否有效，无效则舍弃，有效则进行降噪
        if not FE_lib.legal_pulse_wave(pleth[-final_cycle_length:]):
            continue
        blood_pressure = FE_lib.getBP(ABP)
        if not FE_lib.legal_blood_pressure(blood_pressure):
            continue
        filtered_data = pleth

        # 提取出特征和得到目标ABP和SBP
        features = FE_lib.extractFeature(filtered_data[-final_cycle_length:])

        # 保存 序列数据，特征数据，目标ABP和SBP
        save_row_to_file(filtered_data, sequence_file)
        save_row_to_file(features, feature_file)
        save_row_to_file(blood_pressure, BP_file)


if __name__ == '__main__':
    WINDOW_SIZE = 60 * 100
    chunksize = 1280000
    users = ['055', '224']
    for user_id in users:
        print('start to handle user {}'.format(user_id))
        sequence_file, feature_file, BP_file = init_data_file(user_id)
        data_file = os.path.join(DATA_ROOT, 'origin', user_id + '.csv')
        reader = pd.read_csv(data_file, skiprows=2, names=['ABP', 'PLETH'], chunksize=chunksize, dtype=np.float32)
        for raw_data in reader:
            print('start to traverse a new chunk!')
            get_valid_samples(raw_data)
        sequence_file.close()
        feature_file.close()
        BP_file.close()
