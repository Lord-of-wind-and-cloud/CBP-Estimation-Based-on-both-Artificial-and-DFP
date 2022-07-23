import numpy as np
import matplotlib.pyplot as plt
import csv
from scipy.signal import argrelextrema
from scipy import interpolate
from scipy import integrate
import os
from PyEMD import EEMD


def readDenoisedFile(filename):
    # column1 : PLETH
    # column2 : ABP
    INPUT_LEN = 3000
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        c1 = []
        c2 = []
        for row in reader:
            c1.append(float(row[1]))
            c2.append(float(row[2]))

    return np.array(c1), np.array(c2)


def readFeatureScope(filename):
    # scope[0] is the array of upper bound;
    # scope[1] is the array of lower limit;
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        s = list(reader)
        scope = []
        for row in s:
            scope.append([float(e) for e in row])
        return np.array(scope)


def legal_pulse_wave(y):
    maxx_peaks = argrelextrema(y, np.greater)[0]
    minn_peaks = argrelextrema(y, np.less)[0]
    # 通过峰值数量确定基本形状
    if len(minn_peaks) != 3 or len(maxx_peaks) != 2:
        return False

    min1 = y[minn_peaks[0]]
    max1 = y[maxx_peaks[0]]
    min2 = y[minn_peaks[1]]
    max2 = y[maxx_peaks[1]]
    min3 = y[minn_peaks[2]]
    sAmp = max1 - min1
    dAmp = max1 - min3

    # 进一步限制基本形状
    # 1. 确保重搏波波峰谷不为最值
    if not (min1 < min2 < max2 < max1):
        return False
    # 2. 保证脉搏波起始高度和水平高度相近
    if min3 > (min1 + 0.1 * sAmp) or min3 < (min1 - 0.1 * sAmp):
        return False

    # 限制重搏波位置
    # 1. 初步限制
    if min2 < min1 or min2 < min3 or max2 > max1:
        return False
    # 2. 限制波谷
    if min2 > min3 + 0.5 * dAmp or min2 < min3 + 0.1 * dAmp:
        return False
    # 3. 限制波峰
    if max2 < min3 + 0.25 * dAmp or max2 > min3 + 0.9 * dAmp:
        return False

    return True


def legal_blood_pressure(BP):
    return BP[0] < 165 and BP[1] > 30


def getAbscissa(f, value, lo, hi, up=1, accuracy=0.01):
    # 二分法取中间值
    # up ==  1 时 [lo:hi] 为f的上升区间
    # up == -1 时 [lo:hi] 为f的下降区间
    if lo > hi: return f(lo)
    while hi - lo > accuracy:
        mid = (hi + lo) / 2
        v = f(mid)
        if (v > value and up == 1) or (v < value and up == -1):
            hi = mid
        else:
            lo = mid
    return (hi + lo) / 2


def denoise_data_with_EEMD(unfilter_data, window_size=3000):
    # filter process
    imfs = EEMD().eemd(unfilter_data)
    Emd_out = np.zeros(window_size, )

    k = 2
    if np.shape(imfs)[0] == 12:
        m = 7
    else:
        m = 6
    while k < np.shape(imfs)[0] - m:
        Emd_out += imfs[k, :]
        k = k + 1
    return Emd_out


def extractFeature(y):
    AMP_PERCENT = [0.1, 1 / 4, 1 / 3, 1 / 2, 2 / 3, 3 / 4, 9 / 10]

    feature = np.zeros(61)
    end = len(y)
    fragment_t = range(end)
    maxx_peaks = argrelextrema(y, np.greater)[0]
    minn_peaks = argrelextrema(y, np.less)[0]

    min1 = minn_peaks[0]
    max1 = maxx_peaks[0]
    min2 = minn_peaks[1]
    max2 = maxx_peaks[1]
    min3 = minn_peaks[2]
    f = interpolate.interp1d(fragment_t, y, kind='cubic')
    # height feature
    amplitude = y[max1] - y[min1]
    dAmp = y[max1] - y[min3]
    feature[0] = y[min1]
    feature[1] = y[max1]
    feature[2] = y[min2]
    feature[3] = y[min3]
    feature[4] = y[max2] - y[min2]

    # time feature
    feature[5] = max1 - min1
    for i in range(len(AMP_PERCENT)):  # [0,7)
        # feature[6-12]
        feature[6 + i] = max1 - getAbscissa(f, AMP_PERCENT[i] * amplitude + y[min1], min1, max1)

    feature[13] = min3 - max1
    feature[14] = getAbscissa(f, 0.1 * dAmp + y[min3], max2, min3, up=-1) - max1
    feature[15] = getAbscissa(f, 1 / 4 * dAmp + y[min3], max2, min3, up=-1) - max1
    feature[16] = min2 - max1
    for i in range(3, 7):
        # feature[17-20]
        feature[14 + i] = getAbscissa(f, AMP_PERCENT[i] * amplitude + y[min3], max1, min2, up=-1) - max1

    # feature[21-28]
    feature[21] = feature[13] / feature[5]
    feature[22] = feature[14] / feature[6]
    feature[23] = feature[15] / feature[7]
    feature[24] = feature[17] / feature[9]
    feature[25] = feature[18] / feature[10]
    feature[26] = feature[19] / feature[11]
    feature[27] = feature[20] / feature[12]

    p = getAbscissa(f, y[min2], max2, min3, up=-1)
    a = (y[max2] + y[min2]) / 2.0
    feature[28] = max2 - min2
    feature[29] = max2 - getAbscissa(f, a, min2, max2)
    feature[30] = p - max2
    feature[31] = getAbscissa(f, a, max2, p, up=-1) - max2
    feature[32] = feature[30] / feature[28]
    feature[33] = feature[31] / feature[29]

    # slope feature
    k_s = [(5, 6, 0.1), (5, 7, 1 / 4), (6, 7, 0.15), (7, 8, 1 / 3 - 1 / 4), (8, 9, 1 / 2 - 1 / 3),
           (9, 10, 2 / 3 - 1 / 2), (10, 11, 3 / 4 - 2 / 3), (11, 12, 9 / 10 - 3 / 4), (12, -1, 0.1)]
    for i in range(9):
        # feature[35-43]
        feature[34 + i] = amplitude * k_s[i][2] / (feature[k_s[i][0]] - feature[k_s[i][1]])
    feature[43] = amplitude * 0.1 / feature[12]

    feature[44] = dAmp * 0.1 / feature[20]
    feature[45] = dAmp * 0.25 / feature[19]
    feature[46] = dAmp * (9 / 10 - 3 / 4) / (feature[19] - feature[20])
    feature[47] = dAmp * (0.15) / (feature[14] - feature[15])
    feature[48] = dAmp * (0.25) / (feature[13] - feature[15])
    feature[49] = dAmp * (0.1) / (feature[13] - feature[14])

    feature[50] = feature[4] * 0.5 / (feature[28] - feature[29])
    feature[51] = feature[4] * 0.5 / (feature[29])
    feature[52] = feature[4] * 0.5 / (feature[30] - feature[31])
    feature[53] = feature[4] * 0.5 / (feature[31])

    # acreage feature
    feature[54] = integrate.quad(f, min1, max1)[0] - (max1 - min1) * y[min1]
    feature[55] = integrate.quad(f, max1, min2)[0] - (min2 - max1) * y[min3]
    feature[56] = integrate.quad(f, min2, max2)[0] - (max2 - min2) * y[min3]
    feature[57] = integrate.quad(f, max2, min3)[0] - (min3 - max2) * y[min3]
    feature[58] = feature[54] / (feature[55] + feature[56] + feature[57])
    feature[59] = feature[56] / feature[57]
    V_m = (feature[54] + feature[55] + feature[56] + feature[57]) / (min3 - min1)
    feature[60] = (V_m - y[min1]) / (y[max1] - y[min1])
    return feature


def getBP(ABP):
    return np.array([ABP[np.argmax(ABP)], ABP[np.argmin(ABP)]])


def saveFeature(ft, ft_file, bp, bp_file):
    with open(ft_file, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(ft)
    with open(bp_file, 'a+', newline='') as f:
        csv_write = csv.writer(f)
        csv_write.writerow(bp)
