import numpy as np
import os
import scipy.signal as signal
from scipy.io import loadmat
import matplotlib.pyplot as plt

file_dir = '/eeg/data_preprocessed_matlab_top5'


fs = 128                     # 降采样频率为128Hz
N = 8064                     # 每个试验每个电极采集了8064个数据点


eeg_signal_data = []              # 用于存储所有受试者所有电极下的脑电信号shape is [40 * 32,32,8064]
eeg_signal_label = []             # 用于存储所有受试者的情绪标签shape is [40 * 32,4]


def get_file_name(file_dir):

    file_names = []

    for root, dirs, files in os.walk(file_dir, topdown=False):
        for file in files:
            file_names.append(os.path.join(root, file))

    return file_names


def get_eeg_signal_four_frequency_band(eeg_signal):

    theta = [4, 8]
    alpha = [8, 12]
    beta = [12, 30]
    gamma = 30

    filter_theta = signal.firwin(N, theta, pass_zero='bandpass', fs=128)
    filter_alpha = signal.firwin(N, alpha, pass_zero='bandpass', fs=128)
    filter_beta = signal.firwin(N, beta, pass_zero='bandpass', fs=128)
    filter_gamma = signal.firwin(N+1, gamma, pass_zero='highpass', fs=128)  # firwin函数设计的都是偶对称的fir滤波器，当N为偶数时，其截止频率处即fs/2都是zero response的，所以用N+1

    eeg_signal_theta = np.expand_dims(signal.convolve(eeg_signal, filter_theta), 0)
    eeg_signal_alpha = np.expand_dims(signal.convolve(eeg_signal, filter_alpha), 0)
    eeg_signal_beta = np.expand_dims(signal.convolve(eeg_signal, filter_beta), 0)
    eeg_signal_gamma = np.expand_dims(signal.convolve(eeg_signal, filter_gamma, mode='same'), 0)

    return np.concatenate(
                (np.concatenate((eeg_signal_theta, eeg_signal_alpha), axis=0),
                 np.concatenate((eeg_signal_beta, eeg_signal_gamma), axis=0)), axis=0)


file_names = get_file_name(file_dir)
for index in range(len(file_names)):              # len(file_names)等于32，意为有32个受试者

    preprocessing_eeg = loadmat(file_names[index])                # x为字典型数据，存有data和label，x['data'].shape是[40,40,8064]，40个不同的视频试验，40个生理信号，128hz的采样频率持续采集63秒
    eeg_signal_data.append(preprocessing_eeg['data'][:][:32][:])  # deap数据集中前32个才是脑电信号，后8个是身体上的信号
    eeg_signal_label.append(preprocessing_eeg['label'])

PSD = np.zeros(shape=(eeg_signal_data.shape[0], eeg_signal_data.shape[1]))

for trial in len(eeg_signal_data[0]):
    for channel in len(eeg_signal_data[0][0]):

        '''
            此处待写滤波函数，从eeg_signal中滤出四个频段的的脑电信号
            theta (4Hz < f < 8Hz), alpha (8Hz < f < 12Hz),
            beta(12Hz < f < 30Hz) and gamma (30Hz < f)，
        '''
        eeg_four_frequency_band = get_eeg_signal_four_frequency_band(eeg_signal_data[trial][channel])

        for i in len(eeg_four_frequency_band):
            f, Pper_spec = signal.periodogram(eeg_four_frequency_band[i], fs, 'hamming', scaling='spectrum')
            Pper_spec = np.square(Pper_spec).sum()        # 数字信号，能量就是各点信号幅度值平方后的求和，至此求得一个电极下特定频率的能量
            PSD.append(Pper_spec)