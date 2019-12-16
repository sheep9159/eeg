import numpy as np
import os
import scipy.signal as signal
from scipy.io import loadmat, savemat

file_dir = r'E:\study\Projects_data\eeg2.0\data_preprocessed_matlab'
processed_data_dir = r'E:\study\Projects_data\eeg2.0\processed_data'

fs = 128                     # 降采样频率为128Hz
N = 8064                     # 数据点长度为8064

PSD = []                          # 用于存储所有的脑电极下所有频段的功率shape is [32, 40, 32, 4]
eeg_signal_label = []             # 用于存储所有受试者的情绪标签shape is [32, 40, 4]

PSD_shape = [32, 40, 32, 4]
eeg_signal_label_shape = [32, 40, 4]

SUBJECTS = 32                     # 32代表有32个受试者
TRIALS = 40                       # 40代表40个视频刺激试验
CHANNEL = 32                      # 32代表32个脑电极通道
BANDS = 4                         # 4代表4个不同的频段


def get_matfile_name(file_dir):

    file_names = []

    for root, dirs, files in os.walk(file_dir, topdown=False):
        for file in files:
            if '.mat' in file:
                file_names.append(os.path.join(root, file))

    return file_names


def get_eeg_signal_four_frequency_band(eeg_signal_data):

    theta = [4, 8]
    alpha = [8, 12]
    beta = [12, 30]
    gamma = 30

    filter_theta = signal.firwin(N, theta, pass_zero='bandpass', fs=128)
    filter_alpha = signal.firwin(N, alpha, pass_zero='bandpass', fs=128)
    filter_beta = signal.firwin(N, beta, pass_zero='bandpass', fs=128)

    '''
        firwin函数设计的都是偶对称的fir滤波器，当N为偶数时，其截止频率处即fs/2都是zero response的，所以用N+1
    '''
    filter_gamma = signal.firwin(N+1, gamma, pass_zero='highpass', fs=128)

    eeg_signal_data_theta = signal.convolve(eeg_signal_data, filter_theta)
    eeg_signal_data_alpha = signal.convolve(eeg_signal_data, filter_alpha)
    eeg_signal_data_beta = signal.convolve(eeg_signal_data, filter_beta)
    eeg_signal_data_gamma = signal.convolve(eeg_signal_data, filter_gamma, mode='same')  # 得到fir数字滤波器后直接与信号做卷积

    return np.array([eeg_signal_data_theta, eeg_signal_data_alpha, eeg_signal_data_beta, eeg_signal_data_gamma])  # 将四个频段组合在一起


file_names = get_matfile_name(file_dir)                  # 扫描装有格式为.mat的脑电数据的文件夹，得到所有的脑电信号文件

for index in range(SUBJECTS):

    '''
        x为字典型数据，存有data和label，x['data'].shape是[40,40,8064]，
        40个不同的视频试验，40个生理信号，128hz的采样频率持续采集63秒
    '''
    preprocessing_eeg = loadmat(file_names[index])
    eeg_signal_label.append(preprocessing_eeg['labels'])

    for trial in range(TRIALS):
        for channel in range(CHANNEL):

            '''
                此处待写滤波函数，从eeg_signal中滤出四个频段的的脑电信号
                theta (4Hz < f < 8Hz), alpha (8Hz < f < 12Hz),
                beta(12Hz < f < 30Hz) and gamma (30Hz < f)，
            '''
            eeg_four_frequency_band = get_eeg_signal_four_frequency_band(preprocessing_eeg['data'][trial][channel])

            for i in range(BANDS):
                f, Pper_spec = signal.periodogram(eeg_four_frequency_band[i], fs, 'hamming', scaling='spectrum')
                Pper_spec = Pper_spec.sum()              # 数字信号，至此求得一个电极下特定频率的能量
                PSD.append(Pper_spec)

            total_energy = sum(PSD[index][trial][channel])
            for i in range(BANDS):
                PSD[index][trial][channel][i] = PSD[index][trial][channel][i] / total_energy  # 求得每个电极下特定频率所占的能量比

    print('{} down'.format(file_names[index]))


PSD = np.array(PSD).reshape(PSD_shape)
eeg_signal_label = np.array(eeg_signal_label).reshape(eeg_signal_label_shape)

savemat(processed_data_dir + '\\PSD.mat', {'PSD': PSD})
savemat(processed_data_dir + '\\eeg_signal_label.mat', {'eeg_signal_label': eeg_signal_label})