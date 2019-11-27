import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

fs = 128
N = 8064

file_dir = '/Users/yangyifeng/PycharmProjects/eeg/picture'

eeg_signal = loadmat('/Users/yangyifeng/PycharmProjects/eeg/data_preprocessed_matlab_top5/s01.mat')
eeg_signal_data = eeg_signal['data'][0][0]

f, Pper_spec = signal.periodogram(eeg_signal_data, fs, 'hamming', scaling='spectrum')

plt.figure(1)
plt.plot(f, 20 * np.log10(abs(Pper_spec)))
plt.xlabel('frequency [Hz]')
plt.ylabel('Amplitude Response [dB]')
plt.grid()
plt.savefig(file_dir + '/eeg_signal_AmplitudeResponse.png', format='png')
plt.show()


theta = [4, 8]
alpha = [8, 12]
beta = [12, 30]
gamma = 30

four_frequency_band = ['theta', 'alpha', 'beta', 'gamma']

filter_theta = signal.firwin(N, theta, pass_zero='bandpass', fs=128)
filter_alpha = signal.firwin(N, alpha, pass_zero='bandpass', fs=128)
filter_beta = signal.firwin(N, beta, pass_zero='bandpass', fs=128)
filter_gamma = signal.firwin(N+1, gamma, pass_zero='highpass', fs=128)

eeg_signal_data_theta = signal.convolve(eeg_signal_data, filter_theta)
eeg_signal_data_alpha = signal.convolve(eeg_signal_data, filter_alpha)
eeg_signal_data_beta = signal.convolve(eeg_signal_data, filter_beta)
eeg_signal_data_gamma = signal.convolve(eeg_signal_data, filter_gamma, mode='same')

eeg_four_frequency_band = np.array([eeg_signal_data_theta, eeg_signal_data_alpha, eeg_signal_data_beta, eeg_signal_data_gamma])

for i in range(4):

    f, Pper_spec = signal.periodogram(eeg_four_frequency_band[i], fs, 'hamming', scaling='spectrum')

    plt.figure(i+1)
    plt.plot(f, 20 * np.log10(abs(Pper_spec)))
    plt.title('{}'.format(four_frequency_band[i]))
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Amplitude Response [dB]')
    plt.grid()
    plt.savefig(file_dir + '/eeg_{}_AmplitudeResponse.png'.format(four_frequency_band[i]), format='png')
    plt.show()
