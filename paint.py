import numpy as np
import scipy.signal as signal
from scipy.signal import hilbert, chirp
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

# fs = 128
# N = 8064
#
# file_dir = '/Users/yangyifeng/PycharmProjects/eeg/picture'
#
# eeg_signal = loadmat('/Users/yangyifeng/PycharmProjects/eeg/data_preprocessed_matlab_top5/s01.mat')
# eeg_signal_data = eeg_signal['data'][0][0]
#
# f, Pper_spec = signal.periodogram(eeg_signal_data, fs, 'hamming', scaling='spectrum')
#
# plt.figure(1)
# plt.plot(f, 20 * np.log10(abs(Pper_spec)))
# plt.xlabel('frequency [Hz]')
# plt.ylabel('Amplitude Response [dB]')
# plt.grid()
# plt.savefig(file_dir + '/eeg_signal_AmplitudeResponse.png', format='png')
# plt.show()
#
#
# theta = [4, 8]
# alpha = [8, 12]
# beta = [12, 30]
# gamma = 30
#
# four_frequency_band = ['theta', 'alpha', 'beta', 'gamma']
#
# filter_theta = signal.firwin(N, theta, pass_zero='bandpass', fs=128)
# filter_alpha = signal.firwin(N, alpha, pass_zero='bandpass', fs=128)
# filter_beta = signal.firwin(N, beta, pass_zero='bandpass', fs=128)
# filter_gamma = signal.firwin(N+1, gamma, pass_zero='highpass', fs=128)
#
# eeg_signal_data_theta = signal.convolve(eeg_signal_data, filter_theta)
# eeg_signal_data_alpha = signal.convolve(eeg_signal_data, filter_alpha)
# eeg_signal_data_beta = signal.convolve(eeg_signal_data, filter_beta)
# eeg_signal_data_gamma = signal.convolve(eeg_signal_data, filter_gamma, mode='same')
#
# eeg_four_frequency_band = np.array([eeg_signal_data_theta, eeg_signal_data_alpha, eeg_signal_data_beta, eeg_signal_data_gamma])
#
# for i in range(4):
#
#     f, Pper_spec = signal.periodogram(eeg_four_frequency_band[i], fs, 'hamming', scaling='spectrum')
#
#     plt.figure(i+1)
#     plt.plot(f, 20 * np.log10(abs(Pper_spec)))
#     plt.title('{}'.format(four_frequency_band[i]))
#     plt.xlabel('frequency [Hz]')
#     plt.ylabel('Amplitude Response [dB]')
#     plt.grid()
#     plt.savefig(file_dir + '/eeg_{}_AmplitudeResponse.png'.format(four_frequency_band[i]), format='png')
#     plt.show()

duration = 1.0
fs = 400.0
samples = int(fs*duration)
t = np.arange(samples) / fs

input_signal = chirp(t, 20.0, t[-1], 100.0)
input_signal *= (1.0 + 0.5 * np.sin(2.0*np.pi*3.0*t) )

analytic_signal = hilbert(input_signal)
amplitude_envelope = np.abs(analytic_signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)

fig = plt.figure(1)
ax0 = fig.add_subplot(311)
ax0.plot(t, input_signal, label='signal')
ax0.plot(t, amplitude_envelope, label='envelope')
ax0.set_xlabel("time in seconds")
ax0.legend()
ax1 = fig.add_subplot(312)
ax1.plot(t[1:], np.diff(instantaneous_phase))
ax1.set_xlabel("time in seconds")
ax1.set_ylabel("phase difference")
ax1.set_ylim(0.0, np.pi)
ax2 = fig.add_subplot(313)
ax2.plot(t[1:], instantaneous_frequency)
ax2.set_xlabel("time in seconds")
ax2.set_ylabel("frequency")
ax2.set_ylim(0.0, 120.0)

fig.show()
