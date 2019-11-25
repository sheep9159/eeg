import numpy as np
import scipy.signal as signal
from scipy.fftpack import fft, rfft, fftshift
import matplotlib.pyplot as plt

# fs = 10e3
# N = 1e5
# amp = 2*np.sqrt(2)
# freq = 1270.0                            # fs和freq都是赫兹频率
# noise_power = 0.001 * fs / 2
# time = np.arange(N) / fs                 # 采样间隔Ts = 1 / fs，以下的x直接就是采样后的x
# x = amp*np.sin(2*np.pi*freq*time)        # w = 2 * np.pi * freq，对于其他各种信号如何表示，如冲激信号，矩形脉冲信号，直流信号等
# x += np.random.normal(scale=np.sqrt(noise_power), size=time.shape)
#
# f, Pper_spec = signal.periodogram(x, fs, 'hamming', scaling='spectrum')
#
# plt.figure(1)
# plt.semilogy(f, Pper_spec)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD')
# plt.grid()
# plt.show()
#
# print('len(Pper_spec) is {}, and len(x) is {}'.format(len(Pper_spec), len(x)))  # len(Pper_spec) is 50001, and len(x) is 100000，频谱只有一半的数据量

# x = np.array([1, 1, 1, 1])
# y = fftshift(x)                                 # 实值信号的离散傅立叶变换
# print(y)

# fs = 6
# N_1 = 64
# N_2 = 512
#
# time_1 = np.arange(N_1) / fs
# time_2 = np.arange(N_2) / fs
# x_1 = 2 * np.cos(2 * time_1) + 0.5 * np.sin(12 * time_1)
# x_2 = 2 * np.cos(2 * time_2) + 0.5 * np.sin(12 * time_2)
#
# f, Pper_spec = signal.periodogram(x_1, fs, 'hamming', scaling='spectrum')
#
# plt.figure(1)
# plt.plot(f, Pper_spec)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD')
# plt.grid()
# plt.show()
#
# f, Pper_spec = signal.periodogram(x_2, fs, 'hamming', scaling='spectrum')
#
# plt.figure(2)
# plt.plot(f, Pper_spec)
# plt.xlabel('frequency [Hz]')
# plt.ylabel('PSD')
# plt.grid()
# plt.show()