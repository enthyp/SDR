import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal
from scipy.fftpack import fftshift


def lowpass(input_signal, sampling_freq_hz, cutoff_freq_hz, N=10):
    b, a = signal.butter(N=N, Wn=cutoff_freq_hz, fs=sampling_freq_hz, btype='low', analog=False)
    return signal.lfilter(b, a, input_signal)


def bandpass(input_signal, sampling_freq_hz, lower_freq_hz, upper_freq_hz, N=10):
    b, a = signal.butter(N=N, Wn=[lower_freq_hz, upper_freq_hz], fs=sampling_freq_hz, btype='band', analog=False)
    return signal.lfilter(b, a, input_signal)


def welch(samples, sample_rate, nper=1024, fsize=(20, 10)):
    f, Pxx = signal.welch(samples, sample_rate, nperseg=nper, detrend=lambda x: x)
    f, Pxx = fftshift(f), fftshift(Pxx)
    ind = np.argsort(f)
    f, Pxx = np.take_along_axis(f, ind, axis=0), np.take_along_axis(Pxx, ind, axis=0)
    
    plt.figure(figsize=fsize)
    plt.semilogy(f/1e3, Pxx)
    plt.xlabel('f [kHz]')
    plt.ylabel('PSD [Power/Hz]')
    plt.grid()

    plt.xticks(np.linspace(-sample_rate/2e3, sample_rate/2e3, 31))
    plt.xlim(-sample_rate/2e3, sample_rate/2e3)


def plot_signals(signals, start=0, width=None, figsize=(20, 10), normalize=True):
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    for s in signals:
        n_s = s / max(np.abs(s)) if normalize else s
        ax.plot(n_s[start:-1 if width is None else start + width])


def plot_signal_with_highlighted(signal, selected_indexes, start=0, width=None):
    fig, ax = plt.subplots()
    fig.set_size_inches((10, 5))
    
    ax.plot(signal[start:start + width])
    
    selected_indexes_sub = [idx for idx in selected_indexes if start <= idx <= start + width]
    ax.plot([idx - start for idx in selected_indexes_sub], [signal[idx] for idx in selected_indexes_sub], '*')
