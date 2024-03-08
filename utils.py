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


def read_iq(f_name):
    real_sample = np.fromfile(f_name, dtype=np.int8)
    return real_sample[::2] + 1j * real_sample[1::2]


def read_iq_gqrx(f_name):
    real_sample = np.fromfile(f_name, dtype=np.float32)
    return real_sample[::2] + 1j * real_sample[1::2]
    

def frequency_shift(samples, offset, sample_rate):
    t_s = np.linspace(0, len(samples) / sample_rate, len(samples))
    return samples * np.exp(2 * np.pi * offset * t_s * 1j)


def psd(signal, n_samples, offset=0):
  signal = signal[offset:offset + n_samples] * np.hamming(n_samples)
  fft = np.fft.fftshift(np.fft.fft(signal))
  norm_mag_fft = 1 / n_samples * np.abs(fft)
  return 20 * np.log10(norm_mag_fft)


    
def plot_signals(signals, start=0, width=None, figsize=(20, 10), normalize=True):
    fig, ax = plt.subplots()
    fig.set_size_inches(figsize)
    for s in signals:
        n_s = s / max(np.abs(s)) if normalize else s
        ax.plot(n_s[start:-1 if width is None else start + width])
    

def plot_psd(psd_samples, fs):
  f = np.arange(-fs/2, fs/2, fs/len(psd_samples))
  plt.plot(f, psd_samples)
  plt.xlabel('Frequency [Hz]')
  plt.ylabel('PSD [dB]')


def plot_correlations(correlations, start=0, width=None, max_indexes=None):
    fig, ax = plt.subplots()
    fig.set_size_inches((20, 10))
    ax.plot(correlations[start:-1 if width is None else start + width])
    if max_indexes is not None:
        indexes_on_plot = list(filter(lambda idx: idx >= start and (True if width is None else idx <= start + width), max_indexes))
        ax.plot(indexes_on_plot, [correlations[idx] for idx in indexes_on_plot], '*')


# find best approximating indices in source for target values
# so, e.g. find_positions(x-axis labels we want, x-axis values in plot)
#
# O(n) because assumes both source and target sorted ascending
def find_positions(source, target):
  positions = []
  cur_target_i = 0
  cur_approx_i = 0

  i = 0
  while i < len(source) and cur_target_i < len(target):
    new_diff = np.abs(target[cur_target_i] - source[i])
    if new_diff <= np.abs(target[cur_target_i] - source[cur_approx_i]):
      cur_approx_i = i
    else:
      positions.append(cur_approx_i)
      cur_target_i += 1
      i -= 1
    i += 1

  return positions


def plot_spectrogram(signal, fs, chunk=1024):
  t_step = 100
  rows = [psd(signal[i:i + chunk], chunk) for i in range(0, len(signal), chunk * t_step)]

  fig, ax = plt.subplots(1,1)
  ax.imshow(np.stack(rows))
  
  freq_order = 10 ** round(np.log10(fs / 2))
  xlabels_pos = np.arange(0, fs / 2, freq_order / 10)
  xlabels = np.concatenate([-np.flip(xlabels_pos)[:-1], xlabels_pos])
  freqs = (np.arange(0, chunk) - chunk // 2) * fs / chunk
  
  xticks = find_positions(freqs, xlabels)
  ax.xaxis.set_major_locator(ticker.FixedLocator(xticks))
  ax.set_xticklabels((xlabels / 1000).astype(int))
  ax.set_xlabel('Frequency [kHz]')
  
  t_max = len(signal) / fs
  time_order = 10 ** round(np.log10(t_max))
  ylabels = np.arange(0, t_max, t_max / 10).astype(int)
  
  times = np.arange(0, t_max, chunk * t_step / fs)
  yticks = find_positions(times, ylabels)
  ax.yaxis.set_major_locator(ticker.FixedLocator(yticks))
  ax.set_yticklabels(ylabels)
  ax.set_ylabel('Time [s]')

