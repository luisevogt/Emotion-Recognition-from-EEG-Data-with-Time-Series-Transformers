import numpy as np
import pandas as pd
import seaborn
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq


def plot_signal(signal: np.ndarray, sec: np.ndarray, channel_names: str):
    assert signal.shape[0] == len(channel_names)
    inf_index = np.where(signal == np.NINF)
    col_index = inf_index[1][0]

    signal_length = col_index
    time = np.linspace(0, sec, signal_length)

    df_dict = {ch_name: sig[0:signal_length] for ch_name, sig in zip(channel_names, signal)}
    df_dict['time in seconds'] = time

    df = pd.DataFrame(df_dict)

    seaborn.lineplot(x='time in seconds', y='amplitude', hue='channel',
                     data=pd.melt(df, ['time in seconds'], var_name='channel', value_name='amplitude'))

    plt.show()


def plot_fft(signal: np.ndarray, sec: np.ndarray, channel_names: str):
    assert signal.shape[0] == len(channel_names)
    inf_index = np.where(signal == np.NINF)
    col_index = inf_index[1][0]

    signal_length = col_index
    T = signal_length / sec
    frequency = fftfreq(signal_length, 1 / T)

    df_dict = {ch_name: np.real(fft(sig[0:signal_length])) for ch_name, sig in zip(channel_names, signal)}
    df_dict['frequency'] = frequency

    df = pd.DataFrame(df_dict)

    seaborn.lineplot(x='frequency', y='amplitude', hue='channel',
                     data=pd.melt(df, ['frequency'], var_name='channel', value_name='amplitude'))

    plt.show()


def plot_patient(signals: np.ndarray, sample_frequency: int, channel_names: [str]):
    signal = signals[0]
    seconds = signal.shape[1] / sample_frequency

    plot_fft(signal, seconds, channel_names)


def main():
    data = np.load('../data/DREAMER_data.npy')
    patient_signals = data[0:18]
    sample_freq = 128
    channel_names = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']

    plot_patient(patient_signals, sample_freq, channel_names)


if __name__ == '__main__':
    main()
