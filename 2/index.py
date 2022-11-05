import wave
import tkinter as tk
from tkinter import filedialog
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

ENERGY_THRESHOLD = 0.02
NKS_THRESHHOLD = 0.1


def chooseFile():
    return filedialog.askopenfilename(filetypes=(
        ("wav files", "*.wav"), ("all files", "*.*")))


def readWav(path):
    return wave.open(path, 'rb')


def processFile(file):
    framerate = file.getframerate()
    n_frames = file.getnframes()
    n_channels = file.getnchannels()
    l_audio = n_frames / framerate
    b_signal = file.readframes(n_frames)
    v_signal = np.frombuffer(b_signal, dtype=np.int16)
    norm_values = v_signal / v_signal.max()
    times = np.linspace(0, (n_frames - 1) / framerate, num=n_frames)

    visualizeChannels(norm_values, n_channels, times, l_audio)

    l_frame = int(framerate / 1000 * 25)
    l_hop = int(l_frame / 2)

    energy = getEnergy(norm_values, l_frame, l_hop)
    visualize(energy, l_audio, 'T(s)', 'Energy',
              'Sound Energy', ENERGY_THRESHOLD)

    zcr = getZeroCrossRate(norm_values, l_frame, l_hop)
    visualize(zcr, l_audio, 'T(s)', 'ZCR',
              'Zero Cross Rate', NKS_THRESHHOLD)


def visualizeChannels(norm_values, n_channels, times, length):
    if n_channels > 1:
        visualizeStereoChannels(norm_values, n_channels, times, length)
        return
    visualizeMonoChannel(norm_values, times, length)


def visualizeMonoChannel(norm_values, times, length):
    plt.figure(figsize=(13, 4))
    plt.plot(times, norm_values, zorder=1)
    plt.title('Channel')
    plt.ylabel('A')
    plt.xlabel('T(s)')
    plt.xlim(0, length)
    plt.show()


def visualizeStereoChannels(norm_values, n_channels, times, length):
    figure, plotChannel = plt.subplots(n_channels, figsize=(13, 6))
    for i in range(n_channels):
        plotChannel[i].set_title(f"Channel #{i}")
        plotChannel[i].set_ylabel('A')
        plotChannel[i].set_xlabel('T(s)')
        plotChannel[i].plot(times, norm_values[i::n_channels], zorder=1)
        plotChannel[i].set_xlim(0, length)
        print('hey')

    figure.tight_layout()
    plt.show()


def visualize(data, length, x_label, y_label, title, threshold):
    time = np.linspace(0, length, num=len(data))
    plt.figure(figsize=(13, 4))
    plt.plot(time, data)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axhline(threshold, color='r')
    plt.show()


def getEnergy(signal_values, l_frame, l_hop):
    energy = []
    for i in range(0, len(signal_values), l_hop):
        energy.append(
            np.sum(signal_values[i: i + l_frame] ** 2) / l_frame)

    return energy


def sign(s):
    if s >= 0:
        return 1
    return -1


def getZeroCrossRate(signal_values, l_frame, l_hop):
    zcr = []
    for i in range(0, len(signal_values), l_hop):
        frame_zcr = 0
        for f in range(i, i + l_hop):
            if f < len(signal_values):
                frame_zcr += abs(sign(signal_values[f]) -
                                 sign(signal_values[f - 1]))
        zcr.append(frame_zcr / (2 * l_frame))

    return zcr


def main():
    root = tk.Tk()
    root.withdraw()
    file_name = chooseFile()
    root.update()
    file = readWav(file_name)
    processFile(file)


main()
