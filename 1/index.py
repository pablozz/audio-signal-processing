import tkinter as tk
from tkinter import filedialog
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np


def choose_file():
    return filedialog.askopenfilename()


def get_dimension(array, dimension):
    try:
        return array[:, dimension]
    except:
        return array


def read_wav(file_name):
    sample_rate, data = wavfile.read(file_name)
    try:
        channels_no = data.shape[1]
    except:
        channels_no = 1
    length = data.shape[0] / sample_rate
    return channels_no, length, data


def plot(channels_count, length, data, marker_time):
    time = np.linspace(0, length - 1, data.shape[0])
    for channel_index in range(channels_count):
        plt.figure(channel_index, figsize=(7, 5))
        plt.plot(time,
                 get_dimension(data, channel_index),
                 label=f"channel #{channel_index}")
        if (marker_time):
            plt.plot(marker_time, 0, 'ro', label="marker")
        plt.legend()
        plt.xlabel("t")
        plt.ylabel("s(t)")
    plt.show()


def main():
    marker_time = float(input('Time in seconds to mark:\n'))
    root = tk.Tk()
    root.withdraw()
    file_name = choose_file()
    root.update()
    channels_count, length, data = read_wav(file_name)
    plot(channels_count, length, data, marker_time)


main()
