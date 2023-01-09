import math
import wave
from tkinter import filedialog
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import ifft, irfft
import simpleaudio as sa
import struct

matplotlib.use('TkAgg')  # to ignore warning message

RESULT_FILE_NAME = 'result.wav'


def chooseFile():
    return filedialog.askopenfilename(filetypes=(
        ("wav files", "*.wav"), ("all files", "*.*")))


def readWav(path):
    return wave.open(path, 'rb')


def playSound(file_name):
    wave_obj = sa.WaveObject.from_wave_file(file_name)
    wave_obj.play()


def getSegment(file, start_time, end_time):
    framerate = file.getframerate()
    n_frames = file.getnframes()
    frames = file.readframes(n_frames)
    frames_array = np.frombuffer(frames, dtype=np.int16)
    segment_frames_array = frames_array[math.floor(
        framerate * start_time):math.floor(framerate * end_time)]
    return segment_frames_array


def visualizeChannels(frames_array, times, l_audio=None, n_channels=1):
    if n_channels > 1:
        visualizeStereoChannels(frames_array, times, l_audio, n_channels)
        return
    visualizeMonoChannel(frames_array, times, l_audio)


def visualizeMonoChannel(values, times, length):
    plt.figure(figsize=(13, 4))
    plt.plot(times, values, zorder=1)
    plt.title('Channel')
    plt.ylabel('A')
    plt.xlabel('T(s)')
    if (length):
        plt.xlim(0, length)
    plt.show()


def visualizeStereoChannels(values, times, length, n_channels):
    figure, plot_channel = plt.subplots(n_channels, figsize=(13, 6))
    for i in range(n_channels):
        plot_channel[i].set_title(f"Channel #{i}")
        plot_channel[i].set_ylabel('A')
        plot_channel[i].set_xlabel('T(s)')
        plot_channel[i].plot(times, values[i::n_channels], zorder=1)
        if (length):
            plot_channel[i].set_xlim(0, length)

    figure.tight_layout()
    plt.show()


def applyHamming(frames_array):
    hamming = np.hamming(len(frames_array))
    return np.multiply(frames_array, hamming)


def plotOriginal(file_name):
    file = readWav(file_name)
    framerate = file.getframerate()
    n_frames = file.getnframes()
    n_channels = file.getnchannels()
    l_audio = n_frames / framerate
    b_signal = file.readframes(n_frames)
    v_signal = np.frombuffer(b_signal, dtype=np.int16)
    times = np.linspace(0, (n_frames - 1) / framerate, num=n_frames)
    visualizeChannels(v_signal, times, l_audio, n_channels)
    file.close()


def plotSegment(file_name):
    start_time = float(input('start time (s): '))
    end_time = float(input('duration (s): '))
    file = readWav(file_name)
    segment = getSegment(file, start_time, start_time + end_time)
    framerate = file.getframerate()
    n_frames = len(segment)
    l_audio = n_frames / framerate
    times = np.linspace(0, (n_frames - 1) / framerate, num=n_frames)
    visualizeChannels(segment, times, l_audio)


def plotSpectrumFunction(file_name):
    start_time = float(input('start time (s): '))
    end_time = float(input('duration (s): '))
    file = readWav(file_name)
    frames_array = getSegment(file, start_time, start_time + end_time)
    frames_array_windowed = applyHamming(frames_array)
    framerate = file.getframerate()

    dft = np.fft.fft(frames_array_windowed)
    dft_abs = np.abs(dft)

    if dft_abs.size % 2 == 0:
        cutted_dtf = dft_abs[:int((dft_abs.size / 2) + 1)]
        doubled_dft = []
        doubled_dft.append(cutted_dtf[0])
        for i in range(1, len(cutted_dtf) - 1):
            doubled_dft.append(cutted_dtf[i] * 2)
        doubled_dft.append(cutted_dtf[len(cutted_dtf) - 1])
    else:
        cutted_dtf = dft_abs[:int((dft_abs.size + 1) / 2)]
        doubled_dft = []
        doubled_dft.append(cutted_dtf[0])
        for i in range(1, len(cutted_dtf) - 1):
            doubled_dft.append(cutted_dtf[i] * 2)

    dft_frequences = np.fft.rfftfreq(
        len(doubled_dft) * 2 - 1, 1 / framerate)

    plt.figure(figsize=(15, 5))
    plt.plot(dft_frequences, doubled_dft)
    plt.title('Spectrum function')
    plt.ylabel('Power (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.show(block=False)


def cutDft(dft):
    cutted_result = []
    if dft.size % 2 == 0:
        cutted_result = dft[:int((dft.size + 1) / 2)]
    else:
        cutted_result = dft[:int((dft.size / 2) + 1)]
    doubled_result = [cutted_result[0]]
    doubled_result += [i*2 for i in cutted_result[1: len(cutted_result) - 1]]
    if dft.size % 2 == 1:
        doubled_result += cutted_result[len(cutted_result) - 1]
    return doubled_result


def removeFrequencies(file_name):
    target_frequency_from = int(input('target frequency from (Hz): '))
    target_frequency_to = int(input('target frequency to (Hz): '))
    file = wave.open(file_name, 'rb')
    frames = file.readframes(file.getnframes())
    framerate = file.getframerate()
    file.close()
    samples = np.frombuffer(frames, np.int16)
    n_channels = file.getnchannels()

    dft = np.fft.rfft(samples)

    frequencies = np.fft.rfftfreq(len(dft), 1 / framerate)
    frequencies = frequencies[0: len(frequencies) - 1]
    points_per_frequency = len(frequencies) / (framerate / 2)
    remove_from = int(points_per_frequency * target_frequency_from)
    remove_to = int(points_per_frequency * target_frequency_to)
    dft[remove_from:remove_to] = 0

    plt.figure(figsize=(12, 12))
    plt.plot(frequencies, cutDft(np.abs(dft)))
    plt.title('Spectrum function')
    plt.ylabel('Power (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.show(block=False)

    inverse_dft = np.fft.irfft(dft)
    inverse_dft = np.int16((inverse_dft / inverse_dft.max()) * 32767)

    wav_writer = wave.open(RESULT_FILE_NAME, 'wb')
    wav_writer.setnchannels(n_channels)
    wav_writer.setsampwidth(file.getsampwidth())
    wav_writer.setframerate(framerate)
    wav_writer.setnframes(len(inverse_dft))
    for sample in inverse_dft:
        wav_writer.writeframes(struct.pack('h', int(sample)))
    wav_writer.close()

    playSound(RESULT_FILE_NAME)

    times = np.linspace(0, len(inverse_dft) / framerate, num=len(inverse_dft))
    visualizeChannels(inverse_dft, times)


def interface(file_name):
    is_running = True

    while (is_running):
        print('1 - Plot original signal')
        print('2 - Plot a segment of signal')
        print('3 - Plot spectrum function of a segment')
        print('4 - Play sound')
        print('5 - Remove frequency')
        print('q - Quit program')

        user_input = input()

        if (user_input == '1'):
            plotOriginal(file_name)

        if (user_input == '2'):
            plotSegment(file_name)

        if (user_input == '3'):
            plotSpectrumFunction(file_name)

        if (user_input == '4'):
            playSound(file_name)

        if (user_input == '5'):
            removeFrequencies(file_name)

        if (user_input == 'q'):
            is_running = False


def main():
    file_name = chooseFile()

    interface(file_name)


main()
