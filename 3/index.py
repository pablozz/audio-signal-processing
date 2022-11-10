import wave
from tkinter import filedialog
import matplotlib.pyplot as plt
import numpy as np
import simpleaudio as sa

RESULT_FILE_NAME = 'result.wav'
LATENCY = 0.01  # in seconds


def chooseFile():
    return filedialog.askopenfilename(filetypes=(
        ("wav files", "*.wav"), ("all files", "*.*")))


def readWav(path):
    return wave.open(path, 'rb')


def playSound(file_name):
    wave_obj = sa.WaveObject.from_wave_file(file_name)
    wave_obj.play()


def visualizeChannels(file):
    framerate = file.getframerate()
    n_frames = file.getnframes()
    n_channels = file.getnchannels()
    l_audio = n_frames / framerate
    b_signal = file.readframes(n_frames)
    v_signal = np.frombuffer(b_signal, dtype=np.int16)
    times = np.linspace(0, (n_frames - 1) / framerate, num=n_frames)
    if n_channels > 1:
        visualizeStereoChannels(v_signal, n_channels, times, l_audio)
        return
    visualizeMonoChannel(v_signal, times, l_audio)


def visualizeMonoChannel(values, times, length):
    plt.figure(figsize=(13, 4))
    plt.plot(times, values, zorder=1)
    plt.title('Channel')
    plt.ylabel('A')
    plt.xlabel('T(s)')
    plt.xlim(0, length)
    plt.show()


def visualizeStereoChannels(values, n_channels, times, length):
    figure, plot_channel = plt.subplots(n_channels, figsize=(13, 6))
    for i in range(n_channels):
        plot_channel[i].set_title(f"Channel #{i}")
        plot_channel[i].set_ylabel('A')
        plot_channel[i].set_xlabel('T(s)')
        plot_channel[i].plot(times, values[i::n_channels], zorder=1)
        plot_channel[i].set_xlim(0, length)

    figure.tight_layout()
    plt.show()


def processFile(file):
    framerate = file.getframerate()
    n_frames = file.getnframes()
    frames = file.readframes(n_frames)
    frames_array = np.frombuffer(frames, dtype=np.int16)

    latency_n_frames = framerate * LATENCY
    latency_frames_array = np.repeat(0, latency_n_frames).astype(np.int16)

    first_channel_frames_array = np.concatenate(
        (frames_array, latency_frames_array))
    second_channel_frames_array = np.concatenate(
        (latency_frames_array, frames_array))

    result_frames_array = np.ravel(
        [first_channel_frames_array, second_channel_frames_array], 'F')

    result_frames = result_frames_array.tobytes()

    wav_writer = wave.open(RESULT_FILE_NAME, 'wb')
    wav_writer.setnchannels(2)
    wav_writer.setsampwidth(file.getsampwidth())
    wav_writer.setframerate(framerate)
    wav_writer.writeframes(result_frames)
    wav_writer.close()


def interface(file_name):
    is_running = True
    while (is_running):
        print('1 - Plot original signal')
        print('2 - Plot processed signal')
        print('3 - Play original signal')
        print('4 - Play processed signal')
        print('q - Quit program')

        user_input = input()

        if (user_input == '1'):
            file = readWav(file_name)
            visualizeChannels(file)
            file.close()

        elif (user_input == '2'):
            file = readWav(file_name)

            processFile(file)

            result_file = readWav(RESULT_FILE_NAME)
            visualizeChannels(result_file)

            file.close()
            result_file.close()

        elif (user_input == '3'):
            playSound(file_name)

        elif (user_input == '4'):
            file = readWav(file_name)

            processFile(file)
            playSound(RESULT_FILE_NAME)

            file.close()

        elif (user_input == 'q'):
            is_running = False


def main():
    file_name = chooseFile()

    interface(file_name)


main()
