import tkinter as tk
from tkinter import filedialog
import wave
import numpy
import struct
from numpy import ndarray
import matplotlib.pyplot as plt

window = tk.Tk()
window.withdraw()


def openFile():
    filePath = filedialog.askopenfilename(filetypes=[("Sound files", '*.wav')])
    if not filePath == '':
        return filePath


def readSound(fileName):
    file = wave.open(fileName, 'rb')
    frames = file.readframes(file.getnframes())
    samplingRate = file.getframerate()
    nchannels = file.getnchannels()
    sampleWidth = file.getsampwidth()
    file.close()
    samples = numpy.frombuffer(frames, numpy.int16)
    return samples, samplingRate, nchannels, sampleWidth


def writeFile():
    return filedialog.asksaveasfilename(filetypes=[("Sound files", '*.wav')])


def writeSound(fileName, samples, sampleRate, nchannels, sampwidth):
    file = wave.open(fileName, 'w')
    file.setnchannels(nchannels)
    file.setsampwidth(sampwidth)
    file.setframerate(sampleRate)
    file.setnframes(len(samples))
    file.setcomptype("NONE", "not compressed")
    for sample in samples:
        file.writeframes(struct.pack('h', int(sample)))
    file.close()
    return None


def createSignalDiagram(title, duration, data, xlabel, ylabel):
    durationArray = numpy.linspace(0, duration, num=len(data))
    plt.figure(title, figsize=(12, 12))
    plt.plot(durationArray, data)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show(block=False)


def createFourierDiagram(frequencies, fourier):
    plt.figure(figsize=(12, 12))
    plt.plot(frequencies, fourier)
    plt.title('Signalo spektras')
    plt.ylabel('Garsumas, dB')
    plt.xlabel('Daznis, Hz')
    plt.show(block=False)


def cropFourier(fourier):
    croppedResult = []
    if fourier.size % 2 == 0:
        croppedResult = fourier[:int((fourier.size + 1) / 2)]
    else:
        croppedResult = fourier[:int((fourier.size / 2) + 1)]
    doubledResult = [croppedResult[0]]
    doubledResult += [i*2 for i in croppedResult[1: len(croppedResult) - 1]]
    if fourier.size % 2 == 1:
        doubledResult += croppedResult[len(croppedResult) - 1]
    return doubledResult


def removeFrequencies(samplingRate, frequencies, fourier):
    print("Iveskite dazniu intervala, kuri noretumete pasalinti is garso iraso")
    pointsPerFrequency = len(frequencies) / (samplingRate / 2)
    frequencyToRemoveMin = int(pointsPerFrequency*int(input()))
    frequencyToRemoveMax = int(pointsPerFrequency*int(input()))
    fourier[frequencyToRemoveMin:frequencyToRemoveMax] = 0
    return fourier


def addFrequencies(samplingRate, frequencies, fourier):
    print("Iveskite dazniu intervala, kuri noretumete prideti prie garso iraso")
    pointsPerFrequency = len(frequencies) / (samplingRate / 2)
    frequencyToAddMin = int(pointsPerFrequency*int(input()))
    frequencyToAddMax = int(pointsPerFrequency*int(input()))
    fourier[frequencyToAddMin:frequencyToAddMax] = fourier.max()
    return fourier


path = openFile()
samples, samplingRate, nchannels, sampleWidth = readSound(path)
createSignalDiagram("Viso signalo diagrama", len(
    samples)/samplingRate, samples, "Laikas, s", "Amplitude")

fourier = numpy.fft.rfft(samples)

cropped = cropFourier(numpy.abs(fourier))
frequencies = numpy.fft.rfftfreq(len(cropped) * 2 - 1, 1 / samplingRate)
createFourierDiagram(frequencies, cropped)

X = numpy.fft.rfftfreq(len(fourier), 1 / samplingRate)
fourier = removeFrequencies(samplingRate, X, fourier)
fourier = addFrequencies(samplingRate, X, fourier)

createFourierDiagram(frequencies, cropFourier(numpy.abs(fourier)))

inverseFourier = numpy.fft.irfft(fourier)
createSignalDiagram("Signalas po atvirkstines DFT", len(
    inverseFourier)/samplingRate, inverseFourier, "Laikas, s", "Amplitude")
pathToWrite = writeFile()
writeSound(pathToWrite, inverseFourier, samplingRate, nchannels, sampleWidth)

input()
