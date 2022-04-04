from __future__ import division, print_function
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import keras
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
import os
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import f1_score
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import activations
"""
The pre-processing part is adapted from an online source
"""
def wav_to_spectrogram(audio_path, save_path, spectrogram_dimensions=(64, 64), noverlap=16, cmap='gray_r'):
    """ Creates a spectrogram of a wav file.
    :param audio_path: path of wav file
    :param save_path:  path of spectrogram to save
    :param spectrogram_dimensions: number of pixels the spectrogram should be. Defaults (64,64)
    :param noverlap: See http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    :param cmap: the color scheme to use for the spectrogram. Defaults to 'gray_r'
    :return:
    """

    sample_rate, samples = wav.read(audio_path)

    fig = plt.figure()
    fig.set_size_inches((spectrogram_dimensions[0]/fig.get_dpi(), spectrogram_dimensions[1]/fig.get_dpi()))
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.specgram(samples, cmap=cmap, Fs=2, noverlap=noverlap)
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)

def dir_to_spectrogram(audio_dir, spectrogram_dir, spectrogram_dimensions=(64, 64), noverlap=16, cmap='gray_r'):
    """ Creates spectrograms of all the audio files in a dir
    :param audio_dir: path of directory with audio files
    :param spectrogram_dir: path to save spectrograms
    :param spectrogram_dimensions: tuple specifying the dimensions in pixes of the created spectrogram. default:(64,64)
    :param noverlap: See http://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.spectrogram.html
    :param cmap: the color scheme to use for the spectrogram. Defaults to 'gray_r'
    :return:
    """
    file_names = [f for f in listdir(audio_dir) if isfile(join(audio_dir, f)) and '.wav' in f]

    for file_name in file_names:
        print(file_name)
        audio_path = audio_dir + file_name
        spectogram_path = spectrogram_dir + file_name.replace('.wav', '.png')
        wav_to_spectrogram(audio_path, spectogram_path, spectrogram_dimensions=spectrogram_dimensions, noverlap=noverlap, cmap=cmap)

    if __name__ == '__main__':
        audio_dir = "/Users/macbookpro/Desktop/MLPROJECT/recordings/"
        spectrogram_dir = "/Users/macbookpro/Desktop/MLPROJECT/spectograms/"
        dir_to_spectrogram(audio_dir, spectrogram_dir)

# Load all images files, convert to numpy array with related labels in a list format
imagesDir = "/Users/macbookpro/Desktop/MLPROJECT/spectograms/"
trainset = []
testset = []
for file in os.listdir(imagesDir):
    if (file=='.DS_Store'):
        continue
    label = file.split('_')[0]
    sample_number = file.split('_')[2]

    img = image.load_img(imagesDir+file)
    if sample_number in ['0.png','1.png','2.png','3.png','4.png']:
        testset.append([image.img_to_array(img), label])
    else:
        trainset.append([image.img_to_array(img), label])