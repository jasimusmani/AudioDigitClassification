import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
import librosa,librosa.display
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras import models
import tqdm

import tensorflow.keras.backend as K

# Set the seed value for experiment reproducibility.
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

dataset_path = '../data/recordings/'

#extracting recordings
filenames = tf.io.gfile.glob(dataset_path + '*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)

#load data
data = pd.DataFrame(columns=['raw_data','len','duration','digit','sample_rate','dir','shape'])
for i in tqdm(os.listdir(dataset_path)):
    raw_data, frame_rate = librosa.load(dataset_path+i, sr=None, mono=False)
    duration = librosa.get_duration(y=raw_data,sr=frame_rate)
    data.loc[len(data.index)] = [raw_data,len(raw_data),duration,i.split('_')[0],frame_rate,i,raw_data.shape]

#extracting digitLabels
digits = data['digit'].unique()
digits.sort()
print("Digit labels: ", digits)

#extracting data infos
data.sort_values(by='len')
max_len_data = data['len'].max()
print("max_len_data: ", max_len_data)

print('Number of total examples:', num_samples)

avg_len_data = data['len'].mean()
print("avg_len_data: ", avg_len_data)
min_len_data = data['len'].min()
print("min_len_data: ", min_len_data)

#seperating train and valid and test
train_files = filenames[:2700]
val_files = filenames[2700:2850]
test_files = filenames[-150:]

print('Training set size: ', len(train_files))
print('Validation set size: ', len(val_files))
print('Test set size: ', len(test_files))

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_digit(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    return tf.strings.split(input=parts[-1], sep='_')[0]

def get_waveform_and_digit(file_path):
    digit = get_digit(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, digit

AUTOTUNE = tf.data.AUTOTUNE

files_ds = tf.data.Dataset.from_tensor_slices(train_files)
waveform_ds = files_ds.map(
    map_func=get_waveform_and_digit,
    num_parallel_calls=AUTOTUNE)

rows = 3
cols = 3
n = rows * cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 12))

for i, (audio, label) in enumerate(waveform_ds.take(n)):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    ax.plot(audio.numpy())
    ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
    ax.set_xticks(np.arange(0, 5000, 1000))
    label = label.numpy().decode('utf-8')
    ax.set_title(label)

plt.show()


def get_spectrogram(waveform):
    # zero-padding for audios to have same lenght
    waveform = waveform[:max_len_data]
    zero_padding = tf.zeros(
        [max_len_data] - tf.shape(waveform), dtype=tf.float32)

    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)

    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)

    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram

for waveform, label in waveform_ds.take(1):
    label = label.numpy().decode('utf-8')
    spectrogram = get_spectrogram(waveform)

print('Label:', label)
print('Waveform shape:', waveform.shape)
print('Spectrogram shape:', spectrogram.shape)

def plot_spectrogram(spectrogram, ax):
    if len(spectrogram.shape) > 2:
        assert len(spectrogram.shape) == 3
        spectrogram = np.squeeze(spectrogram, axis=-1)
    log_spec = np.log(spectrogram.T + np.finfo(float).eps)
    height = log_spec.shape[0]
    width = log_spec.shape[1]
    X = np.linspace(0, np.size(spectrogram), num=width, dtype=int)
    Y = range(height)
    ax.pcolormesh(X, Y, log_spec)

fig, axes = plt.subplots(2, figsize=(12, 8))
timescale = np.arange(waveform.shape[0])
axes[0].plot(timescale, waveform.numpy())
axes[0].set_title('Waveform')

plot_spectrogram(spectrogram.numpy(), axes[1])
axes[1].set_title('Spectrogram')
plt.show()

def get_spectrogram_and_digit_id(audio, digit):
    spectrogram = get_spectrogram(audio)
    digit_id = tf.argmax(digit == digits)
    return spectrogram, digit_id

spectrogram_ds = waveform_ds.map(
    map_func=get_spectrogram_and_digit_id,
    num_parallel_calls=AUTOTUNE)

rows = 3
cols = 3
n = rows*cols
fig, axes = plt.subplots(rows, cols, figsize=(10, 10))

for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
    r = i // cols
    c = i % cols
    ax = axes[r][c]
    plot_spectrogram(spectrogram.numpy(), ax)
    ax.set_title(digits[label_id.numpy()])
    ax.axis('off')

plt.show()