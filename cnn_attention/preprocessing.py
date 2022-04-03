import os
import numpy as np
import tensorflow as tf
import librosa.display
import pandas as pd
from tqdm import tqdm

dataset_path = 'data/recordings/'
AUTOTUNE = tf.data.AUTOTUNE

def get_filenames():
    # extracting recordings
    filenames = tf.io.gfile.glob(dataset_path + '*')
    filenames = tf.random.shuffle(filenames)
    return filenames


def get_data():
    data = pd.DataFrame(columns=['raw_data', 'len', 'duration', 'digit', 'sample_rate', 'dir', 'shape'])
    for i in tqdm(os.listdir(dataset_path)):
        raw_data, frame_rate = librosa.load(dataset_path + i, sr=None, mono=False)
        duration = librosa.get_duration(y=raw_data, sr=frame_rate)
        data.loc[len(data.index)] = [raw_data, len(raw_data), duration, i.split('_')[0], frame_rate, i, raw_data.shape]
    return data


def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_digit(file_path):
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    return tf.strings.split(input=parts[-1], sep='_')[0]


def get_waveform_and_digit(file_path, digits):
    digit = get_digit(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    digit_id = tf.argmax(digit == digits)
    return waveform, digit_id


def get_spectrogram(waveform):
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)

    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def get_spectrogram_and_digit_id(audio, digit_id):
    spectrogram = get_spectrogram(audio)
    return spectrogram, digit_id


def padded(waveform, digit_id, max_len):
    # zero-padding for audios to have same length
    waveform = waveform[:max_len]
    zero_padding = tf.zeros(
        [max_len] - tf.shape(waveform), dtype=tf.float32)

    waveform = tf.cast(waveform, dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)
    return waveform, digit_id


def preprocess_dataset(files_ds, digits, max_len):
    output_ds = files_ds.map(
        map_func=lambda f: get_waveform_and_digit(f, digits),
        num_parallel_calls=AUTOTUNE)

    output_ds = output_ds.map(
                 map_func=lambda waveform, digit_id: padded(waveform, digit_id, max_len),
                 num_parallel_calls=AUTOTUNE)

    output_ds = output_ds.map(
        map_func=get_spectrogram_and_digit_id,
        num_parallel_calls=AUTOTUNE)

    return output_ds