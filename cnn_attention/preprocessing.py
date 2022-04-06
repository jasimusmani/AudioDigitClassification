import os
import numpy as np
import tensorflow as tf
import librosa.display
import pandas as pd
from tqdm import tqdm

dataset_path = 'data/recordings/'
AUTOTUNE = tf.data.AUTOTUNE

def get_random_shuffle_files():
    """This function extract recording files and Randomly shuffles them along its first dimension.
        then split the data into train/val/test parts by 90-5-5 percents respectively."""

    # url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/ionosphere.csv'
    # dataframe = read_csv(url, header=None)

    # extracting recordings
    filenames = tf.io.gfile.glob(dataset_path + '*')
    filenames = tf.random.shuffle(filenames)

    train_files = filenames[:2700]
    val_files = filenames[2700:2850]
    test_files = filenames[-150:]

    return train_files, val_files, test_files


def get_data():
    """This function put the information of the recording wav into a tabular data structure.
       and librosa load an audio file as a floating point time series."""
    data = pd.DataFrame(columns=['raw_data', 'len', 'duration', 'digit', 'sample_rate', 'dir', 'shape'])
    for i in tqdm(os.listdir(dataset_path)):
        raw_data, frame_rate = librosa.load(dataset_path + i, sr=None, mono=False)
        duration = librosa.get_duration(y=raw_data, sr=frame_rate)
        data.loc[len(data.index)] = [raw_data, len(raw_data), duration, i.split('_')[0], frame_rate, i, raw_data.shape]
    return data


def decode_audio(audio_binary):
    """This function preprocesses the dataset's raw WAV audio files into audio tensors.
    :param audio_binary:torch.Tensor(str)
    :return:audio:torch.Tensor(float32)
    """
    audio, _ = tf.audio.decode_wav(contents=audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_digit(file_path):
    """This function extract the difit number from the file_path
    :param file_path:torch.Tensor(str)
    :return torch.Tensor(str)
    """
    parts = tf.strings.split(
        input=file_path,
        sep=os.path.sep)
    return tf.strings.split(input=parts[-1], sep='_')[0]


def get_waveform_and_digit(file_path, digits):
    """
    This function gets the filepath and the list of the digits(0-9) and return the waveform and the digit_id
    :param file_path:torch.Tensor(str)
    :param digits:numpy.ndarray
    :return:
    """
    digit = get_digit(file_path)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    digit_id = tf.argmax(digit == digits)
    return waveform, digit_id


def get_spectrogram(waveform):
    """
    This function get the waveform and convert it to a spectrogram
    The STFT produces an array of complex numbers representing magnitude and phase.
    :param waveform:torch.Tensor(float32)
    :return: spectrogram
    """
    # Convert the waveform to a spectrogram via a STFT.
    spectrogram = tf.signal.stft(waveform, frame_length=255, frame_step=128)

    # Obtain the magnitude of the STFT.
    spectrogram = tf.abs(spectrogram)
    # Add a `channels` dimension, so that the spectrogram can be used
    # as image-like input data with convolution layers (which expect
    # shape (`batch_size`, `height`, `width`, `channels`).
    spectrogram = spectrogram[..., tf.newaxis]
    return spectrogram


def get_spectrogram_and_digit_id(audio, digit_id):
    """
    This function gets the audio and digit_id and return the spectrogram of the audio and the digit_id
    :param audio:torch.Tensor(float32)
    :param digit_id:torch.Tensor(int64)
    :return:
    """
    spectrogram = get_spectrogram(audio)
    return spectrogram, digit_id


def padded(waveform, digit_id, max_len):
    """
     The waveforms need to be of the same length, so that when you convert them to spectrograms,
     the results have similar dimensions.
     This can be done by simply zero-padding the audio clips that are shorter to be the same size of the longest audio.
     return the padded waveform and the digit_id of the wave
    :param waveform:torch.Tensor(float32)
    :param digit_id: torch.Tensor(int64)
    :param max_len: numpy.int64
    :return: (waveform, digit_id)
    """
    waveform = waveform[:max_len]
    zero_padding = tf.zeros(
        [max_len] - tf.shape(waveform), dtype=tf.float32)

    waveform = tf.cast(waveform, dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)
    return waveform, digit_id


def preprocess_dataset(files_ds, digits, max_len):
    """
    This function gets the file dataset and the max_len of the data and the digits and
    return the (spectrogram, digit_id) of the waveform from the files_ds

    :param files_ds:TensorSliceDataset
    :param digits: numpy.ndarray
    :param max_len: numpy.int64
            maximum length of the waves
    :return: (spectrogram, digit_id)
    """

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