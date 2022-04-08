from cnn_attention import preprocessing, model
import numpy as np
import tensorflow as tf
import pytest

class DATASET:
    def __init__(self, train_data, val_data, test_data, digits):
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.digits = digits

@pytest.fixture
def dataset():
    data = preprocessing.get_data(size=100)
    train_data, valid_data, test_data = preprocessing.get_random_shuffle_files(data['dir'].to_list(),
                                            train_sz=40, val_sz=30, test_sz=10)

    digits = data['digit'].unique()
    digits.sort()

    return DATASET(train_data, valid_data, test_data, digits)


def test_get_train_val_test_ds(dataset):
    train, val, test = model.get_train_val_test_ds(dataset.train_data,
                                dataset.val_data, dataset.test_data)

    assert len(train) == len(dataset.train_data)
    assert len(val) == len(dataset.val_data)
    assert len(test) == len(dataset.test_data)


def test_get_digit(dataset):
    digit = preprocessing.get_digit(dataset.train_data[0])
    digit_id = tf.argmax(digit == dataset.digits)

    assert int(digit_id.numpy()) == int(str(dataset.train_data[0].numpy()).split('/')[-1].split('_')[0]), 'wrong digit'


def test_get_waveform_and_digit(dataset):
    waveform, digit_id = preprocessing.get_waveform_and_digit(dataset.train_data[0], dataset.digits)
    assert int(digit_id.numpy()) == int(str(dataset.train_data[0].numpy()).split('/')[-1].split('_')[0]), 'wrong digit'
    tf.debugging.assert_type(waveform, tf_type= tf.float32)


def test_spectrogram(dataset):
    waveform, digit_id = preprocessing.get_waveform_and_digit(dataset.train_data[0], dataset.digits)
    spectrogram = preprocessing.get_spectrogram(waveform)
    shape = spectrogram.get_shape().as_list()

    assert len(shape) == 3, 'wrong shape of the spectrogram'


def test_padded(dataset):
    waveform, digit_id = preprocessing.get_waveform_and_digit(dataset.train_data[0], dataset.digits)

    max_len = 100000
    waveform_padded, _ = preprocessing.padded(waveform, digit_id, max_len)

    assert len(waveform_padded[:len(waveform)]) == len(waveform), 'wrong padding'
    assert waveform_padded[len(waveform):].numpy().all() == 0, 'wrong value for the padded area'

