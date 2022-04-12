import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from cnn_attention import preprocessing

from tensorflow.keras import layers
from tensorflow.keras import models


AUTOTUNE = tf.data.AUTOTUNE


def get_train_val_test_ds(train_files, val_files, test_files):
    """
    This function creates the tensorSlideDataset from the files to then extract the audio-label from them
    :param train_files:Tensor(str)
    :param val_files:Tensor(str)
    :param test_files:Tensor(str)
    :return: tf.data.Dataset of the train, validation and test dataset
    """
    train_ds = tf.data.Dataset.from_tensor_slices(train_files)
    train_ds = train_ds.shuffle(len(train_files))
    val_ds = tf.data.Dataset.from_tensor_slices(val_files)
    test_ds = tf.data.Dataset.from_tensor_slices(test_files)
    return train_ds, val_ds, test_ds


def get_input_shape(spectrogram_ds):
    """
    This function returns the Input shape: (124, 129, 1) of the train_spectrogram
    :param spectrogram_ds:ParallelMapDataset
    :return:TensorShape(141,129,1)
    """
    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape
    return input_shape


def norm(train_ds):
    """
    This function normalize each pixel in the image based on the mean and standard deviation of the pixels.
    :param train_ds:ParallelMapDataset
    :return:layers.Normalization
    """

    #Create the "tf.keras.layers.Normalization" layer.
    norm_layer = layers.Normalization()
    #Fit the state of the layer to the spectrogram's with "Normalization.adapt".
    norm_layer.adapt(data=train_ds.map(map_func=lambda spec, label: spec))
    return norm_layer

#Heavily modified from 'https://github.com/kobiso/CBAM-keras/blob/master/models/attention_module.py#L92'
class SpatialAttention_maxAvg(layers.Layer):
    """
    This function get the layer and add the max and avg attention to the axis=3 and calculate the score map of attention
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        self.max_pool = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))
        self.avg_pool = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))
        self.feature = layers.Conv2D(filters=1,
                                     kernel_size=self.kernel_size,
                                     strides=1,
                                     padding='same',
                                     activation='sigmoid',
                                     kernel_initializer='he_normal',
                                     use_bias=False)

    def call(self, input_feature):
        score_map = input_feature

        avg_pool = self.avg_pool(score_map)
        max_pool = self.max_pool(score_map)
        concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
        score_map = self.feature(concat)

        return layers.multiply([input_feature, score_map])


def create_model(num_labels, input_shape, norm_layer):
    """

    :param num_labels:
    :param input_shape:
    :param norm_layer:
    :return:
    """
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Down-sample the input.
        layers.Resizing(32, 32),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Conv2D(128, 3, activation='relu'),
        SpatialAttention_maxAvg(),
        # layers.GlobalAveragePooling2D(),
        layers.GlobalMaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])
    model.summary()
    return model


def compile_model(model):
    """
    This function Configure the model with the Adam optimizer and the cross-entropy loss:
    :param model:models.Sequential
    :return:
    """

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )


def fit_model(model, train_ds, val_ds, checkpoint_filepath):
    """
    This function fits the train_ds and valid_ds into our model with early stopping of patience=2
    :param model:models.Sequential
    :param train_ds:ParallelMapDataset
    :param val_ds:ParallelMapDataset
    :return:
    """

    #train the model for 100 epochs
    EPOCHS = 100
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[tf.keras.callbacks.ModelCheckpoint(save_best_only=True, save_weights_only=True,
                                                      filepath=checkpoint_filepath)],
    )


def predict_model(model, test_ds):
    """
    This function calculate the predicts of the test_ds in our model, in our test part we do not batch the test dataset,
    we make an array of the audios and labels and fit into the model
    :param model:models.Sequential
    :param test_ds:ParallelMapDataset
    :return:
    """
    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')
    return test_acc


def train_model():
    # Set the seed value for experiment reproducibility.
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    checkpoint_filepath = "checkpoint.pth"

    batch_size = 64

    #retreive the data
    data = preprocessing.get_data()

    digits = data['digit'].unique()
    digits.sort()

    num_labels = len(digits)
    max_len = data['len'].max()

    train_files, val_files, test_files = preprocessing.get_random_shuffle_files(data['dir'].to_list())

    train_ds, val_ds, test_ds = get_train_val_test_ds(train_files, val_files, test_files)

    train_spectrogram = preprocessing.preprocess_dataset(train_ds, digits, max_len)
    val_spectrogram = preprocessing.preprocess_dataset(val_ds, digits, max_len)
    test_spectrogram = preprocessing.preprocess_dataset(test_ds, digits, max_len)

    input_shape = get_input_shape(train_spectrogram)
    norm_layer = norm(train_spectrogram)

    train_spectrogram = train_spectrogram.batch(batch_size)
    train_spectrogram = train_spectrogram.cache().prefetch(AUTOTUNE)

    val_spectrogram = val_spectrogram.batch(batch_size)
    val_spectrogram = val_spectrogram.cache().prefetch(AUTOTUNE)

    model = create_model(num_labels, input_shape, norm_layer)
    compile_model(model)
    fit_model(model, train_spectrogram, val_spectrogram, checkpoint_filepath)

    model.load_weights(checkpoint_filepath)

    test_acc = predict_model(model, test_spectrogram)
    return test_acc






