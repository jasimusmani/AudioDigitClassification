import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from cnn_attention import preprocessing

from tensorflow.keras import layers
from tensorflow.keras import models


AUTOTUNE = tf.data.AUTOTUNE


def get_train_val_test_ds(train_files, val_files, test_files):
    train_ds = tf.data.Dataset.from_tensor_slices(train_files)
    val_ds = tf.data.Dataset.from_tensor_slices(val_files)
    test_ds = tf.data.Dataset.from_tensor_slices(test_files)

    return train_ds, val_ds, test_ds


def get_input_shape(spectrogram_ds):
    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape
    return input_shape


def norm(train_ds):
    # Instantiate the `tf.keras.layers.Normalization` layer.
    norm_layer = layers.Normalization()
    # Fit the state of the layer to the spectrograms
    # with `Normalization.adapt`.
    norm_layer.adapt(data=train_ds.map(map_func=lambda spec, label: spec))
    return norm_layer


class SpatialAttention_maxAvg(layers.Layer):
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
        assert avg_pool.get_shape()[-1] == 1
        max_pool = self.max_pool(score_map)
        assert max_pool.get_shape()[-1] == 1
        concat = layers.Concatenate(axis=3)([avg_pool, max_pool])
        assert concat.get_shape()[-1] == 2
        score_map = self.feature(concat)
        assert score_map.get_shape()[-1] == 1

        return layers.multiply([input_feature, score_map])


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, mode='mean'):
        super().__init__()
        self.kernel_size = kernel_size
        if mode == 'mean':
            func = lambda x: K.mean(x, axis=3, keepdims=True)
        else:
            func = lambda x: K.max(x, axis=3, keepdims=True)

        self.pool = layers.Lambda(func)
        self.feature = layers.Conv2D(filters=1,
                                     kernel_size=self.kernel_size,
                                     strides=1,
                                     padding='same',
                                     activation='sigmoid',
                                     kernel_initializer='he_normal',
                                     use_bias=False)

    def call(self, input_feature):

        score_map = input_feature

        pool = self.pool(score_map)
        assert pool.get_shape()[-1] == 1

        score_map = self.feature(pool)
        assert score_map.get_shape()[-1] == 1

        return layers.multiply([input_feature, score_map])


def create_model(num_labels, input_shape, norm_layer):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # Downsample the input.
        layers.Resizing(64, 64),
        # Normalize.
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation='relu'),
        layers.Conv2D(128, 3, activation='relu'),
        # SpatialAttention_maxAvg(),
        #     SpatialAttention(mode='max'),
        #     layers.GlobalAveragePooling2D(),
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
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )


def fit_model(model, train_ds, val_ds):
    EPOCHS = 30
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )


def predict_model(model, test_ds):
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
    batch_size = 64

    filenames = preprocessing.get_filenames()
    data = preprocessing.get_data()

    digits = data['digit'].unique()
    digits.sort()

    num_labels = len(digits)
    max_len = data['len'].max()

    train_files = filenames[:2700]
    val_files = filenames[2700:2850]
    test_files = filenames[-150:]

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
    fit_model(model, train_spectrogram, val_spectrogram)
    test_acc = predict_model(model, test_spectrogram)
    print("test accuracy:", test_acc)






