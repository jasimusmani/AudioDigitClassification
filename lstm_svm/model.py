import keras
import os
import numpy as np
import tensorflow as tf
from functools import partial
from keras.preprocessing import image
from keras.utils.np_utils import to_categorical
from tensorflow.keras.layers import Input, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras import layers
from tensorflow.keras import activations
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from lstm_svm import preprocessing

import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix


# Bi-directional LSTM model using Sci-kit Learn package
#The code is PEP8 Compliant
def lstm(data_shape1):
    model = keras.Sequential()
    model.add(Bidirectional(
        LSTM(50, input_shape=data_shape1, return_sequences=False)))
    model.add(Dropout(0.3))
    model.add(Dense(10))
    model.add(layers.Activation(activations.softmax))

    ad = tf.optimizers.Adam(learning_rate=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=ad, metrics=['accuracy'])

    return model

def SVM(X, y, cv=3):
    """This function is the implementation of the SVM for classification.
    Arguments:
    X is the preprocessed array generated from the spectrogram of the audio sample
    y is the label of the audio sample
    cv is the size of cross validation

    return:
    This function returns the SVC object and the accuracy of the model

    In order to identify the best parameters, this function using randomized search
    to find the appropriate parameters

    """

    parameters = {'kernel': ['rbf'], 'gamma': [0.001, 0.0001, 0.00001],
                  'C': [5, 7, 15, 20, 30, 35, 40]}

    tuned_value = RandomizedSearchCV(SVC(), parameters, cv=cv, n_iter=9)
    tuned_value.fit(X, y)

    return tuned_value

def pre(audio_dir):
    imagesDir = "data/spectograms/"

    if not os.path.exists(imagesDir):
        os.mkdir(imagesDir)
        preprocessing.convert_audio_to_spectrogram(audio_dir, imagesDir)

    # Load all images files, convert to numpy array with related labels in a list format
    trainset = []
    testset = []
    for file in os.listdir(imagesDir):
        if (file == '.DS_Store'):
            continue
        label = file.split('_')[0]
        sample_number = file.split('_')[2]

        img = image.load_img(imagesDir + file)
        if sample_number in ['0.png', '1.png', '2.png', '3.png', '4.png']:
            testset.append([image.img_to_array(img), label])
        else:
            trainset.append([image.img_to_array(img), label])

    # Get only images in the train list not the Labels
    X_train = np.asanyarray( [item[0] for item in trainset] )
    # Get only Labels in the train list not the images
    y_train = np.asanyarray( [item[1] for item in trainset] )
    # Get only images in the test list not the Labels
    X_test = np.asanyarray( [item[0] for item in testset] )
    # Get only Labels in the test list not the images
    y_test = np.asanyarray( [item[1] for item in testset] )

    # converting y data into categorical (one-hot encoding)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    X_train /= 255
    X_test /= 255

    return X_train, y_train, X_test, y_test

def train_lstm_model(audio_dir):
    X_train, y_train, X_test, y_test = pre(audio_dir)

    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], X_train.shape[2] * X_train.shape[3]))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], X_test.shape[2] * X_test.shape[3]))

    data_shape1 = (X_train.shape[1], X_train.shape[2])

    lstm_model = partial(lstm, data_shape1)

    model = KerasClassifier(build_fn=lstm_model, epochs=100, batch_size=50)
    model.fit(X_train, y_train, verbose=0)
    y_pred = model.predict(X_test)
    # convert one hot label to the respective classes
    y_testclass = np.argmax(y_test, axis=1)
    return accuracy_score(y_pred, y_testclass) * 100.

def train_svm_model(audio_dir):
    X_train, y_train, X_test, y_test = pre(audio_dir)

    X_trainsvm = X_train.reshape(2700, -1)
    X_testsvm = X_test.reshape(300, -1)

    ytrainsvm = np.argmax(y_train, axis=1)
    y_test_ = np.argmax(y_test, axis=1)

    svmmodel = SVM(X_trainsvm, ytrainsvm)
    svmmodel.fit(X_trainsvm, ytrainsvm)

    y_prediction = svmmodel.predict(X_testsvm)
    confusion_matrix(y_test_, y_prediction)

    return sklearn.metrics.accuracy_score(y_test_, svmmodel.predict(X_testsvm)) * 100