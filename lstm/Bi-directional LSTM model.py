data_shape1 = (X_train.shape[1], X_train.shape[2])


# Bi-directional LSTM model using Sci-kit Learn package
#The code is PEP8 Compliant
def lstm():
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


model = KerasClassifier(build_fn=lstm, epochs=100, batch_size=50)
model.fit(X_train, y_train, verbose=0)
y_pred = model.predict(X_test)
# convert one hot label to the respective classes
y_testclass = np.argmax(y_test, axis=1)
print(accuracy_score(y_pred, y_testclass))
