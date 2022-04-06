#MLP

from sklearn.neural_network import MLPClassifier
MLPClassifier = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100,),
                                     activation='relu', solver='adam',batch_size='auto',
                                     learning_rate='constant', learning_rate_init=0.001)
MLP = MLPClassifier.fit(X_trainsvm, ytrainsvm)
print(MLP.score(X_testsvm, y_test_))

