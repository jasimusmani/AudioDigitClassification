import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix


X_trainsvm = X_train.reshape(2700, 12288)
X_testsvm = X_test.reshape(300, 12288)


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
    values = tuned_value.best_estimator_.fit(X, y)
    return values, tuned_value.best_score_
# To find the best parameters using Randomized Search
# SVM(X_trainsvm, ytrainsvm, cv = 3)

#The values found from randomized search are C =15 and gamma = 0.001
svmmodel = sklearn.svm.SVC(C=15, kernel='rbf', gamma= 0.001)
svmmodel.fit(X_trainsvm, ytrainsvm)

y_prediction = svmmodel.predict(X_testsvm)
confusion_matrix(y_test_, y_prediction)

print("Accuracy is ",sklearn.metrics.accuracy_score(y_test_,svmmodel.predict(X_testsvm))*100,"%")
plot_confusion_matrix(svmmodel, X_testsvm, y_test_)
plt.show()
