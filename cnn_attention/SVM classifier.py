import sklearn
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


X_trainsvm = X_train.reshape(2700, 12288)
X_testsvm = X_test.reshape(300, 12288)
svm = sklearn.svm.SVC(gamma='auto',C=20.0)
svm.fit(X_trainsvm,ytrainsvm)
acc = svm.score(X_trainsvm, ytrainsvm)
print("acc=%0.3f" % acc)


#Grid search the  hyperparameters for SVM
# parameters = {'kernel': ['rbf'], 'gamma': [0.001, 0.0001, 0.00001],
#                    'C': [5, 7 ,15,20,30,35,40]}
# gridcv = sklearn.model_selection.GridSearchCV(SVC(), parameters, verbose = 1, cv=5)
# fittedsvm =gridcv.fit(X_trainsvm, ytrainsvm)
# print(fittedsvm.best_params_)
# print(fittedsvm.best_score_)