#accuracy result(took about 15 min with #line 69 ; n_folds=20)
#linear: 0.858361
#poly: 0.841813
#accuracy result(took about 15 min with #line 69 ; n_folds=50)
#linear: 0.853961
#poly: 0.834515

#author Ying Zhang & Yujia Qiu
# CS 453X
# Professor Jacob Whitehill
# Hw4-part2

import sklearn.svm
import sklearn.metrics
from sklearn.model_selection import train_test_split
import numpy as np
import pandas
from tqdm import tqdm
# Load data
d = pandas.read_csv('train.csv')
y = np.array(d.target)  # Labels
X = np.array(d.iloc[:,2:])  # Features

print(X[:5])
print(y[:5])
# Split into train/test folds
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=6)



# Linear SVM
def pred_yhat_linear_svc(X_train, X_test, y_train):
    print('\n')
    print(X_train.shape)
    print(y_train.shape)
    # Linear kernel
    linear_svc = sklearn.svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=True, tol=1e-4,
                                       C=1e15, multi_class='ovr', fit_intercept=True,
                                       intercept_scaling=1, class_weight=None, verbose=0,
                                       random_state=None, max_iter=1000)

    linear_svc.fit(X_train, y_train)

    yhat_linear = linear_svc.decision_function(X_test)

    return yhat_linear


# Non-linear SVM (polynomial kernel)
def pred_yhat_poly_svc(X_train, X_test, y_train):
    # Non-linear kernel
    print(X_train[:5])
    print(y_train[:5])
    poly_svc = sklearn.svm.SVC(C=1.0, kernel='poly', degree=3, gamma='auto',
                                     coef0=0.0, shrinking=True, probability=False,
                                     tol=1e-3, cache_size=200, class_weight=None,
                                     verbose=False, max_iter=-1, decision_function_shape='ovr',
                                     random_state=None)

    poly_svc.fit(X_train, y_train)


    yhat_poly = poly_svc.decision_function(X_test)

    return yhat_poly


## bagging: Apply the SVMs to the test set
def bagging(x_train,y_train,x_test,n_folds=50):
    ## Split data for bagging
    original_index = np.arange(len(x_train))
    np.random.shuffle(original_index)
    bag_list = np.split(original_index, n_folds)

    # create arrays for svm predictions
    pred_linear = np.empty((0, x_test.shape[0]))
    pred_poly = np.empty((0, x_test.shape[0]))

    # iterate for all bags and call for svms to predict results
    for i in tqdm(bag_list):

        linear = pred_yhat_linear_svc(x_train[i], x_test, y_train[i])
        poly = pred_yhat_poly_svc(x_train[i], x_test, y_train[i])

        pred_linear = np.append(pred_linear, [linear], axis=0)
        pred_poly = np.append(pred_poly, [poly], axis=0)

    # bagging by the average of predictions of all svc models
    yhat_linear = np.mean(pred_linear, axis=0)
    yhat_poly = np.mean(pred_poly, axis=0)

    return yhat_linear, yhat_poly


yhat_linear, yhat_poly = bagging(X_train,y_train,X_test)

# Compute accuracy
auc_linear = sklearn.metrics.roc_auc_score(y_test, yhat_linear)
auc_poly = sklearn.metrics.roc_auc_score(y_test, yhat_poly)

print("auc_linear: %f\n" % auc_linear)
print("auc_poly: %f\n" % auc_poly)

# Split into train/test folds
# Split a dataset into 20 folds
# source cited from https://machinelearningmastery.com/implement-bagging-scratch-python/
#def cross_validation_split(train_x, test_x, train_y, n_folds = 20):
    #dataset_split = list()
    #dataset_copy = list(dataset)
    # fold_size = int(len(dataset) / n_folds)
    #   rl= len(train_x)
    #  r2 = np.arange(r1)
    #np.random.shuffle(r2)
    #fold_size = bp.split(r1,n_folds)
    #for i in range(fold_size):
    #   n1,n2 = init(train_x[i],test_x,train_y[i])
    #   all_n1 = np.append(all_n1,[n1],axis=0)
    #   all_n2 = np.append(all_n2,[n2],axis=0)
    #avg_n1 = np.mean(all_n1,axis=0)
    #avg_n1 = np.mean(all_n2,axis=0)
    #fold = list()
#while len(fold) < fold_size:
#     index = randrange(len(dataset_copy))
#     fold.append(dataset_copy.pop(index))
#dataset_split.append(fold)
#return avg_n1,avg_n2
