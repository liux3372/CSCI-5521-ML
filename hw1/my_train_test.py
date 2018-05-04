from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np
from astropy.table import Table
from random import uniform
from sklearn.model_selection import train_test_split
from my_cross_val import compare



def my_train_test(method,X,y,pi,k):
    #select the method and initalize related estimator
    if method=="LinearSVC":
        estimator=LinearSVC()

    elif method=="SVC":
        estimator=SVC()

    elif method=="LogisticRegression":
        estimator=LogisticRegression()

    else:
        print "method should be LinearSVC,SVC or LogisticRegression."

    scores_list=list()
    row_num=int(np.shape(X)[0])#num of obsevations
    col_num=int(np.shape(X)[1])#num of features
    for x in range(k):
        X=np.random.permutation(X)
        y=np.random.permutation(y)
        end=row_num*pi
        X_train=X[0:end,:]
        X_test=X[end+1:-1,:]
        y_train=y[0:end]
        y_test=y[end+1:-1]
        # print "x:"
        # print x
        # print "X_train shape:"
        # print np.shape(X_train)
        # print "y_train shape:"
        # print np.shape(y_train)
        # print "X_test shape:"
        # print np.shape(X_test)
        # print "y_test shape:"
        # print np.shape(y_test)
        #X_train,X_test,y_train,y_test=train_test_split(X,y,train_size=pi)
        estimator.fit(X_train, y_train)
        y_pred=estimator.predict(X_test)
        #print "y_pred"
        scores_list.append(compare(y_pred,y_test))
        #scores_list.append(estimator.fit(X_train, y_train).score(X_test, y_test))
    #changing scores to error rates
    scores=np.array(scores_list)
    error_rates=1-scores
    mean=np.mean(error_rates)
    # print "mean:\n"
    # print mean
    std=np.std(error_rates)
    # print "std:\n"
    # print std
    to_add=np.array([mean,std])
    result=np.append(error_rates,to_add)
    t=Table(result)
    t.pprint(max_width=-1)
