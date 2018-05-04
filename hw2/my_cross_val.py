from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np
from astropy.table import Table
from MultiGaussClassify import MultiGaussClassify

def compare(y1,y2):
    correct=0
    if y1.size!=y2.size:
        print ("two arrays with different sizes!")
        return 0
    else:
        for i in range(y1.size):
            if(y1[i]==y2[i]):
                correct=correct+1
        return correct/float(y1.size)

def my_cross_val(method,X,y,k):
    #select the method and initalize related estimator
    if method=="LinearSVC":
        estimator=LinearSVC()
        # lsvc=LinearSVC()
        # scores=cross_val_score(lsvc,X,y,cv=k)
    elif method=="SVC":
        estimator=SVC()
        #svc=SVC()
        #scores=cross_val_score(svc,X,y,cv=k)
    elif method=="LogisticRegression":
        estimator=LogisticRegression()
        #lr=LogisticRegression()
        #scores=cross_val_score(lr,X,y,cv=k)
    elif method=="MultiGaussClassify":
        estimator=MultiGaussClassify()
    else:
        print ("Undefined method!")
    row_num=np.shape(X)[0]#num of obsevations
    col_num=np.shape(X)[1]#num of features
    size=row_num/k#floor division
    scores_list=list()
    # X_mask=np.ones((row_num,col_num),dtype=bool)
    # y_mask=np.ones(row_num,dtype=bool)
    for i in range (k):
        start=i*size#i*100
        end=(size+i*size)#100+i*100
        X_test=X[int(start):int(end),:]
        #X_mask[start:end,:]=False
        #X_train=np.multiply(X,X_mask)#leave zero row since they don't effect the training process
        # X_temp=np.multiply(X,X_mask)
        # X_train=X[np.nonzero(X_temp)]
        if i==0:
            X_train=X[int(end+1):-1,:]
            y_train=y[int(end+1):-1]
        elif i==k-1:
            X_train=X[0:int(end-1),:]
            y_train=y[0:int(end-1)]
        else:
            X_train=np.concatenate((X[0:int(start-1),:],X[int(end+1):-1,:]),axis=0)
            y_train=np.concatenate((y[0:int(start-1)],y[int(end+1):-1]),axis=0)


        y_test=y[int(start):int(end)]

        # print "i:"
        # print i
        # print "X_train shape:"
        # print np.shape(X_train)
        # print "y_train shape:"
        # print np.shape(y_train)
        # print "X_test shape:"
        # print np.shape(X_test)
        # print "y_test shape:"
        # print np.shape(y_test)
        estimator.fit(X_train, y_train)
        y_pred=estimator.predict(X_test)
        #self defined score
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
