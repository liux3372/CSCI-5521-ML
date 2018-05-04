import my_cross_val
from sklearn.datasets import load_digits
import numpy as np




def rand_proj(X,d):
    row_num=np.shape(X)[0]
    col_num=np.shape(X)[1]
    G=np.random.randn(col_num,d)
    #X_new=(np.matrix(X))*(np.matrix(G)),same as below
    X_new=np.dot(X,G)
    return X_new

# digits=load_digits()
# X,y=digits.data, digits.target
# X_1=rand_proj(X,32)
#
#
# print "LinearSVC with Digits"
# main.my_cross_val("LinearSVC",X_1,y,10)
#
# print "SVC with Digits"
# main.my_cross_val("SVC",X_1,y,10)
#
# print "LogisticRegression with Digits"
# main.my_cross_val("LogisticRegression",X_1,y,10)
