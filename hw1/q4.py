import my_cross_val
from rand_proj import rand_proj
from quad_proj import quad_proj
from sklearn.datasets import load_digits
import numpy as np

digits=load_digits()
X,y=digits.data, digits.target
X_1=rand_proj(X,32)
# print "shape of X_1:"
# print np.shape(X_1)
X_2=quad_proj(X)
# print "shape of X_2:"
# print np.shape(X_2)
temp=quad_proj(X)
X_3=rand_proj(X_2,64)
# print "shape of X_3:"
# print np.shape(X_3)

print "LinearSVC with ~X1"
my_cross_val.my_cross_val("LinearSVC",X_1,y,10)

print "LinearSVC with ~X2"
my_cross_val.my_cross_val("LinearSVC",X_2,y,10)

print "LinearSVC with ~X3"
my_cross_val.my_cross_val("LinearSVC",X_3,y,10)

print "SVC with ~X1"
my_cross_val.my_cross_val("SVC",X_1,y,10)

print "SVC with ~X2"
my_cross_val.my_cross_val("SVC",X_2,y,10)

print "SVC with ~X3"
my_cross_val.my_cross_val("SVC",X_3,y,10)

print "LogisticRegression with ~X1"
my_cross_val.my_cross_val("LogisticRegression",X_1,y,10)

print "LogisticRegression with ~X2"
my_cross_val.my_cross_val("LogisticRegression",X_2,y,10)

print "LogisticRegression with ~X3"
my_cross_val.my_cross_val("LogisticRegression",X_3,y,10)
