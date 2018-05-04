import my_train_test
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
import numpy as np

boston50=load_boston()
X_1, y_1=boston50.data, boston50.target
r50=np.median(y_1)
y_1[y_1>=r50]=1
y_1[y_1!=1]=0
#print y_1
print "LinearSVC with Boston50"
my_train_test.my_train_test("LinearSVC",X_1,y_1,0.75,10)

boston75=load_boston()
X_2, y_2=boston75.data, boston75.target
r75=np.percentile(y_2,75)
y_2[y_2>=r75]=1
y_2[y_2!=1]=0
print "LinearSVC with Boston75"
my_train_test.my_train_test("LinearSVC",X_2,y_2,0.75,10)

digits=load_digits()
X_3,y_3=digits.data, digits.target
print "LinearSVC with Digits"
my_train_test.my_train_test("LinearSVC",X_3,y_3,0.75,10)

print "SVC with Boston50"
my_train_test.my_train_test("SVC",X_1,y_1,0.75,10)

print "SVC with Boston75"
my_train_test.my_train_test("SVC",X_2,y_2,0.75,10)

print "SVC with Digits"
my_train_test.my_train_test("SVC",X_3,y_3,0.75,10)

print "LogisticRegression with Boston50"
my_train_test.my_train_test("LogisticRegression",X_1,y_1,0.75,10)

print "LogisticRegression with Boston75"
my_train_test.my_train_test("LogisticRegression",X_2,y_2,0.75,10)

print "LogisticRegression with Digits"
my_train_test.my_train_test("LogisticRegression",X_3,y_3,0.75,10)
