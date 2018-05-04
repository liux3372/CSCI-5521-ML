import my_cross_val
from sklearn.datasets import load_boston
import numpy as np


boston50=load_boston()
X_1, y_1=boston50.data, boston50.target
r50=np.median(y_1)
y_1[y_1>=r50]=1
y_1[y_1!=1]=0
#print y_1
print ("MyFLDA2 with Boston50")
my_cross_val.my_cross_val("MyFLDA2",X_1,y_1,5)

boston75=load_boston()
X_2, y_2=boston75.data, boston75.target
r75=np.percentile(y_2,75)
y_2[y_2>=r75]=1
y_2[y_2!=1]=0
print ("MyFLDA2 with Boston75")
my_cross_val.my_cross_val("MyFLDA2",X_2,y_2,5)

print ("LogisticRegression with Boston50")
my_cross_val.my_cross_val("LogisticRegression",X_1,y_1,5)

print ("LogisticRegression with Boston75")
my_cross_val.my_cross_val("LogisticRegression",X_2,y_2,5)
