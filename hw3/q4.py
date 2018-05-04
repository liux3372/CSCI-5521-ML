import my_cross_val
from sklearn.datasets import load_digits
from sklearn.datasets import load_boston
import numpy as np
#from quad_proj import quad_proj
from MyLogisticRegGen import MyLogisticRegGen

digits=load_digits()
X_3,y_3=digits.data, digits.target

X_rows,X_cols=X_3.shape
#normalize each feature vector
for i in range(X_cols):
    #standardization
    feature_mean=np.mean(X_3[:,i])
    feature_std=np.std(X_3[:,i])
    if (feature_std==0):
        feature_std=1
    X_3[:,i]=(X_3[:,i]-feature_mean)/feature_std

print ("MyLogisticRegGen with Digits")

my_cross_val.my_cross_val("MyLogisticRegGen",X_3,y_3,5)


X_3_logistic,y_3_logistic=digits.data, digits.target
print ("LogisticRegression with Digits")
my_cross_val.my_cross_val("LogisticRegression",X_3_logistic,y_3_logistic,5)
