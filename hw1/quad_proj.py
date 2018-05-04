import my_cross_val
from sklearn.datasets import load_digits
import numpy as np
#original features:64
#squares of original features:64
#products of all the original features xijxij0 ;
#j < j0; j = 1; : : : ; 64; j0 = j+1; : : : ; 64.
#number of products:(1+63)*(63)/2=2016
def cal_products(data):
    products=list()
    for j in range(np.shape(data)[0]-1,0,-1):
        for i in range(j):
            #print (i,j)
            products.append(data[i]*data[j])
    products_arr=np.array(products)
    return products_arr


def quad_proj(X):
    row_num=np.shape(X)[0]
    col_num=np.shape(X)[1]
    new_col_num=col_num*2+((1+col_num-1)*(col_num-1)/2)
    X_new=np.empty((row_num,new_col_num))
    for observ in range(row_num):
        #1*64
        data=X[observ,:]
        #1*64
        squares=data*data
        #1*2016
        products=cal_products(data)
        temp=np.append(squares,products)
        X_new[observ,:]=np.append(X[observ,:],temp)

    #X_new=(np.matrix(X))*(np.matrix(G)),same as below
    #X_new=np.dot(X,G)
    return X_new


# digits=load_digits()
# X,y=digits.data, digits.target
#X_1=cal_products(X[0,:])
# X_1=quad_proj(X)
# print type(X_1)
# print np.shape(X_1)




# print "LinearSVC with Digits"
# main.my_cross_val("LinearSVC",X_1,y,10)
#
# print "SVC with Digits"
# main.my_cross_val("SVC",X_1,y,10)
#
# print "LogisticRegression with Digits"
# main.my_cross_val("LogisticRegression",X_1,y,10)
