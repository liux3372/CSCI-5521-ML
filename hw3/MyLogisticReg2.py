import numpy as np
from sklearn.utils import check_X_y
from scipy.linalg import logm
from collections import Counter
from numpy.linalg import inv
from numpy.linalg import det
from scipy.sparse import hstack
class MyLogisticReg2():
    def sigmoid(self,x):
        if x>=0:
            m=np.exp(-x)
            return 1/(1+m)
        else:
            m=np.exp(x)
            return m/(1+m)
    def reach_peak(self,arr):
        for i in range(arr.size):
            if np.abs(arr[i])>0.001:
                return False
            return True


    def __init__(self):
        self.classes_=np.array([0])
        #self.label_count=np.array([(0)])

        self.weights=np.array([0])


    def fit(self,X,y):
        #X,y=check_X_y(X,y,accept_sparse='csr',dtype=np.int64, order="C")
        X=X.astype(float)
        y=y.astype(float)
        X_rows,X_cols=X.shape
        #augment_ones=np.empty([X_rows,1])
        augment_ones=np.transpose(np.ones(X_rows))
        new_X=np.column_stack((augment_ones,X))
        #new_X[:,0]=1
        self.weights=np.zeros(X_cols+1)
        maxIteration = 850
        step_size=0.00001
        gradient_arr=np.zeros(X_cols+1)
        for t in range(maxIteration):
        # while (True):
            for i in range(X_cols+1):
                gradient=0
                for l in range(X_rows):
                    x=np.matmul(new_X[l,:],np.transpose(self.weights))
                    #print ("x:",x)
                    #gradient+=new_X[l,i]*(y[l]-(np.exp(x)/((np.exp(x)*(1+np.exp(-x))))))
                    gradient+=new_X[l,i]*(y[l]-self.sigmoid(x))
                    #print ("gradient:",gradient)
                gradient_arr[i]+=gradient
            self.weights+=step_size*gradient_arr
            # print ("gradient_arr",gradient_arr)
            # if (self.reach_peak(gradient_arr)):
            #     break
            #print ("self.weights:",self.weights)


    def predict(self,T):
        #prior_prob=np.log(freq)
        T_rows,T_cols=T.shape
        predict_result=np.empty([T_rows,1])
        augment_ones_pred=np.transpose(np.ones(T_rows))
        new_T=np.column_stack((augment_ones_pred,T))
        for k in range(T_rows):
            z=np.matmul(new_T[k,:],np.transpose(self.weights))
            p_C1_x=self.sigmoid(z)
            if p_C1_x>0.5:
                predict_result[k,0]=1
            else:
                predict_result[k,0]=0

        return predict_result
