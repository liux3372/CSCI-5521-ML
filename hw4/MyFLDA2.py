import numpy as np
from sklearn.utils import check_X_y
from scipy.linalg import logm
from collections import Counter
from numpy.linalg import inv
from numpy.linalg import det
from scipy.sparse import hstack
class MyFLDA2():
    def inverse(self,A):
        if (det(A)!=0):
            return inv(A)
        else:
            print("adding noise to the covar matrix")
            #A[1,1]+=np.random.normal(0,0.01)
            A_rows,A_cols=A.shape
            for i in range(A_rows):
                for j in range(A_cols):
                    if (i==j):

                        A[i,j]+=np.random.normal(0,0.0000000001)
            return inv(A)
    def accuracy_one(self,z):
        new_count_one=0
        new_combine_rows,new_combine_cols=self.new_combine.shape
        for i in range(new_combine_rows):
            if (self.new_combine[i,0]>=z and self.new_combine[i,1]==1):
                new_count_one+=1
        return new_count_one/self.count_one

    def accuracy_zero(self,z):
        new_count_zero=0
        new_combine_rows,new_combine_cols=self.new_combine.shape
        for i in range(new_combine_rows):
            if(self.new_combine[i,0]<z and self.new_combine[i,1]==0):
                new_count_zero+=1
        return new_count_zero/self.count_zero

    def __init__(self):
        self.mean_one=np.array([(0,0)],dtype=float)
        self.mean_zero=np.array([(0,0)],dtype=float)
        self.s_one=np.array([(0,0)],dtype=float)
        self.s_zero=np.array([(0,0)],dtype=float)
        self.s_w=np.array([(0,0)],dtype=float)
        self.count_one=0
        self.count_zero=0
        self.constant=1
        self.w=np.array([(0,0)],dtype=float)
        #self.new_X=np.array([(0,0)],dtype=float)
        self.new_z=np.array([(0,0)],dtype=float)
        self.new_combine=np.array([(0,0)],dtype=float)
        self.z_0=0
        self.accuracy=0


    def fit(self,X,y):
        #X,y=check_X_y(X,y,accept_sparse='csr',dtype=np.int64, order="C")
        X=X.astype(float)
        y=y.astype(float)
        X_rows,X_cols=X.shape
        combine=np.column_stack((X,y))
        combine_rows,combine_cols=combine.shape
        # #turn 1D array into 2D
        # X=X[np.newaxis]
        self.mean_one=np.zeros((1,X_cols))
        self.mean_zero=np.zeros((1,X_cols))
        #calculate data m1 and m2
        for i in range(X_rows):
            if(combine[i,combine_cols-1]==1):
                #print ("X[i,:].shape",X[i,:].shape)
                #(13,) means (1,13)
                self.mean_one+=X[i,:]
                self.count_one+=1
            else:
                self.mean_zero+=X[i,:]
                self.count_zero+=1
        self.mean_one/=self.count_one
        self.mean_zero/=self.count_zero
        self.mean_one=self.mean_one.T
        self.mean_zero=self.mean_zero.T
        self.s_one=np.zeros((X_cols,X_cols))
        self.s_zero=np.zeros((X_cols,X_cols))
        #calculate data s1 and s2
        for j in range(X_rows):
            if (combine[j,combine_cols-1]==1):
                temp_one=X[j,:]-(self.mean_one).T
                self.s_one+=np.matmul(temp_one.T,temp_one)
            else:
                temp_zero=X[j,:]-(self.mean_zero).T
                self.s_zero+=np.matmul(temp_zero.T,temp_zero)

        self.s_one/=self.count_one
        self.s_zero/=self.count_zero
        self.s_w=np.add(self.s_one,self.s_zero)
        self.w=np.zeros((X_cols,1))
        self.w=np.matmul(self.constant*self.inverse(self.s_w),(self.mean_one-self.mean_zero))
        #self.new_X=X
        self.new_z=np.zeros((X_rows,1))
        for k in range(X_rows):
            self.new_z[k,0]=np.matmul(self.w.T,(X[k,:][np.newaxis]).T)
            #print("self.new_z[k,0]",self.new_z[k,0])
            #self.new_X[k,0]=np.matmul(self.w.T,self.new_X[k,:].T)
        #self.new_combine=np.column_stack((self.new_X,y))
        self.new_combine=np.column_stack((self.new_z,y))
        #find threshold z_0
        self.z_0=0
        self.accuracy=0
        new_accuracy=0
        m=0
        while (m<X_rows):
            new_accuracy=self.accuracy_one(self.new_z[m,0])+self.accuracy_zero(self.new_z[m,0])
            if(new_accuracy>=self.accuracy):
                self.z_0=self.new_z[m,0]
                self.accuracy=new_accuracy
            m+=1
        #print ("self.z_0",self.z_0)
        #print ("self.accuracy",self.accuracy)



    def predict(self,T):
        T_rows,T_cols=T.shape
        predict_result=np.empty([T_rows,1])
        new_T_X=np.zeros((T_rows,1))
        for i in range(T_rows):
            new_T_X[i,0]=np.matmul(self.w.T,(T[i,:][np.newaxis]).T)
            if(new_T_X[i,0]>=self.z_0):
                predict_result[i,0]=1
            else:
                predict_result[i,0]=0
        return predict_result
