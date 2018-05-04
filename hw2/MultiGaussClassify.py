import numpy as np
from sklearn.utils import check_X_y
from scipy.linalg import logm
from collections import Counter
from numpy.linalg import inv
from numpy.linalg import det
from scipy.sparse import hstack
class MultiGaussClassify():
    # def inverse(self,A):
    #     if (det(A)!=0):
    #         return inv(A)
    #     else:
    #         print("adding noise to the covar matrix")
    #         #A[1,1]+=np.random.normal(0,0.01)
    #         A_rows,A_cols=A.shape
    #         for i in range(A_rows):
    #             for j in range(A_cols):
    #                 if (i==j):
    #
    #                     A[i,j]+=np.random.normal(0,0.0000000001)
    #         return inv(A)



    def __init__(self):
        self.classes_=np.array([0])
        #self.label_count=np.array([(0)])
        self.classes_count=np.array([0])
        self.mean=np.array([(0,0)])
        self.covar=np.array([(0,0)])

    def fit(self,X,y):
        #X,y=check_X_y(X,y,accept_sparse='csr',dtype=np.int64, order="C")
        X=X.astype(float)
        y=y.astype(float)
        self.classes_=np.unique(y)
        X_rows,X_cols=X.shape
        #y_count=np.bincount(y) //only work for int64
        c = Counter(y)
        self.classes_count=np.array([v for k, v in sorted(c.items())])
        #self.classes_count=y_count[self.classes_]
        combine=np.column_stack((X,y))
        #print("combine.shape",combine.shape)
        # X=np.matrix(X)
        # y=np.matrix(y).T
        # combine=hstack([X,y]).toarray()
        # new_combine=np.vstack({tuple(row) for row in combine})
        combine_rows,combine_cols=combine.shape
        # if(new_combine.shape!=combine.shape):
        #     print ("removing duplicate rows")
        self.mean=np.ones((self.classes_.size,X_cols))
        #calculate mean vector
        for i in range(self.classes_.size):
            temp=np.zeros([1,X_cols])
            for j in range(X_rows):
                if (combine[j,combine_cols-1]==self.classes_[i]):
                    temp+=X[j,:]
            temp/=self.classes_count[i]
            self.mean[i,:]=temp

        #initialize covariance matrices
        self.covar=np.empty([X_cols,X_cols,self.classes_.size])
        for k in range(self.classes_.size):
            self.covar[:,:,k]=np.identity(X_cols)
        #calculate covariance matrices

        for i in range(self.classes_.size):
            temp=np.zeros([X_cols,X_cols])
            for j in range(X_rows):
                if (combine[j,combine_cols-1]==self.classes_[i]):
                    #np.matmul is only supported in numpy for python3

                    #temp=temp+np.matmul((X[j,:]-self.mean[i,:])[np.newaxis],np.column_stack((X[j,:]-self.mean[i,:])))
                    temp=temp+np.matmul(((X[j,:]-self.mean[i,:])[np.newaxis]).T,(X[j,:]-self.mean[i,:])[np.newaxis])
                    #print ("temp",temp)
            temp=temp/(self.classes_count[i])
            #print ("self.covar[:,:,i]",temp)
            self.covar[:,:,i]=temp
            #print ("covariance matrix of class ",i,self.covar[:,:,i])
        # #add noise to diagonal elements in the covariance matrices
        # for i in range(self.classes_.size):
        #     temp=np.zeros([X_cols,X_cols])
        #     while(det(self.covar[:,:,i])==0):
        #         c_rows,c_cols=self.covar[:,:,i].shape
        #         for k in range(c_rows):
        #             for j in range(c_cols):
        #                 #if (k==j):
        #                     #self.covar[k,j,i]+=np.random.normal(0,0.0000000001)
        #                 self.covar[k,j,i]+=np.random.randn()


        #add noise to features
        for i in range(self.classes_.size):
            temp=np.zeros([X_cols,X_cols])
            while(det(self.covar[:,:,i])==0):
            #if (det(self.covar[:,:,i])==0):
                for j in range(X_rows):
                    if (combine[j,combine_cols-1]==self.classes_[i]):
                        noise=np.empty([X_cols,X_cols])
                        for k in range(X_cols):
                            for l in range(X_cols):
                                noise[k,l]=np.random.normal(0,0.1)
                        X[j,:]=np.matmul(X[j,:],noise)
                        #X[j,:]=np.matmul(X[j,:],np.random.rand(X_cols,X_cols))
                        #print("adding noise")
                        #X[j,:]*=np.random.randn()
                        temp=temp+np.matmul(((X[j,:]-self.mean[i,:])[np.newaxis]).T,(X[j,:]-self.mean[i,:])[np.newaxis])
                temp/=(self.classes_count[i]-1)
                self.covar[:,:,i]=temp


    def predict(self,T):
        total=np.sum(self.classes_count)
        total_vec=total*np.ones((self.classes_count).shape)
        freq=self.classes_count/total_vec
        log_freq=np.log(freq)
        #prior_prob=np.log(freq)
        T_rows,T_cols=T.shape
        predict_result=np.empty([T_rows,1])
        for k in range(T_rows):
            predict_vec=np.zeros(self.classes_.size)
            for i in range(self.classes_.size):
                #pay attention to sigular matrix
                #print("mean vector:",self.mean[i])
                #print("det of covar",det(self.covar[:,:,i]))
                temp1=np.matmul(np.transpose(T[k,:]-self.mean[i]),(inv(self.covar[:,:,i])))
                temp2=0.5*np.matmul(temp1,(T[k,:]-self.mean[i]))
                # print (-0.5*np.log(det(self.covar[:,:,i])))
                # print (temp2)
                # print (np.log(freq))
                #print (det(self.covar[:,:,i]))
                predict_vec[i]=-0.5*np.log(det(self.covar[:,:,i]))-temp2+log_freq[i]
            predict_class_index=np.argmax(predict_vec)
            predict_result[k,:]=self.classes_[predict_class_index]

        return predict_result
