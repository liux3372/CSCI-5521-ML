import numpy as np
from sklearn.utils import check_X_y
from scipy.linalg import logm
from collections import Counter
from numpy.linalg import inv
from numpy.linalg import det
from scipy.sparse import hstack
import scipy
class MyLogisticRegGen():
    def softmax(self,x):
        x-=np.max(x)
        result=(np.exp(x).T / np.sum(np.exp(x),axis=1)).T
        return result
    #transform y into onHot representation
    def oneHotIt(self,Y):
        m = Y.shape[0]
        #Y = Y[:,0]
        OHX = scipy.sparse.csr_matrix((np.ones(m), (Y, np.array(range(m)))))
        OHX = np.array(OHX.todense()).T
        return OHX


    def __init__(self):
        self.classes_=np.array([0])
        #self.label_count=np.array([(0)])
        self.classes_count=np.array([0])
        self.weights=np.array([0])


    def fit(self,X,y):
        #X,y=check_X_y(X,y,accept_sparse='csr',dtype=np.int64, order="C")
        X=X.astype(float)
        y=y.astype(float)
        self.classes_=np.unique(y)
        X_rows,X_cols=X.shape
        #print ("X_rows,X_cols:",X_rows,X_cols)
        counter = Counter(y)
        self.classes_count=np.array([v for k, v in sorted(counter.items())])
        augment_ones=np.transpose(np.ones(X_rows))
        new_X=np.column_stack((augment_ones,X))
        new_y=self.oneHotIt(y)
        #self.weights=np.random.randn(self.classes_.size,X_cols+1)#attention to dimension
        self.weights=np.zeros([self.classes_.size,X_cols+1])
        #print ("self.weights.size:",self.weights.size)
        maxIteration = 300
        step_size=0.00001
        lam=1
        #gradient_arr=np.zeros([self.classes_.size,X_cols+1])
        x_arr=np.zeros(self.classes_.size)
        gradient=np.zeros([X_cols+1,self.classes_.size])
        # #add noise to features
        # noise=np.empty([X_cols+1,X_cols+1])
        # for k in range(X_cols):
        #     for l in range(X_cols):
        #         noise[k,l]=np.random.normal(0,0.1)
        # new_X=np.matmul(new_X,noise)
        # for c in range(self.classes_.size):
        #     for t in range(maxIteration):
        for t in range(maxIteration):
        #while(gradient):
            #print ("class,iteration:",c,t)
            print("iteration:",t)
            # print ("new_X.shape",new_X.shape)
            # print ("self.weights.shape",self.weights.shape)
            x=np.matmul(new_X,np.transpose(self.weights))
            softmax_arr=self.softmax(x)
            for i in range(X_cols+1):
                    #gradient=0
                for l in range(X_rows):
                        # x=np.matmul(new_X[l,:],np.transpose(self.weights[c,:]))
                        # x_arr[c]+=x
                        # if (y[l]==self.classes_[c]):
                        #     indicator=1
                        # else:
                        #     indicator=0
                        # if x>0:
                        #     n=np.exp(-x)
                        #     gradient+=new_X[l,i]*(indicator-(1/(n*(1+np.sum(x_arr)))))
                        # else:
                        #     gradient+=new_X[l,i]*(indicator-(np.exp(x)/(1+np.sum(x_arr))))
                        #print ("self.softmax(x).shape",self.softmax(x).shape)
                    #class_num=np.where(self.classes_==(y[l]))
                    #indicator=1
                    # print ("new_y.shape",new_y.shape)
                    # print ("softmax_arr.shape",softmax_arr.shape)
                    gradient[i,:]+=new_X[l,i]*(new_y[l,:]-(softmax_arr[l,:]))
                    # for c in range(self.classes_.size):
                    #     if (c!=class_num):
                    #         gradient[c,i]+=new_X[l,i]*(0-(softmax_arr[l,c]))
                        #print ("gradient:",gradient)
                #gradient[:,i]+=gradient[:,i]
                #gradient[i,:]=gradient[i,:]

                #gradient[i,:]=gradient[i,:]*(-1/X_rows)
                # -lam*(self.weights[:,i]).T #with lambda
                #no lambda term
            self.weights=self.weights*(1-step_size*lam)+step_size*(gradient.T)
            #self.weights=self.weights+step_size*(gradient.T)
            # print ("gradient_arr",gradient_arr)
            # if (self.reach_peak(gradient_arr)):
            #     break
            #print ("self.weights:",self.weights)
        #print ("self.weights",self.weights)

    def predict(self,T):
        #prior_prob=np.log(freq)
        T_rows,T_cols=T.shape
        predict_result=np.empty([T_rows,1])
        augment_ones_pred=np.transpose(np.ones(T_rows))
        new_T=np.column_stack((augment_ones_pred,T))
        prob_arr=np.zeros(self.classes_.size)
        z=np.matmul(new_T,np.transpose(self.weights))
        #print ("z",z)
        for k in range(T_rows):
            prob_arr=np.zeros(self.classes_.size)
            #print ("self.softmax(z)[k,:]",self.softmax(z)[k,:])
            prob_arr+=self.softmax(z)[k,:]
            #prob_arr[self.classes_.size-1]+=1
            predict_class_index=np.argmax(prob_arr)
            predict_result[k,:]=self.classes_[predict_class_index]
            #print ("prob_arr",prob_arr)


        return predict_result
