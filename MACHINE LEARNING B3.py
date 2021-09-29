#!/usr/bin/env python
# coding: utf-8

# # BAI 2

# In[6]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
data= pd.read_csv('data_linear.csv')
X=np.array(data['Diện tích'])
Y=np.array(data['Giá'])

X_input=np.concatenate([X.reshape(-1,1),np.ones([np.shape(X)[0], 1])], axis=1)

def findw(x,target):
    A=np.linalg.inv(np.matmul(x.T, x))
    B=np.dot(x.T,target)
    result=np.matmul(A, B)
    def model(x1):
        return x1*result[0]+result[1]
    return model

predict=findw(X_input,Y)
plt.scatter(X,Y)
plt.plot(X,predict(X))
predict(50),predict(100),predict(150)


# # BAI 3

# In[7]:


import numpy as np
from sklearn.datasets import load_boston

boston_data = load_boston()
#there are many features that we don't really need, we only use ROOM AVERAGE AND DIS(weighted distances to five Boston employment centres)
data = boston_data['data']
x = data[:, [5,7]]
t=boston_data['target']

X_input=np.concatenate([x,np.ones([np.shape(x)[0], 1])], axis=1)

def solution(X,target):
    A=np.linalg.inv(np.matmul(X.T, X))
    B=np.dot(X.T,target)
    result=np.matmul(A, B)
    return result

def modellinear(newdata,w):
    newdata1=np.concatenate([newdata,np.ones([1])], axis=0)
    res= np.dot(newdata1,w)
    return res
C=solution(X_input,t)
F=np.vectorize(modellinear)
modellinear([6.575 , 4.09],C)

