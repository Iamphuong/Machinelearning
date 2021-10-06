#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

np.random.seed(1234)

def create_toy_data(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t

def func(x):
    return np.sin(2 * np.pi * x)

def sampledata(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t

x_train, y_train = create_toy_data(func, 100, 0.25)
x_test = np.linspace(0, 1, 100)
y_test = func(x_test)
x_val,y_val=create_toy_data(func, 10, 0.25)

n1 = x_train.shape[0]
n2=x_val.shape[0]
n3=x_test.shape[0]



def solution(X,target):
    A=np.linalg.inv(np.matmul(X.T, X))
    B=np.dot(X.T,target)
    result=np.matmul(A, B)
    return result

def mse(fitted, y):
    return np.power(fitted-y,2).sum()/y.shape[0]
 
    
    
#bac1
# x = np.hstack((np.ones((n1, 1)), x_train.reshape(-1,1)))
# x1= np.hstack((np.ones((n2, 1)), x_val.reshape(-1,1)))
# x0=np.hstack((np.ones((n3, 1)), x_test.reshape(-1,1)))

# T=solution(x,y_train)

# predict1=np.dot(x,T)
# predict2=np.dot(x1,T)
# predict0=np.dot(x0,T)

# A=mse(predict1,y_train)
# B=mse(predict2,y_val)
# print(f'train loss={A}')
# print(f'val loss={B}')

# plt.plot(x_test, predict0, color = "black")
# plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
# plt.scatter(x_val, y_val, facecolor="r", edgecolor="r", s=20, label="validation data")
# plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
# plt.legend()
# plt.show()


# #bac3

# x3=np.hstack((np.ones((n1, 1)), x_train.reshape(-1,1),np.power(x_train, 2).reshape(-1,1),np.power(x_train, 3).reshape(-1,1)))
# x4=x=np.hstack((np.ones((n2, 1)), x_val.reshape(-1,1),np.power(x_val, 2).reshape(-1,1),np.power(x_val, 3).reshape(-1,1)))
# x5=np.hstack((np.ones((n3, 1)), x_test.reshape(-1,1),np.power(x_test, 2).reshape(-1,1),np.power(x_test, 3).reshape(-1,1)))

# C=solution(x3,y_train)

# predict3=np.dot(x3,C)
# predict4=np.dot(x4,C)
# predict5=np.dot(x5,C)

# A1=mse(predict3,y_train)
# B1=mse(predict4,y_val)
# print(f'train loss={A1}')
# print(f'val loss={B1}')

# plt.plot(x_test, predict5, color = "black")
# plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
# plt.scatter(x_val, y_val, facecolor="r", edgecolor="r", s=20, label="validation data")
# plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
# plt.legend()
# plt.show()


#bac9

x9=np.hstack((np.ones((n1, 1)), x_train.reshape(-1,1),np.power(x_train, 2).reshape(-1,1),np.power(x_train, 3).reshape(-1,1),np.power(x_train, 4).reshape(-1,1),np.power(x_train, 5).reshape(-1,1),np.power(x_train, 6).reshape(-1,1),np.power(x_train, 7).reshape(-1,1),np.power(x_train, 8).reshape(-1,1),np.power(x_train, 9).reshape(-1,1)))
x10 =np.hstack((np.ones((n2, 1)), x_val.reshape(-1,1),np.power(x_val, 2).reshape(-1,1),np.power(x_val, 3).reshape(-1,1),np.power(x_val, 4).reshape(-1,1),np.power(x_val, 5).reshape(-1,1),np.power(x_val, 6).reshape(-1,1),np.power(x_val, 7).reshape(-1,1),np.power(x_val, 8).reshape(-1,1),np.power(x_val, 9).reshape(-1,1)))
x11=np.hstack((np.ones((n3, 1)), x_test.reshape(-1,1),np.power(x_test, 2).reshape(-1,1),np.power(x_test, 3).reshape(-1,1),np.power(x_test, 4).reshape(-1,1),np.power(x_test, 5).reshape(-1,1),np.power(x_test, 6).reshape(-1,1),np.power(x_test, 7).reshape(-1,1),np.power(x_test, 8).reshape(-1,1),np.power(x_test, 9).reshape(-1,1)))

C=solution(x9,y_train)

predict9=np.dot(x9,C)
predict10=np.dot(x10,C)
predict11=np.dot(x11,C)

A11=mse(predict9,y_train)
B11=mse(predict10,y_val)
print(f'train loss={A11}')
print(f'val loss={B11}')

plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.scatter(x_val, y_val, facecolor="r", edgecolor="r", s=20, label="validation data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, predict11, color = "black")
plt.legend()
plt.show()


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.model_selection import train_test_split

np.random.seed(1234)

def create_toy_data(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t

def func(x):
    return np.sin(2 * np.pi * x)

def sampledata(func, sample_size, std):
    x = np.linspace(0, 1, sample_size)
    t = func(x) + np.random.normal(scale=std, size=x.shape)
    return x, t

x_train, y_train = create_toy_data(func, 10, 0.25)
x_test = np.linspace(0, 1, 100)
y_test = func(x_test)
x_val,y_val=create_toy_data(func, 10, 0.25)

n1 = x_train.shape[0]
n2=x_val.shape[0]
n3=x_test.shape[0]

x9=np.hstack((np.ones((n1, 1)), x_train.reshape(-1,1),np.power(x_train, 2).reshape(-1,1),np.power(x_train, 3).reshape(-1,1),np.power(x_train, 4).reshape(-1,1),np.power(x_train, 5).reshape(-1,1),np.power(x_train, 6).reshape(-1,1),np.power(x_train, 7).reshape(-1,1),np.power(x_train, 8).reshape(-1,1),np.power(x_train, 9).reshape(-1,1)))
x10 =np.hstack((np.ones((n2, 1)), x_val.reshape(-1,1),np.power(x_val, 2).reshape(-1,1),np.power(x_val, 3).reshape(-1,1),np.power(x_val, 4).reshape(-1,1),np.power(x_val, 5).reshape(-1,1),np.power(x_val, 6).reshape(-1,1),np.power(x_val, 7).reshape(-1,1),np.power(x_val, 8).reshape(-1,1),np.power(x_val, 9).reshape(-1,1)))
x11=np.hstack((np.ones((n3, 1)), x_test.reshape(-1,1),np.power(x_test, 2).reshape(-1,1),np.power(x_test, 3).reshape(-1,1),np.power(x_test, 4).reshape(-1,1),np.power(x_test, 5).reshape(-1,1),np.power(x_test, 6).reshape(-1,1),np.power(x_test, 7).reshape(-1,1),np.power(x_test, 8).reshape(-1,1),np.power(x_test, 9).reshape(-1,1)))


def mse(fitted, y):
    return np.power(fitted-y,2).sum()/y.shape[0]


def solution(X,target):
    A=np.linalg.inv(np.matmul(X.T, X)+ 0.0002*np.identity(X.shape[0]))
    B=np.dot(X.T,target)
    result=np.matmul(A, B)
    return result




x9=np.hstack((np.ones((n1, 1)), x_train.reshape(-1,1),np.power(x_train, 2).reshape(-1,1),np.power(x_train, 3).reshape(-1,1),np.power(x_train, 4).reshape(-1,1),np.power(x_train, 5).reshape(-1,1),np.power(x_train, 6).reshape(-1,1),np.power(x_train, 7).reshape(-1,1),np.power(x_train, 8).reshape(-1,1),np.power(x_train, 9).reshape(-1,1)))
x10 =np.hstack((np.ones((n2, 1)), x_val.reshape(-1,1),np.power(x_val, 2).reshape(-1,1),np.power(x_val, 3).reshape(-1,1),np.power(x_val, 4).reshape(-1,1),np.power(x_val, 5).reshape(-1,1),np.power(x_val, 6).reshape(-1,1),np.power(x_val, 7).reshape(-1,1),np.power(x_val, 8).reshape(-1,1),np.power(x_val, 9).reshape(-1,1)))
x11=np.hstack((np.ones((n3, 1)), x_test.reshape(-1,1),np.power(x_test, 2).reshape(-1,1),np.power(x_test, 3).reshape(-1,1),np.power(x_test, 4).reshape(-1,1),np.power(x_test, 5).reshape(-1,1),np.power(x_test, 6).reshape(-1,1),np.power(x_test, 7).reshape(-1,1),np.power(x_test, 8).reshape(-1,1),np.power(x_test, 9).reshape(-1,1)))

C=solution(x9,y_train)
predict9=np.dot(x9,C)
predict10=np.dot(x10,C)
predict11=np.dot(x11,C)


A11=mse(predict9,y_train)
B11=mse(predict10,y_val)
print(f'train loss={A11}')
print(f'val loss={B11}')


plt.scatter(x_train, y_train, facecolor="none", edgecolor="b", s=50, label="training data")
plt.scatter(x_val, y_val, facecolor="r", edgecolor="r", s=20, label="validation data")
plt.plot(x_test, y_test, c="g", label="$\sin(2\pi x)$")
plt.plot(x_test, predict11, color = "black")
plt.plot(x_test, predict11, color = "black")


# In[ ]:




