#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd

data=pd.read_csv('dataset.csv')
TG=np.array(data['Thời gian làm việc']).reshape(-1,1)
L=np.array(data['Lương']).reshape(-1,1)
n1=TG.shape[0]
x1=np.hstack((TG,L))
X = np.hstack((np.ones((n1, 1)),TG,L))
t=np.array(data['Cho vay'])
w=np.random.randn(3).reshape(3,)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def z(theta, x):
    return np.dot(x, theta)


def prob(theta, x):
    return sigmoid(z(theta, x))


def Loss(theta, x, t):
    y = prob(theta, x)
    L = np.matmul(-t.T, np.log(y))+np.matmul(-(1 - t).T, np.log(1 - y))
    return L/n1


def gradient_descent(theta, x, y, learning_rate):
    error = prob(theta, x) - y
    n = (learning_rate / n1 * np.matmul(x.T, error))
    return theta - n



def minimize(theta, x, y, n, learning_rate):
    for _ in range(n):
        theta = gradient_descent(theta, x, y, learning_rate)
    return theta

def them(newdata):
    newdata1=np.concatenate([np.ones([1]),newdata], axis=0)
    return newdata1


def hypo(a,x):
    if prob(a,x)>=0.5:
        print(f'with salary and time ={x[1],x[2]} then prob={prob(a,x)} so In ')
    else:
        print(f'with salary and time ={x[1],x[2]}then prob={prob(a,x)} so OUT ')
z(w, X)
prob(w, X)
a1=minimize(w,X,t,100, 0.02)
F=np.vectorize(them)
b=them([5 , 5])
c=them([2 , 3])
d=them([1 , 8])
hypo(a1,b),hypo(a1,c),hypo(a1,d)


# In[29]:


import matplotlib.pyplot as plt
import numpy as np

# 100 linearly spaced numbers
x = np.linspace(-200,200,100)

def func(a):
    y = a**2
    return y

# setting the axes at the centre
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('zero')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')

A=[]
B=[]
def change(first,rate):
    after=first-rate*2*first
    if func(after)>=0.0000000005 and func(after)<func(first):
        A.append(first)
        B.append(func(first))
        change(after,rate)
        
    if func(after)==func(first):
        A.append([first,after])
        B.append([func(first),func(after)])
        
    if func(after)>func(first):
        if func(after)<=250000:
            A.append(first)
            B.append(func(first))
            change(after,rate)       
    return A,B

plt.plot(x,func(x), 'r')

# change(2,0.2)
# plt.scatter(A,B,facecolor="b")
# for i in range(len(A)):
#     for j in range(len(B)):
#         plt.arrow(A[i], B[j], 0.2,2,head_width = 0.2,width = 0.05)
#         i=i+1
#     break


# change(2,1)
# plt.scatter(A,B,facecolor="b")



# change(2,2)
# plt.scatter(A,B,color='blue')
# plt.plot(A,B)
# plt.show()


# In[ ]:




