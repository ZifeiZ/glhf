#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


# In[14]:


def LinearRegressionModel(X, y, with_fit_intercept):
    # Inputs:                                                                   
    # X = N x d                                                                 
    # y = N x 1                                                                 
    # Output:                                                                   
    # Linear regression model
    
     clf = LinearRegression(fit_intercept = with_fit_intercept).fit(X, y)
     return clf


# In[15]:


def learnOLERegression(X,y):
    # Inputs:                                                                   
    # X = N x d                                                                 
    # y = N x 1                                                                 
    # Output:                                                                   
    # w = d x 1                                                                 

    X_transpose = X.transpose()
    w = np.dot(np.dot(inv(np.dot(X_transpose, X)), X_transpose), y)
    return w


# In[16]:


def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse

    mse = 0
    N = Xtest.shape[0]
    for i in range(N):
        mse += ((ytest[i] - np.dot(w.transpose(), Xtest[i])) ** 2)
    mse = np.sum(mse) / N
    return mse


# In[17]:


def learnRidgeRegression(X,y,lambd):
    # Inputs:                                                                   
    # X = N x d                                                                 
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                   
    # w = d x 1                           

    X_trans = X.transpose()
    I = np.eye(X_trans.shape[0])
    w = np.dot(np.dot(inv(np.dot(X_trans, X) + lambd * I), X_trans), y)
    return w


# In[18]:


def RidgeRegressionModel(X, y, alpha_value):
    # Inputs:                                                                   
    # X = N x d                                                                 
    # y = N x 1                                                                 
    # Output:                                                                   
    # Linear regression model
    
    clf = Ridge(alpha = alpha_value).fit(X, y)
    return clf


# In[19]:


def mapNonLinear(x,p):
    # Inputs:                                                                   
    # x - a single column vector (N x 1)                                        
    # p - integer (>= 0)                                                        
    # Outputs:                                                                  
    # Xd - (N x (d+1))                                                         

    Xd = np.zeros((len(x), p+1))
    for i in range(p+1):
        Xd[:,i] = x**i
    return Xd


# In[20]:


def regressionObjVal(w, X, y, lambd):
    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    w = np.transpose(np.asmatrix(w))
    error = np.dot((y - np.dot(X, w)).transpose(),(y - np.dot(X, w))) + np.dot((lambd * w.transpose()), w)
    error_grad = np.dot(-2 * X.transpose(),(y - np.dot(X, w))) + 2 * lambd * w
    error_grad = np.squeeze(np.array(error_grad))
    return error, error_grad


# In[21]:


# Main script

# Input data
X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding='latin1')   
# add intercept
x1 = np.ones((len(X),1))
x2 = np.ones((len(Xtest),1))

X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)


# Scikit package
clf_without_intercept = LinearRegressionModel(X, y, False)
w_lr = clf_without_intercept.coef_[0]
w_lr = w_lr.reshape((len(w_lr), 1))
mle_lr = testOLERegression(w_lr,Xtest,ytest)
print('Sklearn MSE without intercept '+str(mle_lr))

clf_with_intercept = LinearRegressionModel(X, y, True)
w_i_lr = np.concatenate((clf_with_intercept.intercept_, clf_with_intercept.coef_[0]), axis = 0)
w_i_lr = w_i_lr.reshape((len(w_i_lr), 1))
mle_i_lr = testOLERegression(w_i_lr,Xtest_i,ytest)
print('Sklearn MSE with intercept '+str(mle_i_lr))

# Matrix computation 
w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)

w_i = learnOLERegression(X_i,y)
mle_i = testOLERegression(w_i,Xtest_i,ytest)

print('MSE without intercept '+str(mle))
print('MSE with intercept '+str(mle_i))


# In[22]:


k = 101
alphas = np.linspace(0, 1, num=k)
i = 0
mses2_train_rr = np.zeros((k,1))
mses2_rr = np.zeros((k,1))
for alpha in alphas:
    clf = RidgeRegressionModel(X,y,alpha)
    w_l_rr = np.concatenate((clf.intercept_, clf.coef_[0]), axis = 0)
    w_l_rr = w_l_rr.reshape((len(w_l_rr), 1))
    mses2_train_rr[i] = testOLERegression(w_l_rr,X_i,y)
    mses2_rr[i] = testOLERegression(w_l_rr,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(alphas,mses2_train_rr)
plt.title('Sklearn MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(alphas,mses2_rr)
plt.title('Sklearn MSE for Test Data')

# mses2_rr represents the errors on the testing data of different alpha's 
# 1. obtain the smallest mse
min_mse_sklearn = np.min(mses2_rr)
print('The smallest Sklearn MSE:', min_mse_sklearn)
# 2. obtain the index of the smallest mse
index_min_mse_sklearn = np.argmin(mses2_rr)
# 3. obtain the corresponding alpha
min_alpha_sklearn = alphas[index_min_mse_sklearn]
print('The corresponding alpha for Sklearn:', min_alpha_sklearn)


i = 0
mses2_train = np.zeros((k,1))
mses2 = np.zeros((k,1))
for alpha in alphas:
    w_l = learnRidgeRegression(X_i,y,alpha)
    mses2_train[i] = testOLERegression(w_l,X_i,y)
    mses2[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(alphas,mses2_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(alphas,mses2)
plt.title('MSE for Test Data')

# mses2 represents the errors on the testing data of different alpha's 
# 1. obtain the smallest mse
min_mse_ole = np.min(mses2)
print('The smallest OLE MSE:', min_mse_ole)
# 2. obtain the index of the smallest mse
index_min_mse_ole = np.argmin(mses2)
# 3. obtain the corresponding alpha
min_alpha_ole = alphas[index_min_mse_ole]
print('The corresponding alpha for OLE:', min_alpha_ole)


# In[23]:


pmax = 7
#alpha_opt = 0 # REPLACE THIS WITH alpha_opt estimated from Problem 2
alpha_opt = alphas[np.argmin(mses2)]
mses3_train = np.zeros((pmax,2))
mses3 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses3_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses3[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,alpha_opt)
    mses3_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses3[p,1] = testOLERegression(w_d2,Xdtest,ytest)
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses3_train)
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))

plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses3)
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))

# print the training erros for different p's
# the first columun means the errors of linear models without using the regularization term
print('Training errors of linear models without using the regularization term:')
for i in range(pmax):
    print('p =',i, 'training error =', mses3_train[i][0])
# the second columun means the errors of linear models using the regularization term

print('Training errors of linear models using the regularization term:')
for i in range(pmax):
    print('p =',i, 'training error =', mses3_train[i][1])

# print the testing erros for different p's
# the first columun means the errors of linear models without using the regularization term
print('Testing errors of linear models without using the regularization term:')
for i in range(pmax):
    print('p =',i, 'testing error =', mses3[i][0])
# the second columun means the errors of linear models using the regularization term

print('Testing errors of linear models using the regularization term:')
for i in range(pmax):
    print('p =',i, 'testing error =',mses3[i][1])


# In[24]:


fig = plt.figure(figsize=[10,8])
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses5_train = np.zeros((k,1))
mses5 = np.zeros((k,1))

opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses5_train[i] = testOLERegression(w_l,X_i,y)
    mses5[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses5_train)
plt.plot(lambdas,mses2_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses5)
plt.plot(lambdas,mses2)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])


# In[ ]:




