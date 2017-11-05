import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
path = os.getcwd() + '\data\ex1data1.txt'
data = pd.read_csv(path, header=None, names=['Population', 'Profit'])
#print(data.head())
#print(data.describe())
#data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))



def compute_cost(X, y, theta):
    cost = np.power(((X * theta.T ) - y) , 2)
    return np.sum(cost) / (2 * len(X))

cols = data.shape[1]
data.insert(0 , 'Ones' , 1)

X = data.iloc[:,0:cols]
y = data.iloc[:,cols:cols+1]
#print(X)
X = np.matrix(X.values)
y = np.matrix(y.values)

theta = np.matrix(np.array([0,0]))

#print(X.shape, theta.shape, y.shape )
print(compute_cost(X, y, theta))


def gradientDescent(X, y, theta, alpha, iters):
    #print('Hey' + str(theta.shape))
    temp =np.matrix(np.zeros(theta.shape))
    parameters = int(theta.shape[1])
    cost= np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term =  np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha/len(X)) * np.sum(term) )
            
        theta = temp
        cost[i] = compute_cost(X, y, theta)
            
    return theta , cost


alpha = 0.02
iters = 5000
g , cost = gradientDescent(X, y, theta, alpha, iters)
print(g,cost[-1])



x =np.linspace(data.Population.min(),data.Population.max(),100)
f = g[0, 0] + (g[0,1] * x)
print(f)
fig , ax = plt.subplots(figsize=(12,8))
ax.plot(x,f,'k',label='Prediction')
ax.scatter(data.Population,data.Profit,label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted profit vs Population size')


figg, axx = plt.subplots(figsize=(12,8))
axx.plot(np.arange(iters),cost,'r')
axx.set_xlabel('Iterations')
axx.set_label('Cost')
axx.set_title('Error vs. Training Epoch')
print(cost)
