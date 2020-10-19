import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('ex2data1.txt',header=None)

data.head()

data.describe()

data_n = data.values
m = len(data_n[:,-1])
X = data_n[:,0:2].reshape(m,2)
y = data_n[:,-1].reshape(m,1)

pos , neg = (y==1).reshape(100,1) , (y==0).reshape(100,1)
plt.scatter(X[pos[:,0],0],X[pos[:,0],1],c="r",marker="+")
plt.scatter(X[neg[:,0],0],X[neg[:,0],1],marker="o",s=10)
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(["Admitted","Not admitted"],loc=0)

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

#testing sigmoid
print(sigmoid(0))

def costFunction(X,y,theta):
  m = len(y)
  predictions = sigmoid(np.dot(X,theta))
  error = (-y * np.log(predictions) - (1-y) * np.log(1 - predictions))
  cost = (1/m) * sum(error)
  
  grad = (1/m) * np.dot(X.transpose(),(predictions - y))
  
  return cost[0], grad

def featureNormalize(X):
  mean = np.mean(X,axis=0)
  std = np.std(X,axis=0)
  X_norm = (X-mean)/std
  
  return X_norm , mean , std

cost , grad = costFunction(X,y,initial_theta)
print("Cost of initial theta is",cost)
print("Gradient at initial theta (zeros):",grad)

data_n = data.values
m = len(data_n[:,-1])
X = data_n[:,0:2].reshape(m,2)
X , X2_mean , X2_std = featureNormalize(X)

y = data_n[:,-1].reshape(m,1)


m , n = X.shape[0] , X.shape[1]
X = np.append(np.ones((m,1)),X,axis=1)
y = y.reshape(m,1)
initial_theta = np.zeros((n+1,1))

def gradientDescent(X,y,theta,alpha,num_iters):
  m = len(y)
  J_history = []
  for i in range(num_iters):
    cost,grad = costFunction(X,y,theta)
    theta -= (alpha * grad)
    J_history.append(cost)
    
  return theta, J_history

alpha = 1
num_iters = 400

theta , J_history = gradientDescent(X,y,initial_theta,alpha,num_iters)

plt.plot(J_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")
plt.title("Cost function using Gradient Descent")

print(J_history[-1])

plt.scatter(X[pos[:,0],1],X[pos[:,0],2],c="r",marker="+",label="Admitted")
plt.scatter(X[neg[:,0],1],X[neg[:,0],2],c="b",marker="x",label="Not admitted")
x_value= np.array([np.min(X[:,1]),np.max(X[:,1])])
y_value=-(theta[0] +theta[1]*x_value)/theta[2]
plt.plot(x_value,y_value, "r")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend(loc=0)

x_test = np.array([45,85])
x_test = (x_test - X2_mean)/X2_std
x_test = np.append(np.ones(1),x_test)
prob = sigmoid(x_test.dot(theta))
print("For a student with scores 45 and 85, we predict an admission probability of",prob[0])

def classifierPredict(theta,X):
    predictions = X.dot(theta)
    
    return predictions>0

p=classifierPredict(theta,X)
print("Train Accuracy:", sum(p==y)[0],"%")



