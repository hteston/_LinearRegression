#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LinearRegression


# ## Explore Bitcoin Prices ##

# In[2]:


df = pd.read_csv('bitcoin.csv')
df


# In[3]:


plt.figure(figsize=(9,9))
plt.plot(df.index, df['Close'])
plt.title('Visualizing Predicted Closing Value vs Actual Closing Value')
plt.show()


# In[4]:


plt.figure(figsize=(10,10))
plt.scatter(df['Open'], df['Close'])
plt.show()


# In[5]:


X = df[['Open']]
y = df[['Close']]

X_train = X.iloc[:-30,:]
y_train = y.iloc[:-30,:]
X_test = X.iloc[-30:,:]
y_test = y.iloc[-30:,:]


# In[6]:


plt.figure(figsize=(5,5))
plt.scatter(X_train, y_train, color='cornflowerblue')
plt.scatter(X_test, y_test, color='darkorange')
plt.title('Visualizing Test Data vs Training Data')
plt.show()


# In[7]:


model = LinearRegression()
model.fit(X, y)


# In[8]:


m = model.coef_[0][0]
b = model.intercept_[0]
print(f'y = {m}x + {b}')


# In[9]:


# PLOT OUR LINEAR REGRESSION VS TRAINING DATA

x0 = min(X_train.min().values[0], X_test.min().values[0])
x1 = max(X_train.max().values[0], X_test.max().values[0])
y0 = m * x0 + b
y1 = m * x1 + b

plt.figure(figsize=(5,5))
plt.scatter(X_train, y_train, color='cornflowerblue')
plt.plot((x0, x1), (y0, y1), color='mediumblue')
plt.title('Visualizing Linear Regression Line vs Training Data')
plt.show()


# In[10]:


# ZOOM IN ON NEW DATA

x0 = X_test.min().values[0]
x1 = X_test.max().values[0]
y0 = m * x0 + b
y1 = m * x1 + b

plt.figure(figsize=(5,5))
plt.scatter(X_test, y_test, color='darkorange')
plt.plot((x0, x1), (y0, y1), color='mediumblue')
plt.title('Visualizing Linear Regression Line vs Actual Data')
plt.show()


# In[11]:


y_pred = model.predict(X_test)


# In[12]:


# PLOT PREDICTION VS ACTUAL INCLUDING OUR LINEAR REGRESSION LINE

plt.figure(figsize=(5,5))
plt.scatter(X_test, y_test, color='cornflowerblue')
plt.scatter(X_test, y_pred, color='darkorange')
plt.plot((x0, x1), (y0, y1), color='mediumblue')
plt.title('Visualizing Linear Prediction Data vs Actual Data')
plt.show()


# In[13]:


# NOW SWITCH BACK TO CLOSING VALUE OVER TIME

plt.figure(figsize=(9,9))
plt.plot(X_train.index, X_train, color='cornflowerblue')
plt.plot(y_test.index, y_pred, color='mediumblue')
plt.plot(y_test.index, y_test, color='darkorange')
plt.title('Visualizing Predicted Closing Value vs Actual Closing Value')
plt.show()


# In[14]:


# NEED TO ZOOM IN ON PREDICTIONS

plt.figure(figsize=(10,10))
plt.plot(y_test.index, y_pred, color='mediumblue', label='Model Prediction')
plt.plot(y_test.index, y_test, color='darkorange', label='Actual Closing Value')
plt.title('Visualizing Linear Prediction Data vs Actual Data')
plt.legend()
plt.show()


# In[15]:


model.score(X_train, y_train) # R^2 score


# In[ ]:





# ## Exploring Housing Prices ##

# In[17]:


df = pd.read_csv('housing_data.csv')
df.head()


# In[18]:


plt.figure(figsize=(10,10))
plt.scatter(df['median_income'], df['median_house_value'])
plt.show()


# In[19]:


plt.scatter(df['longitude'], df['latitude'])
plt.show()


# In[20]:


df['ocean_proximity'].value_counts()


# In[21]:


encoder = OrdinalEncoder()
encoder.fit(df[['ocean_proximity']])
df['location'] = encoder.transform(df[['ocean_proximity']])
df['location'].value_counts()


# In[22]:


import matplotlib.patches as mpatches

plt.figure(figsize=(10,10))
plt.scatter(df['longitude'], df['latitude'], c=df['location'], cmap='tab10')
plt.show()


# In[ ]:

def predict(self, X):
      
  return np.dot(X, self._W)

  def _gradient_descent_step(self, X, targets, lr):
      
    predictions = self.predict(X)
  
  error = predictions - targets
  
  gradient = np.dot(X.T,  error) / len(X)
  
  self._W -= lr * gradient
    

  def fit(self, X, y, n_iter=100000, lr=0.01):

    self._W = np.zeros(X.shape[1])
    for i in range(n_iter):        
      self._gradient_descent_step(x, y, lr)       
      
  return self


# In[ ]:


x = df_train['house_id']
y = df_train['median_house_value']

x = (x - x.mean()) / x.std()
x = np.c_[np.ones(x.shape[0]), x] 

