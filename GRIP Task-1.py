#!/usr/bin/env python
# coding: utf-8

# # Name = Anuj Kumar

# # Task-1

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


url = "http://bit.ly/w-data"
data=pd.read_csv(url)
print("Data import succesfully:")
data.head()


# # Reading Data

# In[3]:


data.isnull().sum()


# In[4]:


data.info()


# In[5]:


data.dtypes


# In[6]:


data.describe()


# # Find outliers

# In[7]:


data.boxplot('Hours')


# In[8]:


data.boxplot('Scores')


# In[9]:


plt.style.use('seaborn-darkgrid')
plt.title('Hours vs Percentage')
sns.scatterplot(x=data.Hours, y=data.Scores, color="Blue")


# # Check Linearity

# In[10]:


sns.pairplot(data)


# # Check Normality

# In[11]:


sns.distplot(data['Hours'])


# In[12]:


sns.distplot(data['Scores'])


# In[13]:


data.skew()


# In[14]:


sns.heatmap(data.corr(),annot=True)


# In[15]:


sns.regplot(x=data.Hours, y=data.Scores, color="Blue",fit_reg=True)


# # Preparing Data

# In[16]:


x=data.iloc[:,:-1].values
y=data.iloc[:,1].values


# In[17]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[18]:


y_test


# # Fit the model

# In[19]:


from sklearn.linear_model import LinearRegression
lr = LinearRegression()  
lr.fit(x_train, y_train)


# In[20]:


y_pred=lr.predict(x_test)
y_pred


# In[21]:


df=pd.DataFrame({'Actual':y_test,'predict':y_pred})


# In[22]:


df


# # Predict of hour=9.25

# In[23]:


# You can also test with your own data
hours = 9.25
own_pred = lr.predict([[hours]])
print("No of Hours = {}".format(hours))
print("Predicted Score = {}".format(own_pred[0]))


# In[24]:


from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
mae=mean_absolute_error(y_pred,y_test)
r2=r2_score(y_test,y_pred)
print('R-Squared = {}'.format(r2))
print('Mean absolute error = {}'.format(mae))


# In[ ]:




