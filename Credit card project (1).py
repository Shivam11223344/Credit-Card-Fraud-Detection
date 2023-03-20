#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing the dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[4]:


# loading the datatset to a pandas dataframe
credit_card_data = pd.read_csv('creditcard.csv')

#finding first and last 5 rows of dataset
credit_card_data.head()


# In[5]:


credit_card_data.tail()


# In[6]:


#dataset information
credit_card_data.info()


# In[7]:


#checking the number of missing values in each column
credit_card_data.isnull().sum()


# In[8]:


#distribution of legit transactions and fraudulent transactions
credit_card_data['Class'].value_counts()


# In[9]:


# This Dataset is highly unblanced 0 --> Normal Transaction 1 --> fraudulent transaction
# setting the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[10]:


print(legit.shape) 
print(fraud.shape)


# In[11]:


# statistical model of the data
legit.Amount.describe()


# In[12]:


fraud.Amount.describe()


# In[13]:


# compare the values of both the transaction
credit_card_data.groupby('Class').mean()


# In[14]:


#Under-Sampling
#Build a sample dataset containing similar distribution of normal transactions and Fraudulent Transactions
#Number of Fraudulent Transactions --> 492
legit_sample = legit.sample(n=492)


# In[15]:


#Concatenating two DataFrames
new_dataset = pd.concat([legit_sample,fraud],axis=0)
# axis=0 means niche jodega axis=1 means bagal m jod dega


# In[16]:


new_dataset.head()


# In[17]:


new_dataset.tail()


# In[18]:


new_dataset['Class'].value_counts()


# In[20]:


new_dataset.groupby('Class').mean()


# In[22]:


#splitting data into features and targets
X = new_dataset.drop(columns='Class', axis=1)
Y = new_dataset['Class']


# In[23]:


print(X)


# In[24]:


print(Y)


# In[25]:


#spli the data into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)


# In[26]:


print(X.shape, X_train.shape, X_test.shape)


# In[27]:


#model training
#logistic regression
model = LogisticRegression()


# In[28]:


#training the logistic regression model with training data
model.fit(X_train, Y_train)


# In[39]:


#model evalation
#accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[40]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[41]:


#accuracy on test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[42]:


print('accuracy on test data:', test_data_accuracy)


# In[ ]:




