#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import libraries

import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
#import sklearn.linear_model as lm
from sklearn.model_selection import train_test_split


# In[2]:


sd = pd.read_csv('data9.csv')
print("Student data read successfully!")


# In[3]:


#print the first 7 student datasets
sd.head(7)


# In[4]:


#check what the values are
sd.columns.values


# In[5]:


#check for missing values 
sd.isna().sum()


# In[6]:


sd.describe()
#used to find unusual data outside of 2 or 3 standard diviations that may skew data


# In[7]:


#this code is used to check which attributes have more than 2 or 3 standarddivations 

data = sd
def plotAttribute(featureName):
  oneSTD = data[featureName].std()
  twoSTD = oneSTD * 2
  threeSTD = oneSTD * 3
  meanValue = data[featureName].mean()
  print("Attribue:", featureName.upper())
  print("Summary:")
  instances = data.shape[0]
  outsideTwo = ((data[featureName] < (meanValue - twoSTD)).sum() + (data[featureName] > (meanValue + twoSTD)).sum())
  outsideThree = ((data[featureName] < (meanValue - threeSTD)).sum() + (data[featureName] > (meanValue + threeSTD)).sum())
  print("N outside of two STD:\t", outsideTwo, "\t(", round((outsideTwo/instances) *100, 2),"%)")
  print("N outside of three STD:\t", outsideThree, "\t(", round((outsideThree/instances) * 100, 2),"%)")
  plt.axvline(x=(meanValue - oneSTD), label='One STD', c="g")
  plt.axvline(x=(meanValue + oneSTD), c="g")
  plt.axvline(x=(meanValue - twoSTD), label='Two STD', c="y")
  plt.axvline(x=(meanValue + twoSTD), c="y")
  plt.axvline(x=(meanValue - threeSTD), label='Three STD', c="r")
  plt.axvline(x=(meanValue + threeSTD), c="r")
  data[featureName].hist(figsize=(7,7))
  plt.legend()
  plt.show()
plotAttribute("studytime")


# In[8]:


#check the data for the amount of students who have dropped out
sd['drop'].value_counts()


# In[9]:


#place the above data in a chart
sns.countplot(sd['drop'])


# In[10]:


#create a graph that looks at drop out rates across the two sexes 
sns.countplot(x="sex", hue ="drop", data=sd)


# In[11]:


#check the failure rate against those who have dropped out
sns.countplot(x="failures", hue="drop", data=sd)


# In[ ]:





# In[12]:


#check the drop out rate against those who have steady internet 
sns.countplot(x="internet", hue="drop", data=sd)


# In[13]:


#convert all non numeric columns to a number, check this cell 
for col in sd.columns:
  if sd[col].dtype == np.number:
    continue
  sd[col] = LabelEncoder().fit_transform(sd[col])


# In[14]:


sd.dtypes


# In[15]:


sd.head(5)


# In[16]:


plt.figure(figsize=(20,10))
sns.heatmap(sd.corr(),annot=True)
plt.show()


# In[17]:


sd.drop('Medu', axis=1, inplace=True)
sd.drop('reason', axis=1, inplace=True)
sd.drop('traveltime', axis=1, inplace=True)
sd.drop('famsup', axis=1, inplace=True)
sd.drop('activities', axis=1, inplace=True)
sd.drop('romantic', axis=1, inplace=True)
sd.drop('freetime', axis=1, inplace=True)
sd.drop('Dalc', axis=1, inplace=True)
sd.drop('higher', axis=1, inplace=True)
sd.drop('Pstatus', axis=1, inplace=True)
sd.drop('Fedu', axis=1, inplace=True)
sd.drop('guardian', axis=1, inplace=True)
sd.drop('paid', axis=1, inplace=True)
sd.drop('famrel', axis=1, inplace=True)
sd.drop('goout', axis=1, inplace=True)
sd.drop('Fjob', axis=1, inplace=True)
sd.drop('Walc', axis=1, inplace=True)
sd.drop('nursery', axis=1, inplace=True)
sd.drop('Mjob', axis=1, inplace=True)
sd.drop('famsize', axis=1, inplace=True)
sd.drop('school', axis=1, inplace=True)


# In[18]:


sd.head(3)


# In[19]:


#scale the data
#X = sd.drop("drop", axis = 1) #feature data set
#X = sd[['sex', 'age', 'studytime', 'failures', 'schoolsup',
#       'internet', 'health', 'absences']]

#X = sd[['sex','schoolsup',
#       'internet', 'address']]
#y = sd["drop"]

#X = StandardScaler().fit_transform(X)


# In[20]:


#split the data into 80% training and 20% testing
x_train, x_test, y_train, y_test = train_test_split(sd[['sex', 'internet','schoolsup', 'address', 'health', 'failures', 'absences']], sd['drop'], test_size=0.2, random_state=0)


# In[ ]:





# In[21]:


#create the model
model = LogisticRegression(C=1)
#train the model
model.fit(x_train, y_train)

y_pred=model.predict(x_test)


# In[22]:


#create the predictions
from sklearn import *
predictions = model.predict(x_test)

#print the predictions
print(predictions)

#testData = np.array([[1,1,1,1,1,1,1,1,1]], dtype=float)

#testDataN = testData / testData.max(axis=0)

#prediction = model.predict(testDataN)
#print("Prediction", prediction[0])

#if predictions.any() == 1:
#  print("At Risk of Dropping Out")
#if predictions.any() == 0:
#  print("Retained")


# In[23]:


#check the precision, recall and f1-score
print(classification_report(y_test, predictions))


# In[24]:


scaler = StandardScaler()
train_features = scaler.fit_transform(x_train)
test_features = scaler.transform(x_test)


# In[25]:


model = LogisticRegression(C=1)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)


# In[26]:


import pickle
pickle_out = open("model.pkl", "wb") 
pickle.dump(model, pickle_out) 
pickle_out.close()


# In[ ]:




