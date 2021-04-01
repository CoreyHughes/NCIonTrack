#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st

# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import pickle
import sklearn
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
#import sklearn.linear_model as lm
from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib


# In[3]:


#header_pic = Image.open('Desktop\frontend\grad.jpg')
#st.image('./grad.png')

#read predictive model from python notebook
pickle_in = open('model.pkl', 'rb')
classifier = pickle.load(pickle_in)


# In[4]:


#tab
st.set_page_config(page_title="NCI on Track", page_icon=":sunglasses:")


# In[5]:


col1,col2 = st.beta_columns([0.75, 0.25])

st.markdown("""
<style>
body {
    color: #fff;
    background-color: #cc99ff;
}
</style>
    """, unsafe_allow_html=True)


# In[7]:


#niceities

st.title("NCI on Track :sparkles:")


loooong_text = ' '.join(["The idea of the desktop application is to create a dashboard which uses an algorithm to identify students who may be at risk of dropping out from univeristy. It is intended to be a tool in the college’s arsenal to allow them to keep student retention rates high during a period of remote learning which has shown to be difficult for students. Trying to build and form relationships with students online can be difficult and tying to gage who is at risk of dropping out can be near impossible in a remote environment. "])

#st.text(loooong_text)

st.markdown(loooong_text)
#st.text(" ")

st.title("Prediction :crystal_ball:")


# In[8]:


#read student data

student_data = pd.read_csv("data9.csv")




#split into two columns layout
selection_col, display_col = st.beta_columns(2)


# In[9]:


# Update sex column to numerical
student_data['sex'] = student_data['sex'].map(lambda x: 0 if x == 'female' else 1)

student_data['internet'] = student_data['internet'].map(lambda x: 1 if x == 1 else 0)

student_data['schoolsup'] = student_data['schoolsup'].map(lambda x: 1 if x == 1 else 0)

student_data['address'] = student_data['address'].map(lambda x: 1 if x == 1 else 0)

student_data['health'] = student_data['health'].map(lambda x: 1 if x == 1 else 0)

student_data['failures'] = student_data['failures'].map(lambda x: 1 if x == 1 else 0)

student_data['absences'] = student_data['absences'].map(lambda x: 1 if x == 1 else 0)
#  student_data['health'] = student_data['health'].map(lambda x: 1 if x == 1 else 0)
#  student_data['failures'] = student_data['failures'].map(lambda x: 1 if x == 1 else 0)
#  student_data['absences'] = student_data['absences'].map(lambda x: 1 if x == 1 else 0)

student_data= student_data[['sex' , 'internet' , 'schoolsup', 'address', 'health', 'failures', 'absences', 'drop']]   


# In[10]:


# student_data = manipulate_sd(student_data)
features= student_data[['sex' , 'internet' , 'schoolsup', 'address', 'health', 'failures', 'absences']]
dropout = student_data['drop']
X_train , X_test , y_train , y_test = train_test_split(features , dropout ,test_size = 0.3)

scaler = StandardScaler()
train_features = scaler.fit_transform(X_train)
test_features = scaler.transform(X_test)

model = LogisticRegression()
model.fit(train_features , y_train)
train_score = model.score(train_features,y_train)
test_score = model.score(test_features,y_test)
y_predict = model.predict(test_features)


# In[11]:


sex = st.sidebar.selectbox("Gender? ", options=["Female","Male"], index=0)
internet = st.sidebar.selectbox('Does this student have stable internet?', options=["Yes","No"], index=0)
schoolsup = st.sidebar.selectbox('Is this student recieving support from student services?', options=["Yes","No"], index=0)
address = st.sidebar.selectbox('Address?', options=["Rural","Urban"], index=0)

health = st.sidebar.selectbox("Has this student's health been affected this year?", options=["Yes", "No"], index=0)
failures = st.sidebar.selectbox("Has this student's failed more than 3 modules?", options=["Yes", "No"], index=0)
absences = st.sidebar.selectbox("Has this student been absent for more than 10 days?", options=["Yes", "No"], index=0)


# In[12]:


sex = 0 if sex == 'Female' else 1
# if they HAVE internet it is 0
internet = 0 if internet == 'No' else 1

# if they HAVE grinds it is 0
schoolsup = 0 if schoolsup == 'No' else 1

address = 0 if address == 'Urban' else 1

# if they HAVE internet it is 0
health = 0 if health == 'No' else 1

# if they HAVE failed it is 0
failures = 0 if failures == 'No' else 1

# if they HAVE been out it is 0
absences = 0 if absences == 'No' else 1


# In[13]:


input_data = scaler.transform([[sex , internet, schoolsup , address, health, failures, absences]])
prediction = model.predict(input_data)
predict_probability = model.predict_proba(input_data)


# In[14]:


#submit button
submit = st.button('Predict Student Risk')



if submit:
    prediction = classifier.predict([[sex, internet, schoolsup, address, health, failures, absences]])
    if prediction == 0:
        st.write("This student is *NOT* at risk of dropping out")
    else:
        st.write("This student is at risk of dropping out and may need some extra supports")
        
        
        

