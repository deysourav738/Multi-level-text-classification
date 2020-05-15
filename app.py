#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask,request,render_template,url_for
import numpy as np
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re


# In[2]:


def feature_processing(ini_features):
    ps=PorterStemmer()
    temp=re.sub('[^a-zA-Z]',' ',ini_features)
    temp=temp.lower()
    temp=temp.split()
    temp=[ps.stem(word) for word in temp if not word in stopwords.words('english')]
    return set(temp)


# In[3]:


def tfid_get_feature_values(y,tf_scores):
    y_feat=[]
    for i in tf_scores:
        if i in y:
            y_feat.append(tf_scores[i])
        else:
            y_feat.append(0)
    return y_feat


# In[4]:


df=pd.read_excel("Desktop/2749936_mongodb_data.xls")


# In[5]:


df=df.dropna(axis=0)


# In[6]:


df=df.reset_index()


# In[7]:


app = Flask(__name__)


# In[8]:


model=pickle.load(open('model.pkl','rb'))
tf_scores=pickle.load(open('tf_scores.pkl','rb'))


# In[9]:


classes=['Audio Video Technology', 'Computer Technology','Electrical Machinery Apparatus Energy', 'Telecommunication']


# In[10]:


@app.route('/')
def home():
    return render_template('index.html')


# In[11]:


@app.route('/predict',methods=['POST'])
def predict():
    pun=request.form.values()
    features=list(df.loc[df['Publication number'] == pun]['Description'])[0]
    fin_features=feature_processing(features)
    fin_features=[tfid_get_feature_values(fin_features,tf_scores)]
    prediction=model.predict(fin_features)
    output=classes[prediction[0]]
    return render_template('index.html',prediction_text='This application belongs to {}'.format(output))


# In[13]:


if __name__ == "__main__":
    app.run(debug = False)


# In[ ]:




