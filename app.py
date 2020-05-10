#!/usr/bin/env python
# coding: utf-8

# In[1]:


from flask import Flask,request,render_template,url_for
import numpy as np
import pandas as pd
import pickle


# In[2]:


df=pd.read_excel("Desktop/2749936_mongodb_data.xls")


# In[3]:


df=df.dropna(axis=0)


# In[4]:


def final_feature(ini_features):
    temp=re.sub('[^a-zA-Z]',' ',ini_features)
    temp=temp.lower()
    temp=temp.split()
    temp=[ps.stem(word) for word in temp if not word in stopwords.words('english')]
    temp=set(temp)
    final_features=[]
    for i in ft:
        if i in temp:
            val = 1
        else:
            val = 0
        final_features.append(val)
    return list(final_features)


# In[5]:


app = Flask(__name__)


# In[6]:


model=pickle.load(open('model.pkl','rb'))


# In[7]:


@app.route('/')
def home():
    return render_template('index.html')


# In[8]:


@app.route('/predict',methods=['POST'])
def predict():
    pun=request.form.values()
    features=list(df.loc[df['Publication number'] == pun]['Description'])[0]
    prediction =final_feature(features)
    output=classes[prediction]
    return render_template('index.html',prediction_text='This application belongs to {}'.format(output))


# In[9]:


if __name__ == "__main__":
    app.run(debug = True)


# In[10]:


df


# In[ ]:




