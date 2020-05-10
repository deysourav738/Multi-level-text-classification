#!/usr/bin/env python
# coding: utf-8

# # Multiclass-Text-Classification
#     Author : Sourav Dey

# In[1]:


# Importing all the required module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
import re
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier


# In[2]:


data=pd.read_excel("Desktop/2749933_Entities.xlsx")


# In[3]:


data.head()


# In[4]:


data.shape


# In[5]:


columns=list(data.columns)
columns


# In[6]:


# Reshaping the data set for implementation
new_data=[]
for j in columns:
    for i in data[j]:
        new_data.append([i,j])
new_data


# In[7]:


# x are the features and y are target variables 
new_data=pd.DataFrame(new_data,columns=['x','y'])
new_data


# In[8]:


new_data.isnull().sum()


# In[9]:


new_data=new_data.dropna() # Removing Null values


# In[10]:


new_data.groupby('y').count()


# In[11]:


lb = LabelEncoder()
new_data['y']=lb.fit_transform(new_data['y'])


# In[12]:


classes=lb.classes_
classes


# In[13]:


new_data=new_data.reset_index()


# In[14]:


x=new_data['x']
y=new_data['y']


# In[15]:


ps=PorterStemmer()
temp1=[]
for i in range(0,len(x)):
    temp2=re.sub('[^a-zA-Z]',' ',x[i])
    temp2=temp2.lower()
    temp2=temp2.split()
    temp2=[ps.stem(word) for word in temp2]
    temp2=' '.join(temp2)
    temp1.append(temp2)


# In[16]:


cv=CountVectorizer(min_df=1)
x=cv.fit_transform(temp1)


# In[17]:


x=x.toarray()


# In[18]:


x


# In[19]:


ft=cv.get_feature_names()


# In[20]:


ft


# In[21]:


x.shape,y.shape


# In[22]:


nn=MLPClassifier(hidden_layer_sizes=200,random_state=40)
print("Accuracy on training set:")
res=cross_val_score(nn,x,y,cv=10,scoring="accuracy")
print("Average accuracy:\t{0:}\n".format(np.mean(res)))
print("Standard Deviation:\t{0:}\n".format(np.std(res)))
nn.fit(x,y)


# In[23]:


pickle.dump(nn,open('model.pkl','wb'))

