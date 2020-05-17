#!/usr/bin/env python
# coding: utf-8

# # Multiclass-Text-Classification
#     Author : Sourav Dey

# In[155]:


# Importing all the required module
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.preprocessing import LabelEncoder
from nltk.stem.porter import PorterStemmer
import re
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from imblearn.combine import SMOTETomek
from sklearn.linear_model import LogisticRegression


# In[162]:


def feature_processing(ini_features):
    temp=re.sub('[^a-zA-Z]',' ',ini_features)
    temp=temp.lower()
    temp=temp.split()
    temp=[ps.stem(word) for word in temp if not word in stopwords.words('english')]
    return set(temp)


# In[3]:


def tfidf_get_scores(transformed_matrix,feature_names):
    tf_scores={}
    for i in range(transformed_matrix.shape[0]):
        ft_index=transformed_matrix[i,:].nonzero()[1]
        for j in ft_index:
            if transformed_matrix[i,j] != 0:
                tf_scores[feature_names[j]] = transformed_matrix[i,j]
    for i in feature_names:
        tf_scores[i]=tf_scores.pop(i)
    return tf_scores


# In[62]:


def tfid_get_feature_values(y,tf_scores):
    y_feat=[]
    for i in tf_scores:
        if i in y:
            y_feat.append(tf_scores[i])
        else:
            y_feat.append(0)
    return y_feat


# In[6]:


data=pd.read_excel("2749933_Entities.xlsx")


# In[7]:


data.head()


# In[8]:


data.shape


# In[9]:


columns=list(data.columns)
columns


# In[10]:


# Reshaping the data set for implementation
new_data=[]
for j in columns:
    for i in data[j]:
        new_data.append([i,j])
new_data


# In[11]:


# x are the features and y are target variables 
new_data=pd.DataFrame(new_data,columns=['x','y'])
new_data


# In[12]:


new_data.isnull().sum()


# In[13]:


new_data=new_data.dropna() # Removing Null values


# In[14]:


new_data.groupby('y').count()


# In[15]:


smk = SMOTETomek(random_state=42)


# In[16]:


lb = LabelEncoder()
new_data['y']=lb.fit_transform(new_data['y'])


# In[17]:


classes=lb.classes_
classes


# In[18]:


new_data=new_data.reset_index()


# In[130]:


x=new_data['x']
y=new_data['y']


# In[131]:


ps=PorterStemmer()
temp1=[]
for i in range(0,len(x)):
    temp2=re.sub('[^a-zA-Z]',' ',x[i])
    temp2=temp2.lower()
    temp2=temp2.split()
    temp2=[ps.stem(word) for word in temp2]
    temp2=' '.join(temp2)
    temp1.append(temp2)


# In[132]:


tf=TfidfVectorizer(max_features=400)
x=tf.fit_transform(temp1)


# In[133]:


x


# In[134]:


ft=tf.get_feature_names()


# In[135]:


tf_scores=tfidf_get_scores(x,ft)


# In[136]:


len(tf_scores)


# In[137]:


x.shape,y.shape


# In[138]:


x,y


# In[139]:


x=x.toarray()


# In[140]:


x,y=smk.fit_sample(x,y)


# In[141]:


x.shape,y.shape


# In[142]:


reg=LogisticRegression()
print("Accuracy on training set:")
res=cross_val_score(reg,x,y,cv=10,scoring="accuracy")
print("Average accuracy:\t{0:}\n".format(np.mean(res)))
print("Standard Deviation:\t{0:}\n".format(np.std(res)))
reg.fit(x,y)


# In[163]:


pickle.dump(reg,open('model.pkl','wb'))


# In[164]:


pickle.dump(tf_scores,open('tf_scores.pkl','wb'))


# In[ ]:





# In[ ]:





# In[ ]:





# In[91]:


df=pd.read_excel("2749936_mongodb_data.xls")


# In[92]:


df=df.dropna(axis=0)


# In[93]:


df=df.reset_index()


# In[94]:


ini=[df['Description'][i] for i in range(0,len(df))]


# In[95]:


fin=[feature_processing(ini[i]) for i in range(len(ini))]


# In[143]:


final=[tfid_get_feature_values(fin[i],tf_scores) for i in range(len(fin))]


# In[144]:


prediction=reg.predict(final)
output=classes[prediction]


# In[145]:


prediction


# In[146]:


output


# In[147]:


unique,count=np.unique(output,return_counts=True)


# In[152]:


unique,count


# In[154]:


plt.bar(unique,count,color=['r','g','b','y'])
plt.xticks(unique,unique,rotation=90)


# In[ ]:

