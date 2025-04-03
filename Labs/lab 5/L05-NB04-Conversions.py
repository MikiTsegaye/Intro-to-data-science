
# coding: utf-8

# <img src="./images/Banner_NB.png">

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('data/titanic.csv', header=0, sep=',') 


# In[3]:


df["Sex"]= df["Sex"].astype('category')


# In[4]:


df.info()


# In[5]:


replace_map = {'female':1,'male':2}


# In[6]:


df.replace(replace_map, inplace=True)


# In[7]:


df.head()


# In[8]:


bins = [0,10,20,30,40,50,60,70,80,120]
labels = [1,2,3,4,5,6,7,8,9]


# In[9]:


df['Age_binned'] = pd.cut(df['Age'], bins, labels=labels)


# In[10]:


df.Age_binned.describe()

