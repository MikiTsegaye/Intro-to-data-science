
# coding: utf-8

# <img src="./images/Banner_NB.png">

# # Exploring Titanic
# 
# 

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('./data/titanic.csv', header=0, sep=',') 


# In[3]:


df.columns


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.head()


# In[7]:


df.info()


# In[8]:


df.describe(include='all')


# In[9]:


df.Cabin.head().isnull()


# In[10]:


df.Cabin.isnull().sum()


# In[11]:


df2 = df.iloc[:8].copy()
df2.shape


# In[12]:


df2


# In[13]:


df2.dropna(axis=0)


# In[14]:


df2.dropna(axis=1)


# In[15]:


df2.dropna(axis=0, how='all')


# In[16]:


df2.dropna(axis=0, thresh=11).shape


# In[17]:


df2.dropna(axis=1, thresh=5).shape


# In[18]:


df2_clean = df2.dropna(axis=0, thresh=11).copy()


# In[19]:


df2 = df2.dropna(axis=0, thresh=11)
# we can achieve the same with df2.dropna(axis=0, thresh=11, inplace = True)


# In[20]:


df2 = df.iloc[:8].copy()
df2.Age.isnull().sum()


# In[21]:


new_age = df2.Age.fillna(0)
new_age


# In[22]:


new_age = df2.Age.fillna(df2.Age.mean())
new_age


# In[23]:


df2.Age = df2.Age.fillna(df2.Age.mean())


# In[24]:


df2.fillna(df2.median(), inplace=True)
df2.info()


# In[25]:


df.Embarked.describe()


# In[26]:


df["Embarked"].fillna('S',inplace=True)
#df.Embarked = df.Embarked.fillna(df.Embarked.mode()[0])


# In[27]:


df2 = df.iloc[:8].copy()
df2


# In[28]:


df2.fillna(method = 'ffill')


# In[29]:


df2.fillna(method = 'bfill')

