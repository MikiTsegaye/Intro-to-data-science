
# coding: utf-8

# <img src="./images/Banner_NB.png">

# # Exploring Titanic
# 
# 

# In[1]:


import pandas as pd


# In[2]:


df = pd.read_csv('data/titanic.csv', header=0, sep=',') 


# In[3]:


df2 = pd.read_csv('data/titanic_small.csv', header=0, sep=',') 


# In[4]:


df2


# In[5]:


df2.duplicated()


# In[6]:


df2.duplicated().sum()


# In[7]:


df2[df2.duplicated()]


# In[8]:


df[df.duplicated(['Name'])]


# In[9]:


df[df.duplicated(['Name'])].shape


# In[10]:


df2[df2.duplicated(['Name'])]


# In[11]:


df.Embarked.duplicated().sum()


# In[12]:


df.Embarked.unique()


# In[13]:


df2.drop_duplicates()


# In[14]:


df2 = pd.read_csv('data/titanic_small.csv', header=0, sep=',') 


# In[15]:


df2.drop_duplicates(subset=['Sex'])
# We can also use keep='last' if we want to save the last copy of each duplication group 


# In[36]:


df.duplicated(subset=['Ticket']).sum()
df.Ticket.unique().shape


# In[ ]:




