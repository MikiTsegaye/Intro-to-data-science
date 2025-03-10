
# coding: utf-8

# <img src="./images/Banner_QA.png">

# # Intermediate Exercise - Lesson 2
# # Data Acquisition - API
# ## (not graded)

# ## Exercise 1

# In this exercise you are asked to find your computer's IP address. In order to get it, make an API call to this service: `https://api.ipify.org?format=json` and get your current IP address. Print just the IP address itself.
# 

# In[1]:


#your code here
import requests
ip = requests.get("https://api.ipify.org?format=json").content.decode("utf-8")
print(ip)


# ## Exercise 2

# In this exercise we want to figure out historical exchange rates between EURO and other currencies (such as USD (US Dollar), GBP (Great Britain Pound) etc.) <p>
# In order to do it, you will consume the OpenDataSoft API (see [here](https://public.opendatasoft.com/explore/dataset/euro-exchange-rates/information/) for full API documentation) <p>
# Take a look on the API in order to understand its structure. In general, a sample calls looks like the following: `https://public.opendatasoft.com/api/records/1.0/search/?dataset=euro-exchange-rates&sort=date&facet=currency&rows=30&facet=date&q=date:[2021-01-12+TO+2021-01-15]&refine.currency=GBP `
# 
# The call is divided into base url, and parameters. They have the following format:
# + Base url link: (note that we have included some parameters also in the base url - don't change them at this point..) `https://public.opendatasoft.com/api/records/1.0/search/?dataset=euro-exchange-rates&sort=date&facet=currency&rows=30&facet=date` 
# - Date parameter: given by parameter `date` and its format is `date:[2021-01-12+TO+2021-01-15]` - which provides that data range in which we want to get the data for..
# - Currency for conversion - which of the currencies do we want to convert it to. Possible values are: `USD`, `GBP`
#     
# Use the API to consume that conversion rates between USD and EUR from Dec 1st, 2020 to Dec 31st 2020 and present the results in a DataFrame
# 
# 

# In[2]:


# your code here


# In[14]:


baseURL="https://public.opendatasoft.com/api/records/1.0/search/?dataset=euro-exchange-rates&sort=date&facet=currency&rows=30&facet=date"
fromDate="2021-01-12"
toDate="2021-01-15"
dateP= f"q=date:[{fromDate}+TO+{toDate}]"
currP= "refine.currency"
currV= "USD"
url = baseURL + "&" + dateP +'&'+ currP + currV
url
x=requests.get(url)
x.json()


# In[ ]:


import pandas as pd


# In[16]:


rates=[]
dates=[]
currencies=[]

resDict=x.json()
for rec in resDict['records']:
    print(rec['fields'])


# In[ ]:





# In[ ]:





# In[ ]:




