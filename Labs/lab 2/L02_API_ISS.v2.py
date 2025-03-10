
# coding: utf-8

# <img src="./images/Banner_NB.png">

# # Lesson 2: Web Based API
# 
# ### Part A - Simple API Call (no parameters) - After Video 2.6

# We will start with a very simple API example, retrieving the current position of the ISS. We will be using the [requests library](http://www.python-requests.org/en/latest/), which will handle our communication with the server.

# In[1]:


import requests
# Make a get request to get the latest position of the international space station from the opennotify api.
response = requests.get("http://api.open-notify.org/iss-now.json")

response


# That looks good, we've received a response and it has status 200 - which means all was ok.
# 
# Let's see what happens when we try to get a response with a wrong URL:

# In[2]:


response_try2 = requests.get("http://api.open-notify.org/iss")

response_try2


# As we saw in the lecture, response 400 is response of an error (due to the wrong url we sent)

# Let's look at the content of our previous, successful response:

# In[3]:


response.content


# We can already see that this is JSON (though it is stored a `bytes` object), but we can check formally:

# In[4]:


response.headers['content-type']


# In[5]:


response.headers


# We can decode this byte object, then the JSON will be readable.

# In[6]:


response_j = response.content.decode("utf-8")
print(response_j)


# In[7]:


import json
response_d = json.loads(response_j)
print(type(response_d))
print(response_d)
response_d["iss_position"]


# Let's take a look at the JSON here:
# 
# 
# 

# In[8]:


response_d


# This looks a lot like a dictionary! We have key-value pairs.
# 
# We can use the [json library](https://docs.python.org/3/library/json.html) to convert JSON into objects:

# Or, not surprisingly, pandas can also load a json object:

# In[9]:


import pandas as pd

df = pd.read_json(response_j)
df


# Look at the Dataframe we got, this isn't quite what we want - we probably want one row per timestamp and longitude and latitude as columns. For that we will introduce a helper function called `flatten`

# In[10]:


response_d["latitude"] = response_d["iss_position"]["latitude"]
response_d["longitude"] = response_d["iss_position"]["longitude"]
response_d


# In[11]:


def flatten(response_d):
    response_d["latitude"] = response_d["iss_position"]["latitude"]
    response_d["longitude"] = response_d["iss_position"]["longitude"]
    del(response_d["iss_position"])
    return response_d
flatten(response_d)



# That looks better. Let's get a couple of positions of the ISS over time and save it as an array:

# In[12]:


import time

def pull_position():
    """Retreives the position of the ISS and returns it as a flat dictionary"""
    response = requests.get("http://api.open-notify.org/iss-now.json")
    response_j = response.content.decode("utf-8")
    response_d = json.loads(response_j)
    flat_response = flatten(response_d)
    return flat_response

iss_position = []

# calls pull_position 10 times with 3 seconds break
for i in range(10):
    flat_response = pull_position()
    iss_position.append(flat_response)
    print(flat_response)
    time.sleep(3)

len(iss_position)


# Now we can convert this into a nice dataframe:

# In[13]:


import pandas as pd

iss_position_df = pd.DataFrame(iss_position)
iss_position_df['timestamp']  = pd.to_datetime(iss_position_df['timestamp'], unit="s")

iss_position_df = iss_position_df.set_index(pd.DatetimeIndex(iss_position_df['timestamp']))
iss_position_df["latitude"] = iss_position_df["latitude"].map(float)
iss_position_df["longitude"] = iss_position_df["longitude"].map(float)
iss_position_df


# Let's see how the ISS moves

# In[14]:


import matplotlib.pyplot as plt
# This next line tells jupyter to render the images inline
get_ipython().magic('matplotlib inline')
plt.style.use('ggplot')
iss_position_df.plot(kind="scatter", x="latitude", y="longitude")
plt.show()


# Now it's time to go and see video 2.7, continue to part B - once you completed that video..

# ### Part B - API Call with parameters - After Video 2.7

# In the original clip you see the 2nd ISS example, however since ISS changed their API this example no longer works, and we will show you an example with GitHub's api (https://www.github.com)
# 

# Requests can be parametrized. You can search for the number of GitHub's repo based on their language. The repo's language will be passed to the API as a parameter. For example, [retrieve the number of python repo's in GitHub](https://api.github.com/search/repositories?q=language:python)!
# 
# The way to query GitHub with a get request for the python popularity is this:
# 
# `https://api.github.com/search/repositories`
# 
# We, of course, could generate that URL ourselves, but the requests library is helpful here. Since JSON is similar to dictionaries, the requests library takes dictionaries for parameters.

# In[15]:


import requests
url = "https://api.github.com/search/repositories"
lang_param = {"q": "language:python"}

r = requests.get(url, params=lang_param)
data = r.json()
print(data)
print(type(r))
print(type(data))


# as you can see, the json object has quite a bit of information. For our purposes, at this point, we need only the total number of repos, which can be found in the `total_count` attribute.
# 
# Let's compare now, the total number of Python repo's to the total of Java repo's. Who do you think is more popular?

# In[16]:



url = "https://api.github.com/search/repositories"
python_lang_param = {"q": "language:python"}
java_lang_param = {"q": "language:java"}

r = requests.get(url, params=python_lang_param)
data = r.json()
python_count=data["total_count"]

r = requests.get(url, params=java_lang_param)
data = r.json()
java_count=data["total_count"]

print(f"there is a total of {java_count} Java repos, and {python_count} python repos")


# So, in total there are more Java repos..

# Let's see also the other method we showed in the video to send the params

# In[20]:


url = "https://api.github.com/search/repositories"
python_lang_query_string = "q=language:python"
java_lang_query_string = "q=language:java"

r = requests.get(f"{url}?{python_lang_query_string}") # see the difference here, rather than passing a parameter to the function - we send a formatted string with the params in it
data = r.json()
python_count=data["total_count"]

r = requests.get(f"{url}?{java_lang_query_string}") # same here
data = r.json()
java_count=data["total_count"]

print(f"there is a total of {java_count} Java repos, and {python_count} python repos")


# ## In class Exercise
# 
# Write a code in the cell below that will compare the language popularity between javascript and python

# In[22]:


print (max(python_count,java_count))


# In[ ]:




