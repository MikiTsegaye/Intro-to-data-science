
# coding: utf-8

# <img src="images/Banner_NB.png">

# # Lesson 2: JSON

# JSON is a very popular format to transfar data object. The name JSON stands for JavaScript Object Notation, and is a pretty convenient format, as we'll see. 
#  
# 
# Let's look on the following JSON object with data about our course:

# In[ ]:


data={
        'course name' : 'Intro to data science',
        'number of students' : 53,
        'grades average' : 87.23,
        "started": True,
        "most favorite lectures": ["statistics", "crawling","last one"],
        "instructors":{
            "name": "Prof. Israeli",
            "affiliation": "HIT",
            "id": 1234
        }
    }



# ## Reading and Writing JSON Data
# The `json` module provides an easy way to encode and decode data in JSON. The two main functions are `json.dumps()` and `json.loads()`, mirroring the interface used in other serialization libraries. 
# 
# First let's import the `json` module

# In[ ]:


import json


# Here is how you turn a Python data structure into JSON

# In[ ]:


json_str = json.dumps(data)
print(json_str)


# Here is how you turn a JSON-encoded string back into a Python data structure:
# 

# In[ ]:


print("json string:")
print(json_str)
print()
print()
data = json.loads(json_str)
print("data['course name'] is:",data['course name'])
print("half number of students (data['number of students']) is:",
      data['number of students']*.5)
print("course is given in (data['instructors']['affiliation']):",
      data['instructors']["affiliation"])
print()
print()
print(data)
print(type(json_str))
print(type(data))


# In the example below, you can see how we write our data structure to a json file

# In[ ]:


# Writing JSON data
with open('data.json', 'w') as f:
     json.dump(data, f)



# and here is how we read it from a file..

# In[ ]:


# Reading data back
with open('data.json', 'r') as f:
     data = json.load(f)

# let's see that we can print one of the values and that it works..

print(data['grades average'])

