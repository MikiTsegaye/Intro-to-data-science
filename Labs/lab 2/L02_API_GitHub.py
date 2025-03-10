
# coding: utf-8

# <img src="./images/Banner_NB.png">

# # Using API with authorization
# In this notebook, we will learn how to use the GitHub API to fetch and analyze the programming languages used in commits over the past year. We will collect data on a monthly basis and visualize the trends. This exercise will demonstrate how to make authenticated API requests, handle time-series data, and visualize the results in Python.
# 
# 
# When working with commercial APIs, such as the GitHub API, it's essential to understand why authorization is necessary and how to handle your credentials securely. Here are the key points:
# 
# ### Why Authorization is Necessary:
# 
# 1. **Security**: Authorization mechanisms, like OAuth tokens, help secure access to user data. They ensure that only authenticated users can access certain data or functionalities, protecting both the user's data and the API from unauthorized use.
# 
# 2. **Rate Limiting**: APIs often have rate limits to prevent abuse and overuse of resources. GitHub's rate limit for authenticated requests is typically 5000 requests per hour, while for unauthenticated requests, the limit is much lower, at around 60 requests per hour..
# 
# 3. **Compliance**: Using authorization ensures that you comply with the terms of service of the API provider, which often include privacy and usage guidelines.
# 
# ### Best Practices for Handling Authorization Data:
# 
# 1. **Do Not Hardcode Credentials**: Storing your API keys or tokens directly in your code (especially if the code is shared or stored publicly, like on Vocareum) is not secure. It exposes your credentials to anyone who views the code, potentially leading to unauthorized access and misuse.
# 
# 2. **Use a Separate Configuration File or Environment Variables**:
#    - **Configuration File**: Store your credentials in a separate file (e.g., `config.py` or `credentials.py`) and import them into your main code.
#    - **Environment Variables**: Another secure approach is to use environment variables to store your credentials. This method is particularly recommended for deployment or shared environments.
# 
# 3. **Example of Using a Separate Module**:
#    ```python
#    # Example: Importing credentials from a separate file
#    from my_credentials import My_TOKEN
#    ```
# 

# # Getting your GitHub token
# To find your GitHub token, you need to create one in your GitHub account settings. GitHub doesn't store tokens in a way that you can view them after creation, so if you've lost your token, you'll need to generate a new one. Here's how you can create a new Personal Access Token (PAT) on GitHub (you can also read GitHub's documentation [here](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-fine-grained-personal-access-token)):
# 
# 1. **Log in to GitHub**: Go to [GitHub](https://github.com/) and sign in with your account. if you don't have one, now is a good chance to create one.
# 
# 2. **Access Settings**: Click on your profile picture in the top right corner of the GitHub page, and then click on "Settings" from the dropdown menu.
# 
# 3. **Developer Settings**: On the settings page, scroll down to the bottom of the sidebar on the left and click on "Developer settings."
# 
# 4. **Personal Access Tokens**: In the Developer settings, select "Personal access tokens" from the sidebar.
# 
# 5. **Generate New Token**: Click on the “Generate new token” button.
# 
# 6. **Set Up Your Token**:
#    - **Note**: Give your token a descriptive name i.e. "intro2DS"
#    - **Expiration**: Choose 30 days.
#    - **Select Scopes**: Scopes control the access for the token. in this exercise, you will only need "**public_repo**" and "**repo:status**"
# 
# 7. **Generate Token**: After setting up your token, click the “Generate token” button at the bottom of the page.
# 
# 8. **Copy Your New Token**: Once the token is generated, make sure to copy it and store it in the file named "**github_credentials.py**" (we have created an empty file with this name here, just edit it, and add the variable with your token).
# This file should contain the following line
#    ```python
#   GITHUB_TOKEN = 'YOUR_TOKEN_HERE'
#    ```

# In[1]:


from github_credentials import GITHUB_TOKEN


# In[2]:


#other imports required for this notebook
import requests
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import time



# # Fetching Data from GitHub API
# ### Define Function to Fetch Commit Data
# We define a function to make authenticated requests to the GitHub API. This function fetches the number of commits for a given programming language within a specified date range.

# In[3]:


def get_commit_data(language, start_date, end_date):
    url = f'https://api.github.com/search/commits?q=language:{language}+committer-date:{start_date}..{end_date}'
    headers = {'Authorization': f'token {GITHUB_TOKEN}',
               'Accept': 'application/vnd.github.cloak-preview'}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()['total_count']
    else:
        return None


# ### Collecting Commit Data
# We will collect data for several popular programming languages over the last year on a monthly basis.

# In[4]:


languages = ["Python", "Java", "JavaScript", "PHP", "R"]
end_date = datetime.now().replace(day=1) - timedelta(days=1) #last day of previous month
start_date = end_date.replace(year=end_date.year - 1)
date_range = pd.date_range(start=start_date, end=end_date, freq='M')

data = []

for start,end in zip (date_range,date_range[1:]):
    month_start = start.strftime("%Y-%m-%d")
    month_end = end.strftime ("%Y-%m-%d")
    monthly_data = {'Start Date': month_start, 'End Date': month_end}
    for language in languages[:2]:
        commit_count = get_commit_data(language, month_start, month_end)
        monthly_data[language] = commit_count
        time.sleep(2) #play nicely
    data.append(monthly_data)

df = pd.DataFrame(data)
df.set_index('Start Date', inplace=True)


# In[5]:


df


# ### Visualizing the Trends
# We will use a line chart to visualize the number of commits per month for each programming language.

# In[6]:


df.plot(kind='line', figsize=(12, 6))
plt.title('Monthly Commit Counts by Programming Language')
plt.xlabel('Start Date')
plt.ylabel('Number of Commits')
plt.legend(title='Programming Languages')
plt.show()


# In[ ]:




