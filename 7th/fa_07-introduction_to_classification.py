
# coding: utf-8

# ![Final Lesson Exercise](images/Banner_FEX.png)

# # Lesson #7: Introduction to classification
# ## Good Movies - The IMDb movie dataset

# ## About this assignment
# In this assignment, you will continue to explore information regarding good movies.<br/>
# 
# This time you will practice a basic classification flow and perform the following steps:
# * Load the dataset
# * Split the dataset to train and test
# * Scale the data
# * Train a classification model
# * Predict new examples

# ## Preceding Step - import modules (packages)
# This step is necessary in order to use external packages. 
# 
# **Use the following libraries for the assignment, when needed**:

# In[1]:


# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# === CELL TYPE: IMPORTS AND SETUP 

import os                           # for testing use only
import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing, linear_model, model_selection
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# ## 1. Load the IMDb movie dataset
# In this section you will load the IMDB movie dataset and split the dataset to feature vectors (X) and labels (y).

# ### 1. Instructions
# <u>method name</u>: <b>load_dataset</b>
# <pre>The following is expected:
# --- Complete the 'load_dataset' function to  to load a CSV file from 'file_name' path,
#     remove the 'label_column' column, and return the rest of the dataframe as dataframe 'X' and
#     the removed 'label_column' column as series 'y'
# Notes: 
# * The 'X' dataframe should not include the 'label_column' column.
# * The return statement should look similar to the following statement:
# return X, y
# </pre>

# In[2]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def load_dataset(file_name, label_column):
    df=pd.read_csv(file_name)
    y=df[label_column]
    x=df.drop(label_column,axis=1)
    return x,y


# In[3]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
X, y = load_dataset(file_name, category_col_name)
print(X,y)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[4]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[5]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1. - Test 1 (name: test1-1_load_dataset, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'load_dataset' ...")
file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
print ("Good Job!\nYou've passed the 1st test for the 'load_dataset' function implementation :-)")


# In[6]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1. - Test 2 (name: test1-2_load_dataset, points: 0.9)")
print ("\t--->Testing the implementation of 'load_dataset' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
assert X.shape == (1985, 4), 'Wrong shape for feature vector dataframe'
assert y.shape[0] == 1985, 'Wrong number of lables in series'

print ("Good Job!\nYou've passed the 2nd test for the 'load_dataset' function implementation :-)")
X.head()


# ## 2. Split dataset to train and test
# In this section you will split the dataset into a train set and a test set.

# ### 2. Instructions
# <u>method name</u>: <b>split_to_train_and_test</b>
# <pre>The following is expected:
# --- Complete the 'split_to_train_and_test' function to split the dataset (already divided to X & y)
#     to a train set and test set.
# 
# You should split the 'X' dataframe into a 'X_train', 'X_test', where the ratio of the test out of 'X' is 'test_ratio'.
# The 'y' series should be splitted in a corresponding way into 'y_train' and 'y_test'.
# 
# Notes: 
# * Use sklearn's 'train_test_split' method, which was taught in class (and is imported above)
# * Use the 'rand_state' as the value for the 'random_state' parameter in 'train_test_split'.
# 
# The return statement should look similar to the following statement:
# return X_train, X_test, y_train, y_test
# </pre>

# In[7]:


# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def split_to_train_and_test(X, y, test_ratio, rand_state):
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=test_ratio,random_state=rand_state)
    return X_train,X_test,y_train,y_test


# In[8]:


# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
X, y = load_dataset(file_name, category_col_name)
X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
print(X_train, X_test, y_train, y_test)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[9]:


# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[10]:


# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2. - Test 1 (name: test2-1_split_to_train_and_test, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'split_to_train_and_test' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

print ("Good Job!\nYou've passed the 1st test for the 'split_to_train_and_test' function implementation :-)")


# In[11]:


# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2. - Test 2 (name: test2-2_split_to_train_and_test, points: 0.4)")
print ("\t--->Testing the implementation of 'split_to_train_and_test' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert X_train.shape == (1389, 4), 'Wrong shape for feature vector train dataframe'
assert X_test.shape == (596, 4), 'Wrong shape for feature vector test dataframe'
assert y_train.shape[0] == 1389, 'Wrong number of lables in train series'
assert y_test.shape[0] == 596, 'Wrong number of lables in  test series'

print ("Good Job!\nYou've passed the 2nd test for the 'split_to_train_and_test' function implementation :-)")


# In[12]:


# 2.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST  

print ("Part 2. - Test 2 (name: test2-3_split_to_train_and_test, points: 0.5)")
print ("\t--->Testing the implementation of 'split_to_train_and_test' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise

assert list(y_train.value_counts().values) == [785, 604], 'Wrong count of train label values'
assert list(y_test.value_counts().values) == [337, 259], 'Wrong count of test label values'

print ("Good Job!\nYou've passed the 3rd test for the 'split_to_train_and_test' function implementation :-)")


# ## 3. Scale features
# In this section you will scale your dataset, using two possible scaling techniques:
# 1. Minmax scaler (using sklearn's MinMaxScaler)
# + Standard scaler (using sklearn's StandardScaler)
# 
# You will perform a couple of tasks including:
# * Creating the scaler
# * Scale the train set
# * Scale the test set

# ### 3.a. Instructions
# <u>method name</u>: <b>scale_features</b>
# <pre>The following is expected:
# --- Complete the 'scale_features' function to create a scaler (Minmax scaler or Standard scaler) and 
#     scale the given 'X_train' train set with the scaler. 
# 
#      If the value of 'scale_type' equals 'minmax', you should create a Minmax scaler object (using sklearn's MinMaxScaler).
#      If the value of 'scale_type' equals 'standard', you should create a Standard scaler object (using sklearn's StandardScaler).
#          Note: you could assume that the input values for 'scale_type' are either 'minmax' or 'standard'.
#      
#      Note: If 'scale_type' equals 'minmax'scale with 'feature_range=(0, 1)'
# 
# Return the scaler object and the scaled 'X_train' train set.
# 
# Note: the return statement should look similar to the following statement:
# return scaler, X_train_scaled
# </pre>

# In[15]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def scale_features(X_train, scale_type):
    if scale_type =='minmax':
        scaler=MinMaxScaler(feature_range=(0,1))
        X_train_scaled = scaler.fit_transform(X_train)
    elif scale_type=='standard':
        scaler=StandardScaler()
        X_train_scaled=scaler.fit_transform(X_train)
    
    return scaler,X_train_scaled


# In[16]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
X, y = load_dataset(file_name, category_col_name)
X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
print(standard_scaler, X_train_standard_scaled)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[17]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[18]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3a. - Test 1 (name: test3a-1_scale_features, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'scale_features' ...")
file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
print ("Good Job!\nYou've passed the 1st test for the 'scale_features' function implementation :-)")


# In[19]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3a. - Test 2 (name: test3a-2_scale_features, points: 0.1)")
print ("\t--->Testing the implementation of 'scale_features' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
assert X_train_minmax_scaled.shape == (1389, 4), 'Wrong shape for feature vectors in scaled train set'
assert X_train_standard_scaled.shape == (1389, 4), 'Wrong shape for feature vectors in scaled train set'

print ("Good Job!\nYou've passed the 2nd test for the 'scale_features' function implementation :-)")


# In[20]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3a. - Test 3 (name: test3a-3_scale_features, points: 0.4)")
print ("\t--->Testing the implementation of 'scale_features' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert standard_scaler is not None, 'Scaler should not be None'
assert list(np.around(X_train_standard_scaled.mean(axis=0)+0.001, decimals=2)) == [0., 0., 0., 0.], 'Wrong standardized scaled train values'

print ("Good Job!\nYou've passed the 3rd test for the 'scale_features' function implementation :-)")


# In[21]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3a. - Test 4 (name: test3a-4_scale_features, points: 0.4)")
print ("\t--->Testing the implementation of 'scale_features' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert minmax_scaler is not None, 'Scaler should not be None'
assert list(np.around(X_train_minmax_scaled.min(axis=0), decimals=2)) == [0., 0., 0., 0.], 'Wrong minmax scaled train values'

print ("Good Job!\nYou've passed the 4th test for the 'scale_features' function implementation :-)")


# ### 3.b. Instructions
# <u>method name</u>: <b>scale_test_features</b>
# <pre>The following is expected:
# --- Complete the 'scale_test_features' function to scale the given 'X_test' test set,
#      using the given 'scaler'.
# Note: inputs will be 'minmax_scaler' or 'standard_scaler'
# You need to return the scaled test set dataframe.
# </pre>

# In[22]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def scale_test_features(X_test, scaler):
    X_test_scaled = scaler.transform(X_test)
    return X_test_scaled


# In[23]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
X, y = load_dataset(file_name, category_col_name)
X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
X_test_minmax_scaled = scale_test_features(X_test, minmax_scaler)
standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
X_test_standard_scaled = scale_test_features(X_test, standard_scaler)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[24]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[25]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3b. - Test 1 (name: test3b-1_scale_test_features, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'scale_test_features' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    X_test_minmax_scaled = scale_test_features(X_test, minmax_scaler)
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
    X_test_standard_scaled = scale_test_features(X_test, standard_scaler)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

print ("Good Job!\nYou've passed the 1st test for the 'scale_test_features' function implementation :-)")


# In[26]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3b. - Test 2 (name: test3b-2_scale_test_features, points: 0.1)")
print ("\t--->Testing the implementation of 'scale_test_features' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    X_test_minmax_scaled = scale_test_features(X_test, minmax_scaler)
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
    X_test_standard_scaled = scale_test_features(X_test, standard_scaler)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
assert X_test_minmax_scaled.shape == (596, 4), 'Wrong  shape for feature vectors in scaled test set'
assert X_test_standard_scaled.shape == (596, 4), 'Wrong  shape for feature vectors in scaled test set'

print ("Good Job!\nYou've passed the 2nd test for the 'scale_test_features' function implementation :-)")


# In[27]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3b. - Test 3 (name: test3b-3_scale_test_features, points: 0.4)")
print ("\t--->Testing the implementation of 'scale_test_features' ...") 

print ("Test 2 - Testing the implementation of the 'scale_features' and the 'scale_test_features' methods ...\n")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    X_test_minmax_scaled = scale_test_features(X_test, minmax_scaler)
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
    X_test_standard_scaled = scale_test_features(X_test, standard_scaler)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
assert list(np.around(X_test_standard_scaled.max(axis=0), decimals=2)) == [1.61, 3.12, 3.3, 2.97], 'Wrong scaled test values'

print ("Good Job!\nYou've passed the 3rd test for the 'scale_test_features' function implementation :-)")


# In[28]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3b. - Test 4 (name: test3b-4_scale_test_features, points: 0.4)")
print ("\t--->Testing the implementation of 'scale_test_features' ...")

print ("Test 3 - Testing the implementation of the 'scale_features' and the 'scale_test_features' methods ...\n")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 41)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    X_test_minmax_scaled = scale_test_features(X_test, minmax_scaler)
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
    X_test_standard_scaled = scale_test_features(X_test, standard_scaler)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
assert list(np.around(X_test_minmax_scaled.mean(axis=0)+0.001, decimals=2)) == [0.7, 0.27, 0.26, 0.19], 'Wrong scaled test values'

print ("Good Job!\nYou've passed the 4th test for the 'scale_test_features' function implementation :-)")


# ### 4. Train a classification model
# In this section you will train a classification model on your train set.<br/>
# You will build the classification model, using sklearn's LogisticRegression.

# ### 4. Instructions
# <u>method name</u>: <b>train_classifier</b>
# <pre>The following is expected:
# Complete the 'train_classifier' function to train a logistic regression classification model
# Use given 'X_train' dataframe as features and the corresponding 'y_train' as labels.
# Return the classifier: Sklearn's LogisticRegression().fit()
# 
# </pre>

# In[29]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def train_classifier(X_train, y_train):
    class_model = LogisticRegression().fit(X_train,y_train)
    return class_model


# In[30]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
X, y = load_dataset(file_name, category_col_name)
X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 11)
minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
X_test_minmax_scaled = scale_test_features(X_test, minmax_scaler)
standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
X_test_standard_scaled = scale_test_features(X_test, standard_scaler)
classification_model = train_classifier(X_train, y_train)
classification_minmax_model = train_classifier(X_train_minmax_scaled, y_train)
classification_standard_model = train_classifier(X_train_standard_scaled, y_train)
print(type(classification_model))
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[31]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[32]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4. - Test 1 (name: test4-1_train_classifier, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'train_classifier' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 11)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    X_test_minmax_scaled = scale_test_features(X_test, minmax_scaler)
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
    X_test_standard_scaled = scale_test_features(X_test, standard_scaler)
    classification_model = train_classifier(X_train, y_train)
    classification_minmax_model = train_classifier(X_train_minmax_scaled, y_train)
    classification_standard_model = train_classifier(X_train_standard_scaled, y_train)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
print ("Good Job!\nYou've passed the 1st test for the 'train_classifier' function implementation :-)")


# In[33]:


# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4. - Test 2 (name: test4-2_train_classifier, points: 0.4)")
print ("\t--->Testing the implementation of 'train_classifier' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 11)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    X_test_minmax_scaled = scale_test_features(X_test, minmax_scaler)
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
    X_test_standard_scaled = scale_test_features(X_test, standard_scaler)
    classification_model = train_classifier(X_train, y_train)
    classification_minmax_model = train_classifier(X_train_minmax_scaled, y_train)
    classification_standard_model = train_classifier(X_train_standard_scaled, y_train)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
assert sklearn.linear_model.LogisticRegression == type(classification_model), "Wrong retured type from 'train_classifier' method, expected a 'LogisticRegression' object"
    
print ("Good Job!\nYou've passed the 2nd test for the 'train_classifier' function implementation :-)")


# In[34]:


# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4. - Test 3 (name: test4-3_train_classifier, points: 0.5)")
print ("\t--->Testing the implementation of 'train_classifier' ...")


file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 11)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    X_test_minmax_scaled = scale_test_features(X_test, minmax_scaler)
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
    X_test_standard_scaled = scale_test_features(X_test, standard_scaler)
    classification_model = train_classifier(X_train, y_train)
    classification_minmax_model = train_classifier(X_train_minmax_scaled, y_train)
    classification_standard_model = train_classifier(X_train_standard_scaled, y_train)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert list(classification_minmax_model.classes_) == [0, 1], "Wrong class values from 'LogisticRegression' object"
assert list(np.around(classification_standard_model.coef_, decimals=2)[0]) == [-0.15, -0.6, 0.33, 0.44], "Wrong coefficient values from 'LogisticRegression' object"
    
print ("Good Job!\nYou've passed the 3rd test for the 'train_classifier' function implementation :-)")


# ### 5. Predict 
# In this section you will use the trained classification model to predict the class of the examples from the test set.

# ### 5. Instructions
# <u>method name</u>: <b>predict</b>
# <pre>The following is expected:
# --- Complete the 'predict' function to predict the class of each example in the given 'X_test' test set.
# 
# Use the 'classifier.predict()' to predict the examples from the 'X_test'.
# 
# You need to return a dataframe with only two columns (note the names):
# * 'Actual'    - contains the given 'y_test' actual labels of the test set.
# * 'Predicted' - contains the predicted values for each corresponding example in the test set.
# Note: The dataframe should have the same index as the index of 'y_test'.
# </pre>

# In[35]:


# 5.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def predict(classifier, X_test, y_test):
    y_prediction = classifier.predict(X_test)
    ansDF = pd.DataFrame({"Actual": y_test,"Predicted": y_prediction})
    return ansDF


# In[36]:


# 5.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
X, y = load_dataset(file_name, category_col_name)
X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 11)
minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
X_test_minmax_scaled = scale_test_features(X_test, minmax_scaler)
standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
X_test_standard_scaled = scale_test_features(X_test, standard_scaler)
classification_model = train_classifier(X_train, y_train)
classification_minmax_model = train_classifier(X_train_minmax_scaled, y_train)
classification_standard_model = train_classifier(X_train_standard_scaled, y_train)
df_res = predict(classification_model, X_test, y_test)
df_minmax_res = predict(classification_minmax_model, X_test_minmax_scaled, y_test)
df_standard_res = predict(classification_standard_model, X_test_standard_scaled, y_test)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[37]:


# 5.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[38]:


# 5.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 5. - Test 1 (name: test5-1_predict, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'predict' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 11)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    X_test_minmax_scaled = scale_test_features(X_test, minmax_scaler)
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
    X_test_standard_scaled = scale_test_features(X_test, standard_scaler)
    classification_model = train_classifier(X_train, y_train)
    classification_minmax_model = train_classifier(X_train_minmax_scaled, y_train)
    classification_standard_model = train_classifier(X_train_standard_scaled, y_train)
    df_res = predict(classification_model, X_test, y_test)
    df_minmax_res = predict(classification_minmax_model, X_test_minmax_scaled, y_test)
    df_standard_res = predict(classification_standard_model, X_test_standard_scaled, y_test)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
print ("Good Job!\nYou've passed the 1st test for the 'predict' function implementation :-)")


# In[39]:


# 5.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 5. - Test 2 (name: test5-2_predict, points: 0.4)")
print ("\t--->Testing the implementation of 'predict' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 11)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    X_test_minmax_scaled = scale_test_features(X_test, minmax_scaler)
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
    X_test_standard_scaled = scale_test_features(X_test, standard_scaler)
    classification_model = train_classifier(X_train, y_train)
    classification_minmax_model = train_classifier(X_train_minmax_scaled, y_train)
    classification_standard_model = train_classifier(X_train_standard_scaled, y_train)
    df_res = predict(classification_model, X_test, y_test)
    df_minmax_res = predict(classification_minmax_model, X_test_minmax_scaled, y_test)
    df_standard_res = predict(classification_standard_model, X_test_standard_scaled, y_test)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
n_res = 596    
assert df_res.shape == (n_res, 2), 'Wrong  shape for prediction result dataframe'
assert sorted(list(df_res.columns)) == ['Actual', 'Predicted'], 'Wrong  column names in dataframe'
assert df_minmax_res.shape == (n_res, 2), 'Wrong shape for prediction result dataframe'
assert sorted(list(df_minmax_res.columns)) == ['Actual', 'Predicted'], 'Wrong  column names in dataframe'
assert df_standard_res.shape == (n_res, 2), 'Wrong shape for prediction result dataframe'
assert sorted(list(df_standard_res.columns)) == ['Actual', 'Predicted'], 'Wrong  column names in dataframe'
    
print ("Good Job!\nYou've passed the 2nd test for the 'classifer_predict' function implementation :-)")


# In[40]:


# 5.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 5. - Test 3 (name: test5-3_predict, points: 0.5)")
print ("\t--->Testing the implementation of 'predict' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'imdb-movies-good_or_bad_v1_5.csv'
category_col_name = 'high_score'
try:
    X, y = load_dataset(file_name, category_col_name)
    X_train, X_test, y_train, y_test = split_to_train_and_test(X, y, 0.3, 11)
    minmax_scaler, X_train_minmax_scaled = scale_features(X_train, 'minmax')
    X_test_minmax_scaled = scale_test_features(X_test, minmax_scaler)
    standard_scaler, X_train_standard_scaled = scale_features(X_train, 'standard')
    X_test_standard_scaled = scale_test_features(X_test, standard_scaler)
    classification_model = train_classifier(X_train, y_train)
    classification_minmax_model = train_classifier(X_train_minmax_scaled, y_train)
    classification_standard_model = train_classifier(X_train_standard_scaled, y_train)
    df_res = predict(classification_model, X_test, y_test)
    df_minmax_res = predict(classification_minmax_model, X_test_minmax_scaled, y_test)
    df_standard_res = predict(classification_standard_model, X_test_standard_scaled, y_test)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
n_res = 596    
df_res['Incorrect'] = np.abs((df_res['Actual']-df_res['Predicted']).values)
df_minmax_res['Incorrect'] = np.abs((df_minmax_res['Actual']-df_minmax_res['Predicted']).values)
df_standard_res['Incorrect'] = np.abs((df_standard_res['Actual']-df_standard_res['Predicted']).values)
assert n_res - df_res['Incorrect'].sum() == 378, 'Wrong number of correct values'
assert n_res - df_minmax_res['Incorrect'].sum() == 386, 'Wrong number of correct values for minmax scaled'
assert n_res - df_standard_res['Incorrect'].sum() == 382, 'Wrong number of correct values for standard scaled'
    
print ("Good Job!\nYou've passed the 3rd test for the 'classifer_predict' function implementation :-)")


# In[ ]:




