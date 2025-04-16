
# coding: utf-8

# ![Final Lesson Exercise](images/Banner_FEX.png)

# # Lesson #9: Cross validation and classification
# ## Edible and Poisonous Mushrooms dataset

# ## About this assignment
# In this assignment, you will explore information regarding mushrooms.<br/>
# 
# This time you will practice the cross validation techniques to select best classifiers.<br />

# ## Preceding Step - import modules (packages)
# This step is necessary in order to use external packages. 
# 
# **Use the following libraries for the assignment, when needed**:

# In[1]:


# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# === CELL TYPE: IMPORTS AND SETUP 

import os                       # for testing use only

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# --------cross-validation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
# -------- classification
import sklearn
from sklearn import neighbors, tree, ensemble, naive_bayes, svm
# *** KNN
from sklearn.neighbors import KNeighborsClassifier
# *** Decision Tree; Random Forest
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
# *** Naive Bayes
from sklearn.naive_bayes import GaussianNB
# *** SVM classifier
from sklearn.svm import SVC
# --------  metrics:
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import make_scorer


# <a id="dataset_desc"></a>
# [go to basic data exploration](#data_exploration)
# ## The mushroom dataset
# In this exercise you will work with the mushroom dataset.<br/>
# **The mushroom dataset, describes two types of mushrooms**:<br/>
# e - edible, p - poisonous.
# 
# **The following is a list of features in the dataset**:
# 1. cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# 2. cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# 3. cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r, pink=p,purple=u,red=e,white=w,yellow=y
# 4. bruises?: bruises=t,no=f
# 5. odor: almond=a,anise=l,creosote=c,fishy=y,foul=f, musty=m,none=n,pungent=p,spicy=s
# 6. gill-attachment: attached=a,descending=d,free=f,notched=n
# 7. gill-spacing: close=c,crowded=w,distant=d
# 8. gill-size: broad=b,narrow=n
# 9. gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e, white=w,yellow=y
# 10. stalk-shape: enlarging=e,tapering=t
# 11. stalk-root: bulbous=b,club=c,cup=u,equal=e, rhizomorphs=z,rooted=r,missing=?
# 12. stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 13. stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 14. stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 15. stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o, pink=p,red=e,white=w,yellow=y
# 16. veil-type: partial=p,universal=u
# 17. veil-color: brown=n,orange=o,white=w,yellow=y
# 18. ring-number: none=n,one=o,two=t
# 19. ring-type: cobwebby=c,evanescent=e,flaring=f,large=l, none=n,pendant=p,sheathing=s,zone=z
# 20. spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r, orange=o,purple=u,white=w,yellow=y
# 21. population: abundant=a,clustered=c,numerous=n, scattered=s,several=v,solitary=y
# 22. habitat: grasses=g,leaves=l,meadows=m,paths=p, urban=u,waste=w,woods=d
# 
# [go to basic data exploration](#data_exploration)

# ## 1. Load the dataset and prepare dataset for classification
# In this section you will perform the following actions:
# * 1.a. Load the mushrooms dataset
# * 1.b. Basic data exploration
# * 1.c. Prepare data for classification
# * 1.d. Split dataset to train and test

# ### 1.a. Load the mushrooms dataset 
# In this section you will load the mushrooms dataset from a csv file.

# ### 1.a. Instructions
# <u>method name</u>: <b>load_dataset</b>
# <pre>The following is expected:
# --- Complete the 'load_dataset' function to load the mushroom dataset from the 'file_name' csv file
#   into a pandas dataframe.
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return df_dataset</b>

# In[2]:


# 1.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ASSISTANCE TO ANSWER 
# ---- Add assistance code here IF NEEDED:
###
### YOUR CODE HERE
###


# In[3]:


# 1.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def load_dataset(file_name):
    df=pd.read_csv(file_name)
    return df


# In[4]:


# 1.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
raw_dataset = load_dataset(file_name)
print(raw_dataset.describe())
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[5]:


# 1.a. 
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[6]:


# 1.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1.a. - Test 1 (name: test1a-1_load_dataset, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'load_dataset' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'

try:
    raw_dataset = load_dataset(file_name)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

print ("Good Job!\nYou've passed the 1st test for the 'load_dataset' function implementation :-)")


# In[7]:


# 1.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1.a. - Test 2 (name: test1a-2_load_dataset, points: 0.1)")
print ("\t--->Testing the implementation of 'load_dataset' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'

try:
    raw_dataset = load_dataset(file_name)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert raw_dataset.shape == (8127, 23) , 'Wrong shape for dataset dataframe'

print ("Good Job!\nYou've passed the 2nd test for the 'load_dataset' function implementation :-)")


# <a id="data_exploration"></a>
# ### 1.b. Basic data exploration
# In this section you may perform simple processing on the dataset to understand it better.
# 
# * Perform simple dataset exploration. It is suggested to use describe, info and a simple histogram for features' values
# * You could see also the [explanation of the dataset above](#dataset_desc)

# In[8]:


# 1.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: DATA EXPLORATION
# Use the output of the exploration for next section.
# ---- Add assistance code here IF NEEDED:
import matplotlib.pyplot as plt
# It is sugggested to use describe, info and a simple histogram for features' values
# Perform a simple dataset exploration. 
raw_dataset.info()
raw_dataset.describe()

# Transfer dataset's values to numeric ones.
df_numeric = pd.get_dummies(raw_dataset, drop_first=True)
#test
df_numeric.head()
#histograms
df_numeric.hist(bins=20, figsize=(20, 15), edgecolor='pink')
plt.tight_layout()
plt.show()


# ### 1.c. Prepare data for classification
# In this section you will the <u>prepare data in the dataset for classification</u>.
# 
# * Use the results of your [data exploration above](#data_exploration)
# * See also the [explanation of the dataset above](#dataset_desc)
# 

# ### 1.c. Instructions
# <u>method name</u>: <b>transfer_str_to_numeric_vals</b>
# <pre>The following is expected:
# 
# - Remove any rows with one or more missing value. 
# - For any duplicate rows, keep only the first one 
# - Transfer dataset's values to numeric ones.
#  Notes:
#        Each unique string value should be transfered to a corresponding 
#            unique numeric integer value.
#        All the columns contain string values, as described above.
#        It is suggested todo this in a generic way for all columns (you can do this part in two lines of code)
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return dataset</b>

# In[9]:


# 1.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ASSISTANCE TO ANSWER 
# ---- Add assistance code here IF NEEDED:
###
### YOUR CODE HERE
###


# In[10]:


# 1.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def transfer_str_to_numeric_vals(dataset):
    df_copy=dataset.copy()
    df_copy.dropna(inplace=True)
    df_copy.drop_duplicates(inplace=True)
    for col in df_copy.columns:
        df_copy[col]=pd.Categorical(df_copy[col]).codes
    return df_copy


# In[11]:


# 1.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
raw_dataset = load_dataset(file_name)
dataset = transfer_str_to_numeric_vals(raw_dataset)
cols = dataset.select_dtypes(include=[np.number])
arr_nums = [len(dataset[col].unique()) for col in cols]
print(arr_nums)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[12]:


# 1.c. 
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[13]:


# 1.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1.c. - Test 1 (name: test1c-1_transfer_str_to_numeric_vals, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'transfer_str_to_numeric_vals' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'

try:
    raw_dataset = load_dataset(file_name)
    dataset = transfer_str_to_numeric_vals(raw_dataset)
    cols = dataset.select_dtypes(include=[np.number])
    arr_nums = [len(dataset[col].unique()) for col in cols]
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
print ("Good Job!\nYou've passed the 1st test for the 'transfer_str_to_numeric_vals' function implementation :-)")


# In[14]:


# 1.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1.c. - Test 2 (name: test1c-2_transfer_str_to_numeric_vals, points: 0.1)")
print ("\t--->Testing the implementation of 'transfer_str_to_numeric_vals' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'

try:
    raw_dataset = load_dataset(file_name)
    dataset = transfer_str_to_numeric_vals(raw_dataset)
    cols = dataset.select_dtypes(include=[np.number])
    arr_nums = [len(dataset[col].unique()) for col in cols]
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert dataset.shape == (8124, 23) , 'Wrong shape for dataset dataframe'
assert dataset.select_dtypes(include=[np.number]).columns.size == 23, 'Wrong number of numeric columns'

print ("Good Job!\nYou've passed the 2nd test for the 'transfer_str_to_numeric_vals' function implementation :-)")


# In[15]:


# 1.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1.c. - Test 3 (name: test1c-3_transfer_str_to_numeric_vals, points: 0.2)")
print ("\t--->Testing the implementation of 'transfer_str_to_numeric_vals' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'

try:
    raw_dataset = load_dataset(file_name)
    dataset = transfer_str_to_numeric_vals(raw_dataset)
    cols = dataset.select_dtypes(include=[np.number])
    arr_nums = [len(dataset[col].unique()) for col in cols]
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert np.round(np.mean(arr_nums),2)==5.17,'Wrong number of unique values'

print ("Good Job!\nYou've passed the 3rd test for the 'transfer_str_to_numeric_vals' function implementation :-)")


# ### 1.d. Split dataset to train and test
# In this section you will split the dataset into a train set and a test set.

# ### 1.d. Instructions
# <u>method name</u>: <b>split_to_train_and_test</b>
# <pre>The following is expected:
# Step 1 - Split the given dataset into 'X' (feature vectors - dataframe)
#       and 'y' (corresponding labels - series), determined by the given 'label_column' column.
# Step 2 - Split X and y into 'X_train', 'X_test', and corresponding 'y_train' and 'y_test' series.
# Notes: 
#       The 'X_train' and 'X_test' dataframes should not include the 'label_column' column.
#       Use sklearn's 'train_test_split' method, which was taught in class.
#       Use the given 'rand_state' as the value for the 'random_state' parameter in 'train_test_split'.
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return X_train, X_test, y_train, y_test</b>

# In[16]:


# 1.d.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ASSISTANCE TO ANSWER 
# ---- Add assistance code here IF NEEDED:
###
### YOUR CODE HERE
###


# In[17]:


# 1.d.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def split_to_train_and_test(dataset, label_column, test_ratio, rand_state):
    y=dataset[label_column]
    X=dataset.drop(columns=label_column)
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=test_ratio,random_state=rand_state)
    return X_train,X_test,y_train,y_test


# In[18]:


# 1.d.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'
raw_dataset = load_dataset(file_name)
dataset = transfer_str_to_numeric_vals(raw_dataset) 
X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[19]:


# 1.d. 
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[20]:


# 1.d.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1.d. - Test 1 (name: test1d-1_split_to_train_and_test, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'split_to_train_and_test' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'

try:
    raw_dataset = load_dataset(file_name)
    dataset = transfer_str_to_numeric_vals(raw_dataset) 
    X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

print ("Good Job!\nYou've passed the 1st test for the 'split_to_train_and_test' function implementation :-)")


# In[21]:


# 1.d.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1.d. - Test 2 (name: test1d-2_split_to_train_and_test, points: 0.1)")
print ("\t--->Testing the implementation of 'split_to_train_and_test' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'

try:
    raw_dataset = load_dataset(file_name)
    dataset = transfer_str_to_numeric_vals(raw_dataset) 
    X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 


assert X_train.shape == (6499, 22) and X_test.shape == (1625, 22), 'Wrong shape for feature-vector train or test dataframes'
assert y_train.shape[0] == 6499 and y_test.shape[0] == 1625, 'Wrong shape for category train or test series'

print ("Good Job!\nYou've passed the 2nd test for the 'split_to_train_and_test' function implementation :-)")


# ## 2. Auxiliary classification function
# This section includes the following:
# * 2.a. get classifier object 
# * 2.b. get evaluation value on test 

# ### 2.a. get classifier object auxiliary function
# 
# The following 'get_classifier_obj' auxiliary function contain 2 input parameters:
# - classifier_name
# - params
# 
# And outputs the corresponding classifier
# 
# **The 'classifier_name' parameter**, contains a <u>string value</u><br />
#    and indicates which classifier object to return, as following<br />
# * if 'classifier_name' equals 'KNN' return 'KNeighborsClassifier()'
# * if 'classifier_name' equals 'naive_bayes' return 'GaussianNB()'
# * if 'classifier_name' equals 'svm' return 'SVC()'
# * if 'classifier_name' equals 'decision_tree' return 'tree.DecisionTreeClassifier()'
# * if 'classifier_name' equals 'random_forest ' return 'RandomForestClassifier()'
# 
# **<u>The 'param' parameter</u>** is a <u>dictionary object</u>.<br />
# It could be 'None' or contain values as following:<br />
# * If **'classifier_name' equals 'KNN'**:
#     It will contain a value for the <u>'n_neighbors'</u> key in the 'param' parameter.
#     * Set this ('n_neighbors') parameter in the returned 'KNeighborsClassifier' object accordingly.
# * If **'classifier_name' equals 'decision_tree'**:
#     It will contain a value for the <u>'max_depth' & 'min_samples_split'</u> keys in the 'param' parameter.
#     * Set these ('max_depth', 'min_samples_split') parameters in the 'DecisionTreeClassifier' object accordingly.
# * If **'classifier_name' equals 'random_forest'**:
#     It will contain a value for the <u>'n_estimators'</u> key in the 'param' parameter.
#     * Set this ('n_estimators') parameter in the 'RandomForestClassifier' object accordingly.
# * If **'classifier_name' equals 'naive_bayes' or 'svm'**, assume the 'params' will be equal None.
#     * In such a case return the relevant GaussianNB object or SVC object without setting any parameters.

# ### 2.a. Some examples
# 
# **Example 1**: The <u>'classifier_name' parameter equals 'KNN' and the 'param' parameter is not None</u>, <br />
# return a classifier object as following:<br/ >
# return KNeighborsClassifier(n_neighbors=params['n_neighbors'])
# 
# **Example 2**: The <u>'classifier_name' parameter equals 'KNN' and the 'param' parameter is None</u>,<br />
# return a classifier object as following:<br />
# return KNeighborsClassifier()
# 
# **Example 3**: The 'classifier_name' parameter equals 'svm',
# return a classifier object as following:<br />
# return SVC()

# ### 2.a. Instructions
# <u>method name</u>: <b>get_classifier_obj</b>
# <pre>The following is expected:
# --- Complete the 'get_classifier_obj' auxiliary function to return
#     a classifier object, as explained above.
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return clf</b>

# In[22]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ASSISTANCE TO ANSWER 
# ---- Add assistance code here IF NEEDED:
###
### YOUR CODE HERE
###


# In[23]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def get_classifier_obj(classifier_name, params):
    if classifier_name == "KNN":
        if params == None:
            return KNeighborsClassifier()
        else:
            return KNeighborsClassifier(n_neighbors = params['n_neighbors'])
    elif classifier_name == "decision_tree":
        if params == None:
            return tree.DecisionTreeClassifier()
        else:
            return tree.DecisionTreeClassifier(max_depth = params['max_depth'],min_samples_split=params['min_samples_split'])
    elif classifier_name == 'random_forest':
        if params == None:
            return RandomForestClassifier()
        else:
            return RandomForestClassifier(n_estimators=params['n_estimators'])
    elif classifier_name == 'naive_bayes':
        return GaussianNB()
    elif classifier_name == "svm":
        return SVC()


# In[24]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
params_knn = {'n_neighbors':3}
params_random_forest = {'n_estimators':51}
params_decision_tree = {'max_depth':4, 'min_samples_split':4}
clf_naive_bayes = get_classifier_obj("naive_bayes",None)
clf_svm = get_classifier_obj("svm",None)
clf_knn = get_classifier_obj("KNN",None)
clf_random_forest = get_classifier_obj("random_forest",None)
clf_decision_tree = get_classifier_obj("decision_tree",None)
clf_knn_with_params = get_classifier_obj("KNN",params_knn)
clf_random_forest_with_params = get_classifier_obj("random_forest",params_random_forest)    
clf_decision_tree_with_params = get_classifier_obj("decision_tree",params_decision_tree)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[25]:


# 2.a. 
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[26]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 1 (name: test2a-1_get_classifier_obj, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'get_classifier_obj' ...")


params_knn = {'n_neighbors':3}
params_random_forest = {'n_estimators':51}
params_decision_tree = {'max_depth':4, 'min_samples_split':4}
try:
    clf_naive_bayes = get_classifier_obj("naive_bayes",None)
    clf_svm = get_classifier_obj("svm",None)
    clf_knn = get_classifier_obj("KNN",None)
    clf_random_forest = get_classifier_obj("random_forest",None)
    clf_decision_tree = get_classifier_obj("decision_tree",None)
    clf_knn_with_params = get_classifier_obj("KNN",params_knn)
    clf_random_forest_with_params = get_classifier_obj("random_forest",params_random_forest)    
    clf_decision_tree_with_params = get_classifier_obj("decision_tree",params_decision_tree)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

print ("Good Job!\nYou've passed the 1st test for the 'get_classifier_obj' function :-)")


# In[27]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 2 (name: test2a-2_get_classifier_obj, points: 0.1)")
print ("\t--->Testing the implementation of 'get_classifier_obj' ...")

try:
    clf = get_classifier_obj("naive_bayes",None)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert type(clf) == sklearn.naive_bayes.GaussianNB , 'Wrong type for classifier'

print ("Good Job!\nYou've passed the 2nd test for the 'get_classifier_obj' (for 'Naive Nayes') function :-)")


# In[28]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 3 (name: test2a-3_get_classifier_obj, points: 0.1)")
print ("\t--->Testing the implementation of 'get_classifier_obj' ...")

try:
    clf = get_classifier_obj("svm",None)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert type(clf) == sklearn.svm._classes.SVC , 'Wrong type for classifier'

print ("Good Job!\nYou've passed the 3rd test for the 'get_classifier_obj' (for 'SVM') function :-)")


# In[29]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 4i (name: test2a-4i_get_classifier_obj, points: 0.1)")
print ("\t--->Testing the implementation of 'get_classifier_obj' ...")

try:
    clf = get_classifier_obj("KNN",None)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert type(clf) == sklearn.neighbors._classification.KNeighborsClassifier , 'Wrong type for classifier'
assert clf.n_neighbors == 5 , 'Wrong value for n_neighbors in KNN, expected default value'

print ("Good Job!\nYou've passed the 4th test (without paramaters) for the 'get_classifier_obj' (for 'KNN') function :-)")


# In[30]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 4ii (name: test2a-4ii_get_classifier_obj, points: 0.1)")
print ("\t--->Testing the implementation of 'get_classifier_obj' ...")

params = {'n_neighbors':3}
try:
    clf = get_classifier_obj("KNN",params)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert type(clf) == sklearn.neighbors._classification.KNeighborsClassifier , 'Wrong type for classifier'
assert clf.n_neighbors == 3 , 'Wrong value for n_neighbors in KNN, expected to be equal to input value'

print ("Good Job!\nYou've passed the 4th test (with paramaters) for the 'get_classifier_obj' (for 'KNN') function :-)")


# In[31]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 5i (name: test2a-5i_get_classifier_obj, points: 0.1)")
print ("\t--->Testing the implementation of 'get_classifier_obj' ...")

try:
    clf = get_classifier_obj("random_forest",None)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert type(clf) == sklearn.ensemble._forest.RandomForestClassifier , 'Wrong type for classifier'
assert clf.n_estimators == 100 , 'Wrong value for n_estimators in random forest, expected default value'

print ("Good Job!\nYou've passed the 5th test (without paramaters) for the 'get_classifier_obj' (for' Random Forest') function :-)")


# In[32]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 5ii (name: test2a-5ii_get_classifier_obj, points: 0.1)")
print ("\t--->Testing the implementation of 'get_classifier_obj' ...")

params = {'n_estimators':51}

try:
    clf = get_classifier_obj("random_forest",params)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert type(clf) == sklearn.ensemble._forest.RandomForestClassifier , 'Wrong type for classifier'
assert clf.n_estimators == 51 , 'Wrong value for n_estimators in random forest, expected to be equal to input value'

print ("Good Job!\nYou've passed the 5th test (with paramaters) for the 'get_classifier_obj' (for' Random Forest') function :-)")


# In[33]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 6i (name: test2a-6i_get_classifier_obj, points: 0.1)")
print ("\t--->Testing the implementation of 'get_classifier_obj' ...")

try:
    clf = get_classifier_obj("decision_tree",None)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert type(clf) == sklearn.tree._classes.DecisionTreeClassifier , 'Wrong type for classifier'
assert clf.max_depth is None and clf.min_samples_split==2 , 'Wrong values for max_depth or min_samples_split in Decision Tree, expected default value'

print ("Good Job!\nYou've passed the 6th test (without paramaters) for the 'get_classifier_obj' (for 'Decision Tree') function :-)")


# In[34]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 6ii (name: test2a-6ii_get_classifier_obj, points: 0.1)")
print ("\t--->Testing the implementation of 'get_classifier_obj' ...")

params = {'max_depth':4, 'min_samples_split':4}
try:
    clf = get_classifier_obj("decision_tree",params)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert type(clf) == sklearn.tree._classes.DecisionTreeClassifier , 'Wrong type for classifier'
assert clf.max_depth==4 and clf.min_samples_split==4 , 'Wrong values for max_depth or min_samples_split in Decision Tree, expected to be equal to input values'

print ("Good Job!\nYou've passed the 6th test (with paramaters) for the 'get_classifier_obj' (for 'Decision Tree') function :-)")


# ### 2.b. get evaluation value on test auxiliary function
# 
# The following **'calc_evaluation_val'** auxiliary function contain <u>3 input parameters</u>:
# - eval_metric - the evaluation metric expected to return
# - y_test      - the actual test target categories
# - y_predicted - the predicted test categories
# 
# **The 'eval_metric' parameter**, contains a <u>string value</u><br />
#    and indicates which evaluation metric, which object to return, as following:<br />
# * if 'eval_metric' equals <u>'accuracy'</u> return the <u>accuracy_score</u> evaluation float value
# * if 'eval_metric' equals <u>'precision'</u> return the <u>precision_score</u> evaluation float value
# * if 'eval_metric' equals <u>'recall'</u> return the <u>recall_score</u> evaluation float value
# * if 'eval_metric' equals <u>'f1'</u> return the <u>f1_score</u> evaluation float value
# * if 'eval_metric' equals <u>'confusion_matrix'</u> return the <u>confusion_matrix</u> evaluation np.ndarray value 

# ### 2.b. Instructions
# <u>method name</u>: <b>calc_evaluation_val</b>
# <pre>The following is expected:
# --- Complete the 'calc_evaluation_val' auxiliary function to return
#     the evaluation value of the trained model, as explained above.
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return evaluation_val</b>

# In[35]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ASSISTANCE TO ANSWER 
# ---- Add assistance code here IF NEEDED:
###
### YOUR CODE HERE
###


# In[36]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def calc_evaluation_val(eval_metric, y_test, y_predicted):
    if eval_metric == 'accuracy':
        return metrics.accuracy_score(y_true=y_test,y_pred=y_predicted)
    elif eval_metric == 'precision':
        return precision_score(y_test,y_predicted)
    elif eval_metric=='recall':
        return recall_score(y_test,y_predicted)
    elif eval_metric=='f1':
        return f1_score(y_test,y_predicted)
    elif eval_metric=='confusion_matrix':
        return confusion_matrix(y_test,y_predicted)


# In[37]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
y_test = pd.Series([1,0,1,0,1,0,1,0,1,0])
y_predicted = pd.Series([1,1,1,1,1,1,1,0,0,0])
accuracy_val = calc_evaluation_val("accuracy", y_test, y_predicted)
precision_val = calc_evaluation_val("precision", y_test, y_predicted)
recall_val = calc_evaluation_val("recall", y_test, y_predicted)
f1_val = calc_evaluation_val("f1", y_test, y_predicted)
confusion_matrix_val = calc_evaluation_val("confusion_matrix", y_test, y_predicted)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[38]:


# 2.b. 
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[39]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.b. - Test 1 (name: test2b-1_calc_evaluation_val, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'calc_evaluation_val' ...")

try:
    y_test = pd.Series([1,0,1,0,1,0,1,0,1,0])
    y_predicted = pd.Series([1,1,1,1,1,1,1,0,0,0])
    accuracy_val = calc_evaluation_val("accuracy", y_test, y_predicted)
    precision_val = calc_evaluation_val("precision", y_test, y_predicted)
    recall_val = calc_evaluation_val("recall", y_test, y_predicted)
    f1_val = calc_evaluation_val("f1", y_test, y_predicted)
    confusion_matrix_val = calc_evaluation_val("confusion_matrix", y_test, y_predicted)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

print ("Good Job!\nYou've passed the 1st test for the 'calc_evaluation_val' function implementation :-)")


# In[40]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.b. - Test 2 (name: test2b-2_calc_evaluation_val, points: 0.1)")
print ("\t--->Testing the implementation of 'calc_evaluation_val' ...")

try:
    y_test = pd.Series([1,0,1,0,1,0,1,0,1,0])
    y_predicted = pd.Series([1,1,1,1,1,1,1,0,0,0])
    evaluation_val = calc_evaluation_val("accuracy", y_test, y_predicted)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
assert 0.6 == np.round(evaluation_val,2),"Wrong 'accuracy' value"

print("accuracy:",evaluation_val)

print ("Good Job!\nYou've passed the 2nd test for the 'calc_evaluation_val' function implementation (for 'accuracy') :-)")


# In[41]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.b. - Test 3 (name: test2b-3_calc_evaluation_val, points: 0.1)")
print ("\t--->Testing the implementation of 'calc_evaluation_val' ...")

try:
    y_test = pd.Series([1,0,1,0,1,0,1,0,1,0])
    y_predicted = pd.Series([1,1,1,1,1,1,1,0,0,0])
    evaluation_val = calc_evaluation_val("precision", y_test, y_predicted)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert 0.57 == np.round(evaluation_val,2),"Wrong 'precision' value"

print("precision:",evaluation_val)

print ("Good Job!\nYou've passed the 3rd test for the 'calc_evaluation_val' function implementation (for 'precision') :-)")


# In[42]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.b. - Test 4 (name: test2b-4_calc_evaluation_val, points: 0.1)")
print ("\t--->Testing the implementation of 'calc_evaluation_val' ...")

try:
    y_test = pd.Series([1,0,1,0,1,0,1,0,1,0])
    y_predicted = pd.Series([1,1,1,1,1,1,1,0,0,0])
    evaluation_val = calc_evaluation_val("recall", y_test, y_predicted)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
assert 0.8 == np.round(evaluation_val,2),"Wrong 'recall' value"

print("recall:",evaluation_val)

print ("Good Job!\nYou've passed the 4th test for the 'calc_evaluation_val' function implementation (for 'recall') :-)")


# In[43]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.b. - Test 5 (name: test2b-5_calc_evaluation_val, points: 0.1)")
print ("\t--->Testing the implementation of 'calc_evaluation_val' ...")

try:
    y_test = pd.Series([1,0,1,0,1,0,1,0,1,0])
    y_predicted = pd.Series([1,1,1,1,1,1,1,0,0,0])
    evaluation_val = calc_evaluation_val("f1", y_test, y_predicted)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert 0.67 == np.round(evaluation_val,2),"Wrong 'f1' value"

print("f1:",evaluation_val)

print ("Good Job!\nYou've passed the 5th test for the 'calc_evaluation_val' function implementation (for 'f1') :-)")


# In[44]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.b. - Test 6 (name: test2b-6_calc_evaluation_val, points: 0.1)")
print ("\t--->Testing the implementation of 'calc_evaluation_val' ...")

try:
    y_test = pd.Series([1,0,1,0,1,0,1,0,1,0])
    y_predicted = pd.Series([1,1,1,1,1,1,1,0,0,0])
    evaluation_val = calc_evaluation_val("confusion_matrix", y_test, y_predicted)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

assert np.ndarray==type(evaluation_val),'Wrong evaluation value for confusion matrix'

assert 1 == evaluation_val[1,0],"Wrong 'False Negative' value"

print("confusion_matrix:\n---------------------")
df_confusion_matrix = pd.DataFrame(evaluation_val,index=['Actual Negative','Actual Positive'],columns=['Predicted Negative','Predicted Positive'])
print(df_confusion_matrix)

print ("Good Job!\nYou've passed the 6th test for the 'calc_evaluation_val' function implementation (for 'confusion matrix') :-)")


# ## 3. Grid Search Cross-validation - hyper-parameter tuning

# ### 3.a. KNN - find best K for KNN

# ### 3.a. Instructions
# <u>method name</u>: <b>find_best_k_for_KNN</b>
# <pre>The following is expected:
# --- Complete the 'find_best_k_for_KNN' function to find the K-hyperparameter (i.e. the 
#      'n_neighbors' parameter) for the KNN algorithm, running on the mushroom 
#      dataset's train-set, using grid search cross validation 'GridSearchCV' function.
# Notes:
#       You need to use the average f1 score, in order to choose the best K 
#          for this dataset.
#       Use the 3,7,9,11 possible values as possibilities to the 'n_neighbors' parameter.
#       You need to return the best K value as well as the best average f1 score.
#       Use the 'make_scorer', as the function for 'scoring' parameter
# </pre>      
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return best_K, best_f1_val</b>

# In[45]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ASSISTANCE TO ANSWER 
# ---- Add assistance code here IF NEEDED:
###
### YOUR CODE HERE
###


# In[46]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def find_best_k_for_KNN(X_train, y_train):
    params={'n_neighbors':range(3,11,2)}
    knn=KNeighborsClassifier()
    
    clf=GridSearchCV(knn,params,scoring=make_scorer(metrics.f1_score,greater_is_better=True))
    clf.fit(X_train,y_train)
    best_K=clf.best_params_['n_neighbors']
    best_f1_vals=clf.best_score_
    return best_K,best_f1_vals


# In[47]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'
raw_dataset = load_dataset(file_name)
sub_set_of_raw_dataset_2k_4k = raw_dataset.iloc[2000:4000,:]
dataset = transfer_str_to_numeric_vals(sub_set_of_raw_dataset_2k_4k) 
X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
best_K, best_f1_KNN_params = find_best_k_for_KNN(X_train, y_train)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[48]:


# 3.a. 
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[49]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.a. - Test 1 (name: test3a-1_find_best_k_for_KNN, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'find_best_k_for_KNN' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'

try:
    raw_dataset = load_dataset(file_name)
    sub_set_of_raw_dataset_2k_4k = raw_dataset.iloc[2000:4000,:]
    dataset = transfer_str_to_numeric_vals(sub_set_of_raw_dataset_2k_4k) 
    X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
    best_K, best_f1_KNN_params = find_best_k_for_KNN(X_train, y_train)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

print ("Good Job!\nYou've passed the 1st test for the 'find_best_k_for_KNN' function implementation :-)")


# In[50]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.a. - Test 2 (name: test3a-2_find_best_k_for_KNN, points: 0.4)")
print ("\t--->Testing the implementation of 'find_best_k_for_KNN' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'

try:
    raw_dataset = load_dataset(file_name)
    sub_set_of_raw_dataset_2k_4k = raw_dataset.iloc[2000:4000,:]
    dataset = transfer_str_to_numeric_vals(sub_set_of_raw_dataset_2k_4k) 
    X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

found_assertion_combination = False    
for i in range(50):
    try:
        best_K, best_f1_KNN_params = find_best_k_for_KNN(X_train, y_train)
        ###
        ### AUTOGRADER TEST - DO NOT REMOVE
        ###
        assert best_K==7,'Wrong best K hyperparameter value for KNN'
        found_assertion_combination = True
        print ('... successful')
        break
    except Exception as e:
        print ('Try:',i+1,'failed')
        print ('\tError Message:', str(e))
        print ('Will try again ...')

assert found_assertion_combination,'Wrong best K hyperparameter value for KNN'

print ("Good Job!\nYou've passed the 2nd test for the 'find_best_k_for_KNN' function implementation :-)")


# In[51]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.a. - Test 3 (name: test3a-3_find_best_k_for_KNN, points: 0.4)")
print ("\t--->Testing the implementation of 'find_best_k_for_KNN' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'

try:
    raw_dataset = load_dataset(file_name)
    sub_set_of_raw_dataset_2k_4k = raw_dataset.iloc[2000:4000,:]
    dataset = transfer_str_to_numeric_vals(sub_set_of_raw_dataset_2k_4k) 
    X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

found_assertion_combination = False
for i in range(50):
    try:
        best_K, best_f1_KNN_params = find_best_k_for_KNN(X_train, y_train)
        ###
        ### AUTOGRADER TEST - DO NOT REMOVE
        ###
        assert 0.993==np.round(best_f1_KNN_params,3),'Wrong F1 value for best K (number of neigbors) for KNN'        
        found_assertion_combination = True
        print ('... successful')
        break
    except Exception as e:
        print ('Try:',i+1,'failed')
        print ('\tError Message:', str(e))
        print ('Will try again ...')

assert found_assertion_combination, 'Wrong F1 value for best K (number of neigbors) for KNN'

print ("Good Job!\nYou've passed the 3rd test for the 'find_best_k_for_KNN' function implementation :-)")


# ### 3.b. Decision tree - find best parameter pair

# ### 3.b. Instructions
# <u>method name</u>: <b>find_best_decision_tree_params</b>
# <pre>The following is expected:
# --- Complete the 'find_best_decision_tree_params' function to find best
#           'max_depth' and 'min_samples_split' hyperparameter pair,
#           for the decision-trees algorithm, running on the mushroom dataset.
#      Again, use the grid search cross validation 'GridSearchCV' function.
# Notes:
#        You need to use the average f1 score, in order to choose the best  
#            'max_depth' and 'min_samples_split' hyperparameter pair for this dataset.
#        Use 2,4,6 as possible values as possibilities for the 'max_depth' parameter
#            and 5,10,20 for 'min_samples_split'.
#        You need to return the value of best 'max_depth', 'min_samples_split' pair
#            and best average f1 score.
#        Use the 'make_scorer', as the function for 'scoring' parameter
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return best_max_depth, best_min_samples_split, best_f1_val</b>

# In[52]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ASSISTANCE TO ANSWER 
# ---- Add assistance code here IF NEEDED:
###
### YOUR CODE HERE
###


# In[53]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def find_best_decision_tree_params(X_train, y_train):
   
    params={'max_depth':[2,4,6],'min_samples_split':[5,10,20]}
    best_tree=tree.DecisionTreeClassifier()
    clf=GridSearchCV(best_tree,params,scoring=make_scorer(metrics.f1_score,greater_is_better=True))
    clf.fit(X_train,y_train)
    best_max_depth=clf.best_params_['max_depth']
    best_min_samples_split=clf.best_params_['min_samples_split']
    best_f1_vals=clf.best_score_
    return best_max_depth,best_min_samples_split,best_f1_vals


# In[54]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'
raw_dataset = load_dataset(file_name)
sub_set_of_raw_dataset_2k_4k = raw_dataset.iloc[2000:4000,:]
dataset = transfer_str_to_numeric_vals(sub_set_of_raw_dataset_2k_4k) 
X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
best_max_dep, best_min_smpl_splt, best_f1_DT_params = find_best_decision_tree_params(X_train, y_train)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[55]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[56]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.b. - Test 1 (name: test3b-1_find_best_decision_tree_params, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'find_best_decision_tree_params' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'

try:
    raw_dataset = load_dataset(file_name)
    sub_set_of_raw_dataset_2k_4k = raw_dataset.iloc[2000:4000,:]
    dataset = transfer_str_to_numeric_vals(sub_set_of_raw_dataset_2k_4k) 
    X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
    best_max_dep, best_min_smpl_splt, best_f1_DT_params = find_best_decision_tree_params(X_train, y_train)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
         
print ("\nGood Job!\nYou've passed the 1st test for the 'find_best_decision_tree_params' method :-)")


# In[57]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.b. - Test 2 (name: test3b-2_find_best_decision_tree_params, points: 0.4)")
print ("\t--->Testing the implementation of 'find_best_decision_tree_params' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'

try:
    raw_dataset = load_dataset(file_name)
    sub_set_of_raw_dataset_2k_4k = raw_dataset.iloc[2000:4000,:]
    dataset = transfer_str_to_numeric_vals(sub_set_of_raw_dataset_2k_4k) 
    X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

found_assertion_combination = False
for i in range(50):
    try:
        best_max_dep, best_min_smpl_splt, best_f1_DT_params = find_best_decision_tree_params(X_train, y_train)
        ###
        ### AUTOGRADER TEST - DO NOT REMOVE
        ###
        assert best_max_dep==4 and best_min_smpl_splt==5,'Wrong best Decision-tree hyperparameter pair values'
        found_assertion_combination = True
        print ('... successful')
        break
    except Exception as e:
        print ('Try:',i+1,'failed')
        print ('\tError Message:', str(e))
        print ('Will try again ...')

assert found_assertion_combination,'Wrong best Decision-tree hyperparameter pair values'
         
print ("\nGood Job!\nYou've passed the 2nd test for the 'find_best_decision_tree_params' method :-)")


# In[58]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.b. - Test 3 (name: test3b-3_find_best_decision_tree_params, points: 0.4)")
print ("\t--->Testing the implementation of 'find_best_decision_tree_params' ...")
print ("\n\t====> Full grading test - the following test can not be seen before submission")
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ### 3.c. Random forest - find best number of estimators parameter

# ### 3.c. Instructions
# <u>method name</u>: <b>find_best_random_forest_num_estimators</b>
# <pre>The following is expected:
# --- Complete the 'find_best_random_forest_num_estimators' function to find best
#     'n_estimators' hyperparameter values, 
#      for the random-forest algorithm, running on the mushroom dataset's train-set.
#      Again, use the grid search cross validation 'GridSearchCV' function.
# Notes:
#        You need to use the average f1 score, in order to choose the best  
#            'n_estimators' hyperparameter for this dataset.
#        Use 11,51,71 as possible values as possibilities for the 
#            'n_estimators' parameter.
#        You need to return the value of 'n_estimators' best value 
#            and the best average f1 score.
#        Use the 'make_scorer', as the function for 'scoring' parameter     
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return best_num_estimators, best_f1_val</b>

# In[59]:


# 3.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ASSISTANCE TO ANSWER 
# ---- Add assistance code here IF NEEDED:
###
### YOUR CODE HERE
###


# In[60]:


# 3.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def find_best_random_forest_num_estimators(X_train, y_train):
    params = {'n_estimators':[11,51,71]}
    rf=RandomForestClassifier()
    clf=GridSearchCV(rf,params,scoring=make_scorer(metrics.f1_score,greater_is_better=True))
    clf.fit(X_train,y_train)
    best_num_estimators=clf.best_params_['n_estimators']
    best_f1_vals=clf.best_score_
    return best_num_estimators,best_f1_vals


# In[61]:


# 3.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'
raw_dataset = load_dataset(file_name)
sub_set_of_raw_dataset_2k_4k = raw_dataset.iloc[2000:4000,:]
dataset = transfer_str_to_numeric_vals(sub_set_of_raw_dataset_2k_4k) 
X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
best_n_estimators, best_f1_RF_params = find_best_random_forest_num_estimators(X_train, y_train)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[62]:


# 3.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[63]:


# 3.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.c. - Test 1 (name: test3c-1_find_best_random_forest_num_estimators, points: 0.1) - Sanity")
print ("\t--->Testing the implementation of 'find_best_random_forest_num_estimators' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'

try:
    raw_dataset = load_dataset(file_name)
    sub_set_of_raw_dataset_2k_4k = raw_dataset.iloc[2000:4000,:]
    dataset = transfer_str_to_numeric_vals(sub_set_of_raw_dataset_2k_4k) 
    X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
    best_n_estimators, best_f1_RF_params = find_best_random_forest_num_estimators(X_train, y_train)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 
    
print ("Good Job!\nYou've passed the 1st test for the 'find_best_random_forest_num_estimators' function implementation :-)")


# In[64]:


# 3.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.c. - Test 2 (name: test3c-2_find_best_random_forest_num_estimators, points: 0.4)")
print ("\t--->Testing the implementation of 'find_best_random_forest_num_estimators' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'

try:
    raw_dataset = load_dataset(file_name)
    sub_set_of_raw_dataset_2k_4k = raw_dataset.iloc[2000:4000,:]
    dataset = transfer_str_to_numeric_vals(sub_set_of_raw_dataset_2k_4k) 
    X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

found_assertion_combination = False
for i in range(50):
    try:
        best_n_estimators, best_f1_RF_params = find_best_random_forest_num_estimators(X_train, y_train)
        ###
        ### AUTOGRADER TEST - DO NOT REMOVE
        ###
        assert best_n_estimators==71,'Wrong best num-estimator hyperparameter value for Random-Forest'
        found_assertion_combination = True
        print ('... successful')
        break
    except Exception as e:
        print ('\tError Message:', str(e))
        print ('Will try again, num-try:',i+1)

assert found_assertion_combination,'Wrong best num-estimator hyperparameter value for Random-Forest'
         
print ("Good Job!\nYou've passed the 2nd test for the 'find_best_random_forest_num_estimators' function implementation :-)")


# In[65]:


# 3.c.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.c. - Test 3 (name: test3c-3_find_best_random_forest_num_estimators, points: 0.4)")
print ("\t--->Testing the implementation of 'find_best_random_forest_num_estimators' ...")
print ("\n\t====> Full grading test - the following test can not be seen before submission")
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ## 4. Select The best model
# In this section you need to select the best model using the 
#   average cross validation recall score.

# ### 4. Instructions
# <u>method name</u>: <b>find_best_model</b>
# <pre>The following is expected:
# --- Complete the 'find_best_model' function to find best
#     trained model out of the following three models:
#     Model 1: Decision Trees model (with given 'max_depth' and 'min_samples_split' parameters)
#     Model 2: (Gaussian) Naive Bayes model (with default parameters). 
#     Model 3: SVM model (SVC with default parameters).
# Notes:
#        You need to use the average recall score, in order to choose the best  
#            trained classification model.
#        You need to return the best trained classifier object and the best average recall score.
#           The average score, should be performed on a 10-fold cross validation.
#           Use the 'cross_val_score' method to calculate the best average score.
# </pre>
# <hr>
# The return statement should look similar to the following statement:<br />
# <b>return best_clf, best_recall_val</b>

# In[66]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ASSISTANCE TO ANSWER 
# ---- Add assistance code here IF NEEDED:
###
### YOUR CODE HERE
###


# In[67]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def find_best_model(X_train, y_train, max_depth_val, min_samples_split_val):
    params={'max_depth':max_depth_val,'min_samples_split':min_samples_split_val}
    best_clf=None
    best_recall_val=-1
    models = ["decision_tree","naive_bayes","svm"]
    for mod in models:
        clf=get_classifier_obj(mod,params)
        clf.fit(X_train,y_train)
        y_pred_train=clf.predict(X_train)
        rec_val=calc_evaluation_val('recall',y_train,y_pred_train)
        if rec_val>best_recall_val:
            best_recall_val = rec_val
            best_clf=clf
    return best_clf,best_recall_val


# In[68]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:
file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'
max_dep=4
min_smpl_splt=5
raw_dataset = load_dataset(file_name)
dataset = transfer_str_to_numeric_vals(raw_dataset) 
X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
best_clf, best_recall_val=find_best_model(X_train, y_train, max_dep, min_smpl_splt)
# --- add additional code to check your code if needed:
###
### YOUR CODE HERE
###


# In[69]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# ---- Add your additional tests here if needed:
###
### YOUR CODE HERE
###


# In[70]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4. - Test 1 (name: test4-1_find_best_model, points: 0.1) - sanity")
print ("\t--->Testing the implementation of 'find_best_model' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'
max_dep=4
min_smpl_splt=5

try:
    raw_dataset = load_dataset(file_name)
    dataset = transfer_str_to_numeric_vals(raw_dataset) 
    X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
    best_clf, best_recall_val=find_best_model(X_train, y_train, max_dep, min_smpl_splt)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

print ("Good Job!\nYou've passed the 1st test for the 'find_best_model' function implementation :-)")


# In[71]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4. - Test 2 (name: test4-2_find_best_model, points: 0.5)")
print ("\t--->Testing the implementation of 'find_best_model' ...")

file_name = '.' + os.sep + 'data' + os.sep + 'mushrooms.csv'
test_ratio, rand_state = 0.2, 42
category_col_name = 'class'
max_dep=4
min_smpl_splt=5

try:
    raw_dataset = load_dataset(file_name)
    dataset = transfer_str_to_numeric_vals(raw_dataset) 
    X_train, X_test, y_train, y_test = split_to_train_and_test(dataset, category_col_name, test_ratio, rand_state)
    best_clf, best_recall_val=find_best_model(X_train, y_train, max_dep, min_smpl_splt)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print ('\tError Message:', str(e))
    print ('Try fixing your implementation')
    raise 

###
### AUTOGRADER TEST - DO NOT REMOVE
###

assert type(best_clf) == sklearn.tree._classes.DecisionTreeClassifier , 'Wrong type for best classifier'

print ("Good Job!\nYou've passed the 2nd test for the 'find_best_model' function implementation :-)")


# In[72]:


# 4.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4. - Test 3 (name: test4-3_find_best_model, points: 0.4)")
print ("\t--->Testing the implementation of 'find_best_model' ...")
print ("\n\t====> Full grading test - the following test can not be seen before submission")
###
### AUTOGRADER TEST - DO NOT REMOVE
###

