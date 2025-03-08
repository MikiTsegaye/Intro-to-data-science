
# coding: utf-8

# ![Final Lesson Exercise](images/Banner_FEX.png)

# # Lesson #1: Introduction to Data Science
# ## Good chocolate - The 'flavors of cacao' dataset

# <table  align='left' valign="top"><tr>
# <td align='left'>
#     <a title='Chocolate and cocoa beans'> <img width="400" src="./images/a-01-dark-chocolate.jpg" width="px" align='left' /></a>
# </td>
# <td align='left' style="vertical-align:top">
# <div align='left' style="font-size:130%">
# <h3>Lets ask some questions:</h3>
# <ul align='left'>
# <li>What makes a chocolate good?</li>
# <li>Where does good chocolate come from?</li> 
# <li>Do you know what characterizes good chocolate?</li> 
# <li>Is darker chocolate better?</li>
# <li>How influential is the type of beans used?</li>
# <li>Is it the type of beans or their origin?</li>
# <li>Is the company producing it a major factor?</li>
# </ul></div></td>
# </tr>
# </table>

# ## About this assignment
# In this assignment, you will explore a dataset on chocolates.<br/>
# 
# You will do so, using the pandas library, introduced in the lesson.

# ## Preceding Step - import modules (packages)
# This step is necessary in order to use external packages. 
# 
# **Use the following libraries for the assignment, when needed**:

# In[1]:


# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# === CELL TYPE: IMPORTS AND SETUP 

import pandas as pd
import os # for testing use only


# ### 1. Loading the 'flavors of cacao' dataset

# ### Instructions
# <u>method name</u>: <b>load_csv</b>
# <pre>The following is expected:
# --- Complete the 'load_csv' function to load the 'flavors of cacao' dataset 
#     from the csv, located in the file_name parameter into a pandas dataframe
#     and return the dataframe
# </pre>

# In[2]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def load_csv(file_name):
    return pd.read_csv(file_name)


# In[3]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:

file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
df_cocoa = load_csv(file_name)


# In[4]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# Add your additional tests here if needed:
 
df_cocoa


# In[5]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1 - Test 1 (0.2 points) - Sanity")
print ("\t--->Testing the implementation of 'load_csv' ...")
try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))

print ("Good Job!\nYou've passed the 1st test for the 'load_dataset' function implementation :-)")


# In[6]:


# 1.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 1 - Test 2 (0.3 points)")
print ("\t---> - Testing the implementation of 'load_csv' ...")

try:
    # file_name = '.' + os.sep + 'data' + os.sep + 'flavors_of_cacao.csv'
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))
    
assert df_chocolate is not None, "You did not return an object from the 'load_csv' function, try again"
assert isinstance(df_chocolate, pd.DataFrame), "The object you returned is not a dataframe, try again"

print ("Good Job!\nYou've passed the 2nd test for the 'load_dataset' function implementation :-)")


# ### 2. Get some information about the dataset (from the dataframe)
# #### The dataset contains the following columns:
# * Company (Maker-if known) - Name of the company manufacturing the bar
# * Specific Bean Origin or Bar Name -  The specific geo-region of origin for the bar.
# * REF - reference number
# * Review Date - Year of publication of the review.
# * Cocoa Percent - Cocoa percentage (darkness) of the chocolate bar being reviewed.
# * Company Location - Manufacturer base country.
# * Rating - The rating that the experts gave (5 is the best, 1 is the worst)
# * Bean Type - The variety (breed) of bean used, if provided
# * Broad Bean Origin - The broad geo-region of origin for the bean

# #### 2.a. Obtain the number of rows

# ### Instructions
# <u>method name</u>: <b>get_number_of_rows</b>
# <pre>The following is expected:
# --- Complete the 'get_num_of_rows' function to return the number of rows 
#     in the pandas dataframe, given in the parameter 'dataframe'
# </pre>

# In[7]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def get_number_of_rows(dataframe):
    return len(dataframe)


# In[8]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:

file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
df_cocoa = load_csv(file_name)
n_rows = get_number_of_rows(df_cocoa)
print (n_rows)


# In[9]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# Add your additional tests here if needed:

###
### YOUR CODE HERE
###


# In[10]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 1 (0.2 points) - Sanity")
print ("\t--->Testing the implementation of 'get_number_of_rows' ...")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    num_of_rows    = get_number_of_rows(df_chocolate)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))
    
print ("Good Job!\nYou've passed the 1st test for the 'get_number_of_rows' function implementation :-)")


# In[11]:


# 2.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.a. - Test 2 (0.2 points)")
print ("\t--->Testing the implementation of 'get_number_of_rows' ...")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    num_of_rows    = get_number_of_rows(df_chocolate)
    print ('Num of rows=%d' %(num_of_rows))
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))
    
assert num_of_rows != 0, "The number of rows can't be 0"

print ("Good Job!\nYou've passed the 2nd test for the 'get_number_of_rows' function implementation :-)")


# #### 2.b. Obtain the number of columns

# ### Instructions
# <u>method name</u>: <b>get_number_of_columns</b>
# <pre>The following is expected:
# --- Complete the 'get_num_of_columns' function to return the number of columns 
#     in the pandas dataframe, given in the parameter 'dataframe'
# </pre>

# In[12]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def get_number_of_columns(dataframe):
    return len(dataframe.columns)


# In[13]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:

file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
df_cocoa = load_csv(file_name)
n_cols = get_number_of_columns(df_cocoa)
print(n_cols)


# In[14]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# Add your additional tests here if needed:

###
### YOUR CODE HERE
###


# In[15]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.b. - Test 1 (0.2 points) - Sanity")
print ("\t--->Testing the implementation of 'get_number_of_columns' ...")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    num_of_columns = get_number_of_columns(df_chocolate)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))
    
print ("Good Job!\nYou've passed the 1st test for the 'get_number_of_columns' function implementation :-)")


# In[16]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.b. - Test 2 (0.2 points)")
print ("\t--->Testing the implementation of 'get_number_of_columns' ...")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    num_of_columns = get_number_of_columns(df_chocolate)
    print ('Num of columns=%d' %(num_of_columns))
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))
    
assert num_of_columns!=0, "The number of columns can't be 0"

print ("Good Job!\nYou've passed the 2nd test for the 'get_number_of_columns' function implementation :-)")


# In[17]:


# 2.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 2.b. - Test 3 (0.2 points)")
print ("\t--->Testing the implementation of 'get_number_of_rows' and 'get_number_of_columns' ...")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    num_of_rows    = get_number_of_rows(df_chocolate)
    num_of_columns = get_number_of_columns(df_chocolate)
    print ('Num of rows=%d' %(num_of_rows))
    print ('Num of columns=%d' %(num_of_columns))
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))
    
assert 4 == num_of_rows % num_of_columns, "You've got a wrong number of rows or columns"

print ("Good Job!\nYou've passed the 3rd test for the 'get_number_of_columns' function implementation :-)")


# ### 3. Reading information from the dataframe:

# #### 3.a. Get rows in range

# ### Instructions
# <u>method name</u>: <b>get_rows_in_range</b>
# <pre>The following is expected:
# --- Complete the 'get_rows_in_range' function to return only the rows 
#     in a pandas dataframe, given in the parameter 'dataframe', within the
#     range 'first_row' and 'last_row' (last row is excluded)
# </pre>

# In[18]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def get_rows_in_range(dataframe, first_row, last_row):
    return dataframe.iloc[first_row:last_row,:]


# In[19]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:

file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
df_cocoa = load_csv(file_name)
row_begin_range, row_end_range = 1310, 1320
df_rows_in_range = get_rows_in_range(df_cocoa, row_begin_range, row_end_range)



# In[20]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# Add your additional tests here if needed:

###
### YOUR CODE HERE
###


# In[21]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.a. - Test 1 (0.2 points) - Sanity")
print ("\t--->Testing the implementation of 'get_rows_in_range' ...")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    df_ranged_rows = get_rows_in_range(df_chocolate, 1310, 1320)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))
    
print ("Good Job!\nYou've passed the 1st test for the 'get_rows_in_range' function implementation :-)")


# In[22]:


# 3.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.a. - Test 2 (0.5 points)")
print ("\t--->Testing the implementation of 'get_rows_in_range' ...")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    df_ranged_rows = get_rows_in_range(df_chocolate, 1310, 1320)
    num_of_rows    = get_number_of_rows(df_ranged_rows)
    num_of_columns = get_number_of_columns(df_ranged_rows)
    print ('Num of rows=%d' %(num_of_rows))
    print ('Num of columns=%d' %(num_of_columns))
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))
    
assert num_of_rows==10 and num_of_columns==9, "Wrong number of rows or columns"

print ("Good Job!\nYou've passed the 2nd test for the 'get_rows_in_range' function implementation :-)")

print ('\nSub dataframe:')
df_ranged_rows


# #### 3.b. Get columns in range

# ### Instructions
# <u>method name</u>: <b>get_columns_in_range</b>
# <pre>The following is expected:
# --- Complete the 'get_columns_in_range' function to return only the columns 
#     in a pandas dataframe, given in the parameter 'dataframe', within the
#     range 'first_column' and 'last_column' (last column is excluded)
# </pre>

# In[23]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def get_columns_in_range(dataframe, first_column, last_column):
    return dataframe.iloc[:,first_column:last_column]


# In[24]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:

file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
df_cocoa = load_csv(file_name)
row_begin_range, row_end_range = 1310, 1320
df_rows_in_range = get_rows_in_range(df_cocoa, row_begin_range, row_end_range)
col_begin_range, col_end_range = 4, 6
df_cols_in_ranged = get_columns_in_range(df_rows_in_range, col_begin_range, col_end_range)



# In[25]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# Add your additional tests here if needed:
df_cols_in_ranged



# In[26]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.b. - Test 1 (0.2 points) - Sanity")
print ("\t--->Testing the implementation of 'get_columns_in_range' ...")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    df_ranged = get_columns_in_range(df_chocolate, 4, 6)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))
    
print ("Good Job!\nYou've passed the 1st test for the 'get_columns_in_range' function implementation :-)")    


# In[27]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.b. - Test 2 (0.5 points)")
print ("\t--->Testing the implementation of 'get_columns_in_range' ...")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    df_ranged = get_columns_in_range(df_chocolate, 4, 6)
    num_of_rows    = get_number_of_rows(df_ranged)
    num_of_columns = get_number_of_columns(df_ranged)
    print ('Num of rows=%d' %(num_of_rows))
    print ('Num of columns=%d' %(num_of_columns))
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))
    
assert num_of_columns==2, "Wrong number of columns"

print ("Good Job!\nYou've passed the 2nd test for the 'get_columns_in_range' function implementation :-)")   

print ('\nSub dataframe:')
df_ranged


# In[28]:


# 3.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 3.b. - Test 3 (0.6 points)")
print ("\t--->Testing the implementation of 'get_columns_in_range' ...")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    df_ranged_rows = get_rows_in_range(df_chocolate, 1310, 1320)
    df_ranged = get_columns_in_range(df_ranged_rows, 4, 6)
    num_of_rows    = get_number_of_rows(df_ranged)
    num_of_columns = get_number_of_columns(df_ranged)
    print ('Num of rows=%d' %(num_of_rows))
    print ('Num of columns=%d' %(num_of_columns))
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))
    
assert num_of_rows==10 and num_of_columns==2, "Wrong number of rows or columns"

print ("Good Job!\nYou've passed the 3rd test for the 'get_columns_in_range' function implementation :-)")   

print ('\nSub dataframe:')
df_ranged


# ### 4. Conditional Selection of data:

# #### 4.a. Select rows by cell value

# ### Instructions
# <u>method name</u>: <b>select_rows_by_cell_val</b>
# <pre>The following is expected:
# --- Complete the 'select_rows_by_cell_val' function to return only the rows 
#     in a pandas dataframe, given in the parameter 'dataframe', which 
#     match the 'matching_val' value of in the cell in the 'col_name' column.
# </pre>

# In[29]:


# 4.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def select_rows_by_cell_val(dataframe, col_name, matching_val):
    return dataframe.loc[dataframe[col_name]==matching_val]


# In[30]:


# 4.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:

file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
df_cocoa = load_csv(file_name)
col_name_1, col_val_1 = 'Bean Type', 'Forastero'
col_name_3, col_val_3 = 'Cocoa Percent', '100%'
df_rows_pass_cond_1 = select_rows_by_cell_val(df_cocoa, col_name_1, col_val_1)
df_rows_pass_cond_1_3 = select_rows_by_cell_val(df_rows_pass_cond_1, col_name_3, col_val_3)
###
### YOUR CODE HERE
###


# In[31]:


# 4.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# Add your additional tests here if needed:

###
### YOUR CODE HERE
###


# In[32]:


# 4.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4.a. - Test 1 (0.2 points) - Sanity")
print ("\t--->Testing the implementation of 'select_rows_by_cell_val' ...\n")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    df_passes_cond_1 = select_rows_by_cell_val(df_chocolate, 'Bean Type', 'Forastero')
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))

print ("Good Job!\nYou've passed the 1st test for the 'select_rows_by_cell_val' function implementation :-)")   


# In[33]:


# 4.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4.a. - Test 2 (0.5 points)")
print ("\t--->Testing the implementation of 'select_rows_by_cell_val' ...\n")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    df_passes_cond_1 = select_rows_by_cell_val(df_chocolate, 'Bean Type', 'Forastero')
    num_of_rows    = get_number_of_rows(df_passes_cond_1)
    num_of_columns = get_number_of_columns(df_passes_cond_1)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))
    
assert num_of_rows==87 and num_of_columns==9, "Wrong number of rows or columns"

print ("Good Job!\nYou've passed the 2nd test for the 'select_rows_by_cell_val' function implementation :-)")   

print ('First few rows of the dataframe passing 1st condition:')
df_passes_cond_1.head()


# In[34]:


# 4.a.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4.a. - Test 3 (0.5 points)")
print ("\t--->Testing the implementation of 'select_rows_by_cell_val'")
print ("\t\t====> Full grading test - the following test can not be seen before submission ...")

###
### AUTOGRADER TEST - DO NOT REMOVE
###


# #### 4.b. Select rows with value in given range

# ### Instructions
# <u>method name</u>: <b>select_rows_w_vals_in_range</b>
# <pre>The following is expected:
# --- Complete the 'select_rows_w_vals_in_range' function to return only the rows 
#     in a pandas dataframe, given in the parameter 'dataframe', for which 
#     the values of some column given in the 'col_name' parameter, are 
#     between the 'lower_range' and 'higher_range' (inculding lower_range and higher_range)
# </pre>

# In[35]:


# 4.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: ANSWER 

def select_rows_w_vals_in_range(dataframe, col_name, lower_range, higher_range):
    return dataframe[(dataframe[col_name]>=lower_range) & (dataframe[col_name]<=higher_range)]


# In[36]:


# 4.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation)
# === CODE TYPE: SELF TESTING
# Use the following code to test your implementation:

file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
df_cocoa = load_csv(file_name)
col_name_1, col_val_1 = 'Bean Type', 'Forastero'
col_name_2, col_lower_val_2, col_upper_val_2 = 'Review Date', 2015, 2017
df_rows_pass_cond_1 = select_rows_by_cell_val(df_cocoa, col_name_1, col_val_1)
df_rows_pass_cond_1_2 = select_rows_w_vals_in_range(df_rows_pass_cond_1, col_name_2, col_lower_val_2, col_upper_val_2)
###
### YOUR CODE HERE
###


# In[37]:


# 4.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run after implementation, if used)
# === CODE TYPE: SELF TESTING
# Add your additional tests here if needed:
print(get_number_of_rows(df_rows_pass_cond_1_2))
df_rows_pass_cond_1_2


# In[38]:


# 4.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4.b. - Test 1 (0.2 points) - Sanity")
print ("\t--->Testing the implementation of 'select_rows_w_vals_in_range' ...")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    df_passes_cond_2 = select_rows_w_vals_in_range(df_chocolate, 'Review Date', 2015, 2017)
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))

print ("Good Job!\nYou've passed the 1st test for the 'select_rows_w_vals_in_range' function implementation :-)")       


# In[39]:


# 4.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4.b. - Test 2 (0.5 points)")
print ("\t--->Testing the implementation of 'select_rows_by_cell_val' and 'select_rows_w_vals_in_range' ...")

try:
    file_name = 'data' + os.sep + 'flavors_of_cacao.csv'
    df_chocolate = load_csv(file_name)
    df_passes_cond_1 = select_rows_by_cell_val(df_chocolate, 'Bean Type', 'Forastero')
    df_passes_cond_1_2 = select_rows_w_vals_in_range(df_passes_cond_1, 'Review Date', 2015, 2017)
    num_of_rows    = get_number_of_rows(df_passes_cond_1_2)
    num_of_columns = get_number_of_columns(df_passes_cond_1_2)
    print ('Num of rows=%d' %(num_of_rows))
    print ('Num of columns=%d' %(num_of_columns))
except Exception as e:
    print ('You probably have a syntax error, we got the following exception:')
    print (str(e))

assert num_of_rows==14 and num_of_columns==9, "Wrong number of rows or columns passing conditions 1, 2"

print ("Good Job!\nYou've passed the 2nd test for the 'select_rows_by_cell_val' and 'select_rows_w_vals_in_range' function implementation :-)")       

print ('First few rows of the dataframe passing the conditions:')
df_passes_cond_1_2.head()


# In[40]:


# 4.b.
# ------------>>>>>>>> RUN THIS CODE CELL <<<<<<<<------------
# --------  (run only)
# === CODE TYPE: GRADED TEST 

print ("Part 4.b. - Test 3 (0.6 points)")
print ("\t--->Testing the implementation of 'select_rows_by_cell_val' and 'select_rows_w_vals_in_range' ...")
print ("\n\t====> Full grading test - the following test can not be seen before submission")

###
### AUTOGRADER TEST - DO NOT REMOVE
###


# In[ ]:




