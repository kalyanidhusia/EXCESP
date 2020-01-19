#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
get_ipython().run_line_magic('matplotlib', 'inline')
import re
import time
import math
import pytest
import glob


# ### Just to visualise the dimension of data

# In[2]:


import pandas as pd
path_to_file= "/Users/saheeba/Downloads/sorted_978.csv"
sorted_list = pd.read_csv(path_to_file, index_col=False, header=0)
sorted_list.head()


# In[3]:


a=(sorted_list["id"])
print(a)
sorted_list.shape


# In[4]:


filepath= "/Users/saheeba/Downloads/preppi_final.csv"
preppi = pd.read_csv(filepath)
print(preppi.shape)
preppi.head()


# In[5]:


from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer(min_df=1, lowercase=True)

X = vect.fit_transform(sorted_list["id"])
cols_emp = vect.get_feature_names()

X = vect.fit_transform(preppi['prot1'])
cols_desc = vect.get_feature_names()

common_cols_idx = [i for i,col in enumerate(cols_emp) if col in cols_desc]

preppi['Match'] = (X.toarray()[:, common_cols_idx] == 1).any(1)

print(preppi.shape)
preppi.head()


# In[7]:


results = preppi.query("Match==True")
export_csv = results.to_csv (r'/Users/saheeba/Downloads/aligned_result_preppi.csv', index = None, header=True) 
print(results.shape)
results.head()

