#!/usr/bin/env python
# coding: utf-8

# In[93]:


import pandas as pd
import numpy as np


# In[94]:


df = pd.read_csv(r'C:\Users\user\Downloads\India-Tourism-Statistics-2021-Table-5.2.3.csv')


# In[95]:


df.tail()


# In[96]:


df.shape


# In[97]:


missing_data = df.isnull()
missing_data.head(5)


# In[98]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 


# In[99]:


# Handle missing values by filling them with the mean value of each column
df.fillna(df.mean(), inplace=True)


# In[100]:


mean=df["% Growth 2021-21/2019-20-Domestic"].mean()


# In[101]:


df["% Growth 2021-21/2019-20-Domestic"].replace(np.nan, mean)


# In[102]:


mean1=df["% Growth 2021-21/2019-20-Foreign"].mean()


# In[103]:


df["% Growth 2021-21/2019-20-Foreign"].replace(np.nan, mean1)


# In[104]:


missing_data = df.isnull()
missing_data.head(5)


# In[105]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 


# In[106]:


# Reformat data types
df["Circle"] = df["Circle"].astype(object)


# In[107]:


df.dtypes


# In[108]:


df.describe(include="all")


# In[109]:


df=df.sample(n=100, random_state=42)


# In[110]:


df.shape


# In[111]:


df.head()


# In[112]:


df.nunique()


# In[137]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[114]:


df.corr()


# In[115]:


# Heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[116]:


df['Circle'].unique()


# In[117]:


##Use the method value_counts to count the number of each unique values, using the method .to_frame() to convert it to a dataframe.
df['Circle'].value_counts().to_frame()


# In[118]:


df['Circle'].value_counts()


# In[119]:


#Use the function boxplot in the seaborn library to determine whether houses with a waterfront view or without a waterfront view have more price outliers.
get_ipython().run_line_magic('matplotlib', 'inline')
width =14
height = 10
plt.figure(figsize=(width, height))
sns.boxplot(x=df['Domestic-2019-20'],y=df['Circle'])
plt.show()


# In[142]:


grouped = df.groupby('Circle')


# In[121]:


# Calculate the mean of each group
grouped_mean = grouped['Domestic-2019-20'].mean()

print(grouped_mean)


# In[122]:


# Calculate the unique of each group
grouped_unique = grouped['Domestic-2019-20'].unique()

print(grouped_unique)


# In[123]:


df['% Growth 2021-21/2019-20-Domestic'].unique()


# In[124]:


##Use the method value_counts to count the number of each unique values, using the method .to_frame() to convert it to a dataframe.

df['% Growth 2021-21/2019-20-Domestic'].value_counts().to_frame()


# In[125]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[126]:


# Countplot
sns.countplot(y="Circle", data=df)
plt.show()


# In[134]:


import matplotlib as plt
from matplotlib import pyplot

plt.pyplot.hist(df['Domestic-2019-20'],bins=10)
plt.pyplot.ylabel('Domestic-2019-20')
plt.pyplot.xlabel("count")
plt.pyplot.title("Domestic-2019-20_bins")


# In[138]:


# Scatter plot
sns.scatterplot(y="Circle", x="Domestic-2019-20", data=df)
plt.show()


# In[139]:


# Boxplot
sns.boxplot(y="Circle", x="Domestic-2019-20", data=df)
plt.show()


# In[140]:


# Plot a line chart 
plt.plot(df["Domestic-2019-20"],df["Circle"])
plt.xlabel("Domestic-2019-20")
plt.ylabel("Circle")
plt.show()


# In[ ]:




