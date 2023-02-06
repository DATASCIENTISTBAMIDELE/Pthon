#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[3]:


df = pd.read_excel(r'C:\Users\user\Desktop\Flipkart laptops_info.xls')


# In[4]:


df.head(20)


# In[5]:


df.shape


# In[6]:


missing_data = df.isnull()
missing_data.head(5)


# In[7]:


for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 


# In[8]:


df=df.drop(['Unnamed: 10', 'Unnamed: 11', 'Unnamed: 12', 'Unnamed: 13', 'Unnamed: 14', 'Unnamed: 15', 'Unnamed: 16', 'Unnamed: 17', 'Unnamed: 18'], axis=1)
df.head(20)


# In[9]:


df.shape


# In[10]:


df['ProductName'].unique()


# In[11]:


df.nunique()


# In[12]:


df.dropna(subset=['Storage'], axis=0, inplace=True)


# In[13]:


df.shape


# In[29]:


df.dtypes


# In[32]:


print(df.describe())


# In[17]:





# In[24]:


df=df.sample(n=100, random_state=42)


# In[35]:


df.head()


# In[25]:


df.shape


# In[49]:


df.head()


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[38]:


# Countplot
sns.countplot(x="ProductName", data=df)
plt.show()


# In[27]:


# Histogram
df["Current_Price"].plot.hist()
plt.show()


# In[28]:


# Boxplot
sns.boxplot(x="Current_Price", y="ProductName", data=df)
plt.show()


# In[60]:


# Scatter plot
sns.scatterplot(x="ProductName", y="Current_Price", data=df)
plt.show()


# In[23]:


df.hist(column="Current_Price")


# In[61]:


# Heatmap
sns.heatmap(df.corr(), annot=True)
plt.show()


# In[62]:


df.corr()


# In[ ]:




