#!/usr/bin/env python
# coding: utf-8

# ## About Dataset

# *** Context ***
# 
# The Western African Ebola virus epidemic (2013–2016) was the most widespread outbreak of Ebola virus disease (EVD) in history
# Causing major loss of life and socioeconomic disruption in the region, mainly in Guinea, Liberia, and Sierra Leone.
# The ** first cases** were recorded in Guinea in December 2013;
# Later, the disease spread to neighboring Liberia and Sierra Leone, with minor outbreaks occurring elsewhere.
# It caused significant mortality, with the case fatality rate reported which was initially considered, while the rate among hospitalized patients was 57–59%
# The final numbers 28,616 people, including 11,310 deaths, for a case-fatality rate of 40%. ***

# In[4]:


import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[5]:


# Pandas is a software library written for the Python programming language for data manipulation and analysis.
import pandas as pd
# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays
import numpy as np
# Matplotlib is a plotting library for python and pyplot gives us a MatLab like plotting framework. We will use this in our plotter function to plot data.
import matplotlib.pyplot as plt
#Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics
import seaborn as sns
# Preprocessing allows us to standarsize our data
from sklearn import preprocessing
# Allows us to split our data into training and testing data
from sklearn.model_selection import train_test_split
# Allows us to test parameters of classification algorithms and find the best one
from sklearn.model_selection import GridSearchCV
# Logistic Regression classification algorithm
from sklearn.linear_model import LogisticRegression
# Support Vector Machine classification algorithm
from sklearn.svm import SVC
# Decision Tree classification algorithm
from sklearn.tree import DecisionTreeClassifier
# K Nearest Neighbors classification algorithm
from sklearn.neighbors import KNeighborsClassifier


# In[6]:


def plot_confusion_matrix(Y,Y_predict):
    "this function plots the confusion matrix"
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(Y, Y_predict)
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax); #annot=True to annotate cells
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix'); 
    ax.xaxis.set_ticklabels(['DEATH_EVENT=1','DEATH_EVENT=0']); ax.yaxis.set_ticklabels(['DEATH_EVENT=1','DEATH_EVENT=0']) 
    plt.show() 


# In[17]:


#load the ebola_outbreak dataset into pandas
ebola_df = pd.read_csv(r'C:\Users\user\Downloads\ebola_2014_2016_clean.csv')


# In[18]:


ebola_df


# ### Data Cleaning

# In[19]:


#evaluate missing data.
missing_data = ebola_df.isnull()
missing_data.head(5)


# In[20]:


#count missing value per column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 


# In[13]:


# Handle missing values by filling them with the mean value of column
ebola_df.fillna(ebola_df.mean(), inplace=True)


# In[21]:


#calculate the mean of the column having missing data
mean=ebola_df["Cumulative no. of confirmed, probable and suspected cases"].mean()


# In[22]:


#replace NaN with mean
ebola_df["Cumulative no. of confirmed, probable and suspected cases"].replace(np.nan, mean)


# Now we have a dataset with no missing data

# ### Data Wrangling

# In[25]:


ebola_df.dtypes


# In[72]:


ebola_df["Cumulative no. of confirmed, probable and suspected deaths"].astype("float")


# In[73]:


ebola_df["Cumulative no. of confirmed, probable and suspected cases"].astype("float")


# ### Explorative Data Analysis

# In[74]:


ebola_df.shape


# In[75]:


ebola_df.columns


# In[76]:


ebola_df.keys()


# In[77]:


ebola_df.describe(include="all")


# In[78]:


ebola_df.info()


# In[95]:


df=ebola_df['Country'].value_counts().to_frame()
df[:]


# In[86]:


# Countplot
sns.countplot(data = ebola_df, y ="Country" ,label ="count")


# In[65]:


plt.figure(figsize=(12,6))
sns.distplot(ebola_df['Cumulative no. of confirmed, probable and suspected cases'], hist=True, bins=30, color='blue')
plt.xlabel('maximum suspected cases')
plt.ylabel('Frequency')
plt.title('Distribution of suspected cases', fontsize=10)


# In[121]:


plt.figure(figsize=(12,6))
sns.distplot(ebola_df['Cumulative no. of confirmed, probable and suspected deaths'], hist=True, bins=30, color='blue')
plt.xlabel('maximum suspected cases')
plt.ylabel('Frequency')
plt.title('Distribution of suspected cases', fontsize=10)


# In[91]:


# A function to Extract years from the date 
year=[]
def Extract_year():
    for i in ebola_df["Date"]:
        year.append(i.split("-")[0])
    return year
Extract_year()
ebola_df['Date'] = year
ebola_df.head()
    


# In[96]:


df=ebola_df[['Country','Date']].value_counts().to_frame()
df[:]


# In[104]:


# Plot a scatter point chart with x axis to be Payload and y axis to be the Orbit, and hue to be the class value
ebola_df.plot(kind='scatter', y='Country', x='Date', figsize=(10, 6), color='red')

plt.title('Countries per Year')
plt.xlabel('Date')
plt.ylabel('Country')

plt.show()


# In[113]:


sns.catplot(y="Country", x="Date", data=ebola_df, aspect = 5)
plt.xlabel("Cumulative no. of confirmed, probable and suspected cases",fontsize=20)
plt.ylabel("Country",fontsize=20)
plt.show()


# In[116]:


###Let's create a bar chart for the rate of death per country
ebola_df.groupby("Country").mean()['Cumulative no. of confirmed, probable and suspected deaths'].plot(kind='bar')
plt.xlabel("Countries",fontsize=20)
plt.ylabel("Suspected Rate of death",fontsize=20)
plt.title('Suspected Rate of death per Country')
plt.show()


# In[117]:


###Let's create a bar chart for the rate of death per country
ebola_df.groupby("Country").mean()['Cumulative no. of confirmed, probable and suspected cases'].plot(kind='bar')
plt.xlabel("Countries",fontsize=20)
plt.ylabel("Suspected Rate of cases",fontsize=20)
plt.title('Suspected Rate of cases per Country')
plt.show()


# In[130]:


plt.figure(figsize=(18,10))
sns.boxplot(data = ebola_df, orient = "v", palette = "Set1")


# In[ ]:




