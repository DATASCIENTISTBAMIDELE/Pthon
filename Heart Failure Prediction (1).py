#!/usr/bin/env python
# coding: utf-8

# ### About Dataset

# Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worlwide.
# Heart failure is a common event caused by CVDs and this dataset contains 12 features that can be used to predict mortality by heart failure.
# 
# Most cardiovascular diseases can be prevented by addressing behavioural risk factors such as tobacco use, unhealthy diet and obesity, physical inactivity and harmful use of alcohol using population-wide strategies.
# 
# People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.

# In[36]:


#importing all necessary libraries
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')


# In[2]:


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


# ### Data Extraction

# In[3]:


df = pd.read_csv(r'C:\Users\user\Downloads\heart_failure_clinical_records_dataset.csv')


# In[4]:


df.head(20)


# In[5]:


df.shape


# ### Data Cleaning

# In[6]:


#evaluate missing data.
missing_data = df.isnull()
missing_data.head(5)


# In[7]:


#count missing value per column
for column in missing_data.columns.values.tolist():
    print(column)
    print (missing_data[column].value_counts())
    print("") 


# ### Explorative Data Analysis

# In[8]:


#checking the datatype of each column
df.dtypes


# In[9]:


df.columns


# In[10]:


df.describe(include="all")


# In[11]:


df.info()


# In[12]:


df['DEATH_EVENT'].value_counts().to_frame()


# In[13]:


#checking the correllation with the target
df.corr()['DEATH_EVENT'].sort_values()


# In[14]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[15]:


# Heatmap
sns.heatmap(df.corr(), annot=False)
plt.show()


# In[21]:


#Featuring the data
X = np.asarray(df[['serum_creatinine', 'age', 'ejection_fraction', 'platelets',"serum_sodium","creatinine_phosphokinase"]])
X[0:5]


# In[22]:


Y = np.asarray(df['DEATH_EVENT'])
Y[0:5]


# In[20]:


fig = plt.figure(figsize =(10, 7)) 

plt.boxplot(df)

plt.title("Box Plot")

plt.show()


# In[25]:


#Creating a new death_rate column  where 1 = Alive and 0 = Died.
def set_name(DEATH_EVENT):
    if DEATH_EVENT == 1:
        return 'Alive'
    else:
        return 'Died'
    
    
df['DEATH_Rate'] = df['DEATH_EVENT'].apply(set_name)
df


# In[26]:


sns.countplot(data = df, x ='DEATH_Rate' ,label ="count")


# In[31]:


#Creating a new death_rate column  where 1 = Alive and 0 = Died.
def set_name(sex):
    if sex == 1:
        return 'Male'
    else:
        return 'Female'
    
    
df['Gender'] = df['sex'].apply(set_name)
df


# In[32]:


sns.countplot(data = df, x ='Gender' ,label ="count")


# In[33]:


Gender_Death = sns.countplot(x='DEATH_EVENT', data = df, hue='sex')
plt.legend(['Female', 'Male'])
Gender_Death.set_title("Death Rate per Gender")
Gender_Death.set_xticklabels(['Alive', 'Died'])
plt.xlabel("");


# In[37]:


plt.figure(figsize=(15,8))
sns.distplot(df['platelets'], hist=True, bins=30, color='blue')
plt.xlabel('count')
plt.ylabel('Numbers')
plt.title('Platelets distribution', fontsize=15)


# In[34]:


fig = plt.figure(figsize =(10, 7)) 

plt.boxplot(df['platelets'])

plt.title("Box Plot")

plt.show()


# In[51]:


df["age"].value_counts().to_frame()


# In[53]:


###Let's create a bar chart for the rate of death per age
df.groupby("age").mean()['DEATH_EVENT'].plot(kind='bar')
plt.xlabel("age",fontsize=20)
plt.ylabel("Rate of deaths",fontsize=20)
plt.title('Rate of Death per Age')
plt.show()


# In[58]:


sns.scatterplot(x="platelets", y="age", hue="sex", data=df)

# Show plot
plt.show()


# In[59]:


sns.scatterplot(x="platelets", y="age", hue="DEATH_EVENT", data=df)

# Show plot
plt.show()


# ### TRAINING OF MODEL

# In[ ]:





# In[60]:


#we normalize the dataset:

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# In[61]:


#we split the dataset into 80% Training and 20% Testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  Y_train.shape)
print ('Test set:', X_test.shape,  Y_test.shape)


# In[62]:


Y_test[:]


# In[63]:


Y_train[:]


# ### LOGISTIC REGRESSION

# In[64]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,Y_train)
LR


# In[65]:


##LogisticRegression(C=0.01, solver='liblinear')
##Now we can predict using our test set:

yhat = LR.predict(X_test)
yhat


# ### Evaluation

# In[66]:


yhat_prob = LR.predict_proba(X_test)
yhat_prob[0:5]


# In[67]:


from sklearn.metrics import jaccard_score
jaccard_score(Y_test, yhat,pos_label=0)


# In[68]:


from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(Y_test, yhat, labels=[1,0]))


# In[21]:


# Compute confusion matrix
cnf_matrix = confusion_matrix(Y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)


# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['DEATH_EVENT=1','DEATH_EVENT=0'],normalize= False,  title='Confusion matrix')


# In[69]:


print(classification_report(Y_test,yhat))


# ### HYPERPARAMETER TUNING FOR LR

# In[70]:


parameters ={"C":[0.01,0.1,1],'penalty':['l2'], 'solver':['lbfgs']}# l1 lasso l2 ridge


# In[71]:


lr=LogisticRegression()


# In[72]:


logreg_cv = GridSearchCV(lr, parameters, cv=10)


# In[73]:


logreg_cv=logreg_cv.fit(X_train, Y_train)


# In[74]:


yhat=logreg_cv.predict(X_test)
yhat


# ### Evaluation

# In[75]:


from sklearn.metrics import jaccard_score
jaccard_score(Y_test, yhat,pos_label=0)


# In[76]:


# Print the tuned parameters and score
print("tuned hpyerparameters :(best parameters) ",logreg_cv.best_params_)
print("accuracy :",logreg_cv.best_score_)


# In[79]:


from sklearn.metrics import accuracy_score

yhat=logreg_cv.predict(X_test)
print("LOGISTIC REGRESSION's Accuracy: ", logreg_cv.score(X_test, Y_test))
print("Accuracy: ", accuracy_score(Y_test, yhat))


# In[80]:


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


# In[81]:


yhat=logreg_cv.predict(X_test)
plot_confusion_matrix(Y_test, yhat)


# In[82]:


print(classification_report(Y_test,yhat))


# ### SUPPORT_VECTOR_MACHINE

# In[83]:


from sklearn import svm
df= svm.SVC(kernel='rbf')
df=df.fit(X_train, Y_train)


# In[84]:


yhat = df.predict(X_test)
yhat [:]


# In[87]:


print("DecisionTree's Accuracy: ",df.score(X_test,Y_test))
print("DecisionTree's Accuracy: ", accuracy_score(Y_test, yhat))


# In[88]:


jaccard_score(Y_test, yhat,pos_label=0)


# In[89]:


yhat=df.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# In[100]:


#Another Parameter
from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train,Y_train)


# In[101]:


yhat = svm.predict(X_test)
yhat [:]


# In[102]:


print(classification_report(Y_test,yhat))


# ### Using More Parameters

# In[103]:


parameters = {'kernel':('linear','rbf', 'sigmoid'),
              'C': np.logspace(-3, 3, 5),
              'gamma':np.logspace(-3, 3, 5)}
svm = SVC()


# In[104]:


svm_cv = GridSearchCV(svm, parameters, cv=10)


# In[105]:


svm_cv=svm_cv.fit(X_train, Y_train)


# In[106]:


yhat=svm_cv.predict(X_test)
yhat


# In[107]:


print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)

print("tuned hpyerparameters :(best parameters) ",svm_cv.best_params_)
print("accuracy :",svm_cv.best_score_)


# In[108]:


svm_cv.score(X_test, Y_test)


# In[109]:


jaccard_score(Y_test, yhat, pos_label=0)


# In[110]:


print("best :",tree_cv.best_score_)
print("DecisionTree's Accuracy: ",tree_cv.score(X_test,Y_test))
print("DecisionTree's Accuracy: ", accuracy_score(Y_test, yhat))


# In[111]:


'rbf','poly'


# In[112]:


yhat=svm_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# ### DECISION TREE

# In[113]:


parameters = {'criterion': ['gini', 'entropy'],
     'splitter': ['best', 'random'],
     'max_depth': [2*n for n in range(1,10)],
     'max_features': ['auto', 'sqrt'],
     'min_samples_leaf': [1, 2, 4],
     'min_samples_split': [2, 5, 10]}

tree = DecisionTreeClassifier()


# In[114]:


tree_cv = GridSearchCV(tree, parameters, cv=10)


# In[115]:


tree_cv=tree_cv.fit(X_train,Y_train)


# In[116]:


yhat=tree_cv.predict(X_test)
yhat


# In[117]:


print ('Predicted:',yhat [:])
print ('existing.:',Y_test [:])


# In[118]:


from sklearn import metrics
import matplotlib.pyplot as plt
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(Y_test, yhat))


# In[119]:


tree_cv.best_estimator_


# In[120]:


print("tuned hpyerparameters :(best parameters) ",tree_cv.best_params_)
print("accuracy :",tree_cv.best_score_)


# In[121]:


from sklearn.metrics import jaccard_score
jaccard_score(Y_test, yhat,pos_label=0)


# In[122]:


print("DecisionTree's Accuracy: ",tree_cv.score(X_test,Y_test))
print("DecisionTree's Accuracy: ", metrics.accuracy_score(Y_test, yhat))


# In[123]:


yhat = tree_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# ### K-NEAREST NIEGHBOUR. KNN

# In[124]:


parameters = {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              'algorithm': ['auto'],
              'p': [1,2]}

KNN = KNeighborsClassifier()


# In[125]:


knn_cv = GridSearchCV(KNN, parameters, cv=10)


# In[126]:


knn_cv.fit(X_train,Y_train)


# In[127]:


yhat = knn_cv.predict(X_test)
yhat


# In[131]:


print("best :",tree_cv.best_score_)
print("Test set Accuracy: ", metrics.accuracy_score(Y_test, yhat))


# In[132]:


from sklearn.metrics import jaccard_score
jaccard_score(Y_test, yhat,pos_label=0)


# In[133]:


yhat = knn_cv.predict(X_test)
plot_confusion_matrix(Y_test,yhat)


# In[136]:


print(classification_report(Y_test,yhat))


# In[137]:


import matplotlib.pyplot as plt

scores = [0.75, 0.76, 0.75, 0.75]
best_score = max(scores)

plt.plot(np.arange(len(scores)), scores, '-o', label='Accuracy Score')
plt.plot(np.argmax(scores), best_score, 'ro', markersize=10, label='Best Score')

plt.title('Accuracy Score and Best Score')
plt.xlabel('Models')
plt.ylabel('Accuracy Score')
plt.legend()
plt.show()


# In[138]:


# Create logistic regression, decision tree, and random forest objects and perform grid search
logreg = LogisticRegression()
grid_logreg = GridSearchCV(logreg, {"C": [0.1, 1, 10, 100]})
grid_logreg.fit(X, Y)

dt = DecisionTreeClassifier()
grid_dt = GridSearchCV(dt, {"max_depth": [1, 2, 3, 4, 5]})
grid_dt.fit(X, Y)


# Plot the accuracy score for all CV folds and the best score for each classifier
scores_logreg = np.array(grid_logreg.cv_results_['mean_test_score'])
best_score_logreg = grid_logreg.best_score_
scores_dt = np.array(grid_dt.cv_results_['mean_test_score'])
best_score_dt = grid_dt.best_score_


plt.plot(np.arange(len(scores_logreg)), scores_logreg, '-o', label='Logistic Regression')
plt.plot(np.argmax(scores_logreg), best_score_logreg, 'ro', markersize=10, label='Best score Logistic Regression')
plt.plot(np.arange(len(scores_dt)), scores_dt, '-o', label='Decision Tree')
plt.plot(np.argmax(scores_dt), best_score_dt, 'ro', markersize=10, label='Best score Decision Tree')


plt.title('Accuracy score and best score for different classifiers')
plt.xlabel('CV fold')
plt.ylabel('Accuracy score')
plt.legend()
plt.show()


# In[ ]:




