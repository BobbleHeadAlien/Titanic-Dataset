#Importing useful libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Reading the file
train = pd.read_csv('titanic_train.csv')
train.head()

#To find missing data
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Visualizaing various datas

#Count plot of survived column
sns.set_style('whitegrid')
sns.countplot(x='Survived',data=train,palette='RdBu_r')

#Count plot of survived column seperated by sex.
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')

#Count plot of survived column seperated by class.
sns.set_style('whitegrid')
sns.countplot(x='Survived',hue='Pclass',data=train,palette='rainbow')

#Plotting age distribution plot.
sns.distplot(train['Age'].dropna(),kde=False,color='darkred',bins=30)

#Plotting histogram of fare.
train['Fare'].hist(color='green',bins=40,figsize=(8,4))

#Code to clean the dataset.
plt.figure(figsize=(12, 7))
sns.boxplot(x='Pclass',y='Age',data=train,palette='winter')

#Function to fill up missing information in the age column.
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):

        if Pclass == 1:
            return 37

        elif Pclass == 2:
            return 29

        else:
            return 24

    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age,axis=1)

#plotting heatmap to check missing data correction.
sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')

#Dropping cabin data.
train.drop('Cabin',axis=1,inplace=True)
train.head()

#Converting Catagorical features in the 'sex' and 'embarked' columns.
sex = pd.get_dummies(train['Sex'],drop_first=True)
embark = pd.get_dummies(train['Embarked'],drop_first=True)

#Dropping the non-numerical columns from the database.
train.drop(['Sex','Embarked','Name','Ticket'],axis=1,inplace=True)

#concatinating the columns and databse.
train = pd.concat([train,sex,embark],axis=1)
train.head()

#Data is ready to be used for any machine learning algorithm!

#I will be using logistic regression for fitting the model.
#Importing and spliting the data.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1), 
                                                    train['Survived'], test_size=0.30, 
                                                    random_state=101)

#Training and predicting:
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train

predictions = logmodel.predict(X_test)

#Evauating the fit model.
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

#Classification report shows that the model is a good fit for the data.
             


