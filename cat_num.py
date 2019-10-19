##Analyzing and encoding categorcal variables

##Import packages
import pandas as pd 
import numpy as np
from prepro import missingval_report
data = pd.read_csv('/Users/vvithala/Documents/train.csv', sep=',')
data.columns = [x.lower() for x in data.columns]
data.head()

#Filling missing values is essential and should be done before encoding categorical features.
######Treating Missing Data
#Dropping columns which have more than 50% missing values and insignificant columns - Option to drop columns with 100% unique and more than 50% missing values

data1,miss=missingval_report(data,drop_col=True,drop_id=True,target='survival')



##Impute missing values for numeric and categorical features.
##If miss_% is less than we drop those rows - It is a dirty approach as we tend to lose data.
##By using back fill and forward fill if we need to impite missing values by filling them with previous or next values
##Simple Impute mainly replaces the missing values with the mean,median or mode
##
from sklearn.preprocessing import SimpleImputer
from fancyimpute import KNN
data.fillna(method='bfill', inplace=True)
data.fillna(method='ffill', inplace=True)

#def impute_miss(df,method='simple',collist=)





###Now that we have analyzed our data with respect to missing values we split the data into train test split , so that we can impute missing values,
###Scale numeric data , treat outliers, deal with correlated variables.

def selectionlist(methods=[]):
  menu={option_number:action for option_number,action in enumerate(methods)}
  #print(menu)
  for entry in menu:
      print(entry, menu[entry])
  option_input=int(input('Enter option number: '))
  method1=menu[option_input]
  return method1

selectionlist(methods)


##Scalers
scaler_options=['standard','robust','minmax','normal']
impute_options = ['bfill', 'ffill', 'simple', 'knn']

##Supervised Learning
regression_options=['simple_linear','multiple_linear','svm','trees','random_forest']
classification_options=['trees','svm','logistic','naive-bayes','']
