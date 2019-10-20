##The main aim of machine learning is to make a model that is as generalized as possible and we can predict target for unseen data
##as accurately as possible

##By balancing the bias-variance trade-off we ensure that we dont overfit the data by having high accuracy for train and low accuracy for test and vice verca.
##When we find that the model is overfitting we say that there is high variance and when we are underfitting we say that there is high bias.
## train_error = low and test_error = high , we are overfitting and the model has high variance.
## train_error = high and test_error = low , we are underfitting and the model has high bias.
## train_error = high and test_error = high , the model is not able to capture anything and we have high bias and variance.


##Tricks while we do read_csv.

##1.) In the cases when the file is too large - we use the arguement nrow=5 to see only the first 5 rows and by which we can avoid 
##cases where we read the file with the wrong delimiter
##2.) We can use usecols=[] to select specific columns
##3.) We can explicitly specify data-types by using the dtype = {'c1':'str',c2':'str'}
##4.) We can also subset data based on their data types by using the select_dtypes(include=['int64','float64'])
##5.) We use the copy command so that a the original object is not altered by the copied object.
##6.) By using the df.map we can perform easy transformations.
##7.) By using the value_counts method we can get the freq of unique values in a column of a df.(value counts can only be used for columns/series)
## and by adding the normalize arguement we can get the ratio of freq of unique values in a column where the sum = 1 (useful for checking class imbalance) and also we can set drop_na=False to include missing values

##Regularization is a technique that controls the learning of the model from the data.
import pandas as pd
import numpy as np

data = pd.read_csv('train.csv', sep=',')
data = data.reindex(np.random.permutation(data.index))
from sklearn.model_selection import train_test_split
target='Survived'

train_size=0.8
random_state=420
features=data.drop([target],axis=1)

data.Survived.value_counts(normalize=True)
##Check for duplicated rows
data.duplicated().mode()

def target_features(df,target):
    target1=df[target]
    features=df.drop(columns=[target],axis=1).copy()
    return [features,target1]

def train_test(features, target, train_size=0.8, random_state=420):
    xtrain,xtest,ytrain,ytest = train_test_split(features,target,train_size=train_size,random_state=random_state)
    return [xtrain, xtest, ytrain, ytest]

features_targ = target_features(data,target)
data_split=train_test(features_targ[0],features_targ[1])

##Note everytime you drop do a copy!
