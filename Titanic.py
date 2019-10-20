from sklearn.preprocessing import LabelEncoder
import pandas as pd 
import numpy as np
from prepro import get_percent
from prepro import num_cat_data_stats
from prepro import l
from prepro import s
from fancyimpute import KNN
import seaborn as sns

###Workflow - 
##Read Data and check sample.
data = pd.read_csv('/Users/vvithala/Documents/train.csv', sep=',')
data.dtypes
data.columns = [x.lower() for x in data.columns]
data.sample(10)
rows = data.shape[0]
features = data.shape[1]
print('Number of features: ', features)
print('Number of rows : ', rows)
##Missing train DF
missing = data.isna().sum().reset_index().rename(\
    columns={'index': 'feature', 0: 'freq'})
missing['miss_%']=missing.freq.apply(lambda x: get_percent(x,rows))
missing=missing.sort_values(by='miss_%', ascending=False)
miscol=missing[missing['miss_%']>missing['miss_%'].mean()].feature.unique()

##Dropping cabin columns
data=data.drop(columns=['cabin','passengerid'],axis=1)

###Supervised Learning Approach
##Objective : 'From Kaggle'

##Divide data into features and target
target=data['survived'].copy()
features=data.drop(columns=['survived']).copy()

##Divide features according to categorical and numeric data
nuni = features.nunique().reset_index().rename(\
    columns={'index': 'feature', 0: 'freq'})
##Dealing with categorical variables
catcol=l(nuni[nuni['freq']<10].feature.unique())
numcol=l(set(features.columns).difference(set(catcol)))
###Split data into train and validation 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest= train_test_split(features,target,train_size=0.7,random_state=123,shuffle=True)
data.dtypes
##Analyzing numeric data in train data
cat_data = xtrain[catcol]
num_data=xtrain[numcol]

###Analyze numeric train data
cat_data

##There are three ways we will encode categorical records
##Label Encoder - Used mainly for ordinal categories or binary categories
##One Hot Encoding - Not Ordinal categories enables sparse vectors
##It is advisable to use boht one hot encoder and label endcoder depending on the data
###To use label encoder on a df we have to apply it each column, while in one hot encoding we fit the df subset we want to encode
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
def label(df,col=[]):
    le=LabelEncoder()
    df[col]=df[col].apply(lambda col : le.fit_transform(col))

def one_hot(df,col=[]):
    ohe=OneHotEncoder(categorical_features=col,sparse=False)
    return ohe.fit_transform(df)




##Fare Analysis
#fare_analysis=num.Fare.value_counts().reset_index().rename(columns={'index':'Price','Fare':'Freq'})
#fare_analysis['multi_ticket'] = fare_analysis.Freq.apply(lambda x: x > 1)
#fare_single = fare_analysis[fare_analysis['multi_ticket'] == False].copy()
#fare_analysis.describe().T
#num.groupby('Pclass').agg({'Fare': np.mean})

##Analyze cabin , fare , notice
#train.dropna(subset=['Cabin'])
