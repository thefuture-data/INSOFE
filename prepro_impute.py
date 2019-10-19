
###Script to include functions that 
##1.) Automatically impute values based on hard rules
##2.) Documents various methods to impute missing variables
##3.) Dealing with numeric and categorical missing values

##***Human Intervention - Selecting appropriate imputing strategy.

##Notes for Imputation
## 3 types of missing data - MCAR : 'missing completely at random' , MAR : 'missing at random' , NMAR: 'not missing at random'
# -> 6 ways we can deal with missing values

#1.) Do nothing , this is the simplest as we let the algorithm handle missing data , how this happens is , algorthims like - 
#XGBoost is able to learn the best imputation of missing values in order to reduce training loss. LGBM has the option of ignoring missing values.

#2.) Impute values using median or mode.
##True for continous numeric variables, Advantages - easy & fast , works well with small numeric datasets ** 
# Disadvantages - Doesnt factor correlations between variables , Poor results on encoded categorical variables (NO!), not accurate and no accountability for uncertainity.

#3.) Imputing missing values with specific value or most frequent (mode) in the case of categorical variables . - Works very well for categorical variables 
##Disadvanatges being it doesnt factor correlation between features and it can introduce bias in the data

#4.) Imputation using k-NN - k-NN algorithm is an algorithm that is used for simple classification. It is based on the assumption that points that are similar in nature
##are generally nearby each other. It uses the feature similarity to predict values of new points.
#Library we can use is impyute - How this works is it generates basic mean impute and constructs a KD-Tree, Then it uses the KD-Tree to find the 'k' nearest neighbors.
##After that we take the weighted average of the nearest neighbors to fill missing values.

###Example Code for implementing k-nn imputation
import sys
from impyute.imputation.cs import fast_knn
sys.setrecursionlimit(100000)  # Increase the recursion limit of the OS
# start the KNN training
imputed_training = fast_knn(train.values, k=30)


from impyute.imputation.cs import fast_knn
import sys
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from math import sqrt
import random
import numpy as np
random.seed(0)

#Fetching the dataset
