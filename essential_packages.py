
##Import essential packages for reading and plotting
import pandas as pd
import numpy as np
import sys
import random
import seaborn as sns


##Datasets
from sklearn.datasets import fetch_california_housing

##Data Prep
from impyute.imputation.cs import fast_knn
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, RobustScaler, Normalizer, MinMaxScaler


##Models
from sklearn.linear_model import LinearRegression
from sklearn import svm,ensemble

##Model Evaluation
from sklearn.metrics import mean_squared_error,mean_absolute_error,multilabel_confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV, train_test_split



