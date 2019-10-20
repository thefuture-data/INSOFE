
import datawig
import essential_packages

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
##Advantages are that it is more accurate than mean,median,mode (depending on the dataset)
##Disadvantages are that it is computationally expensive and it is sensitive to outliers. (Unlike SVM**)

###Example Code for implementing k-nn imputation
#sys.setrecursionlimit(100000)  # Increase the recursion limit of the OS
# start the KNN training
#imputed_training = fast_knn(train.values, k=30)

##5.) Imputation using multi-variate imputation by chained equation (MICE)
##This type of imputation works by filling missing values multiple times and by doing so uncertainity of the missing values  is 
##measured better 

##Example Code for MICE
from impyute.imputation.cs import mice
# start the MICE training
imputed_training = mice(train.values)

##6.) Imputing using deep neural networks (Datawig)
##This method works really well with numeric and categorical variables . It is a library that learns ML models by using DNN to impute
##missing values. It has support for both CPU and GPU for training
##Advantages are that it is quite accurate compared to other imputation techniques,it can handle categorical data with 'Feature Encoder'
##Disadvatages are that it is slow with large datasets, a requirement is that you need to specify the columns that contain information about the target column
##that will be impyuted

##Example Code for imputation using neural networks

import datawig
df_train, df_test = datawig.utils.random_split(train)

#Initialize a SimpleImputer model
imputer = datawig.SimpleImputer(
    # column(s) containing information about the column we want to impute
    input_columns=['1', '2', '3', '4', '5', '6', '7', 'target'],
    output_column='0',  # the column we'd like to impute values for
    output_path='imputer_model'  # stores model data and metrics
)

train=data_split[0].copy()
##For the above data we will use various imputation methods.
train2=train.copy()
train3=train.copy()
train4=train.copy()


impute_methods=['std','robust','minmax','normal','knn','nn','mice','']
from sklearn.preprocessing import SimpleImputer