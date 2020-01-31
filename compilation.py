##Import datasets
from sklearn.preprocessing import SimpleImputer
from impyute.imputation.cs import mice
import datawig
import prepro
import essential_packages
import inquirer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet,SGDRegressor
from psutil import virtual_memory


data=pd.read_csv('/Users/vvithala/Downloads/PewDiePie.csv')

target = data['Subscribers'].copy()
features = data.drop(columns=['Subscribers'])
xtrain, xtest, ytrain, ytest = train_test_split(
    features, target, train_size=0.7, random_state=0)


model = LinearRegression()
model.fit(xtrain, ytrain)

predictions = model.predict(xtest)

np.sqrt(np.sum(ytest-predictions))

##5 important steps of data science
#Getting the data
##Cleaning the data
###Exploring the data
####Building the model
#####Presenting the data

##Incorprating the scikit learn cheat sheet

##Inputs: dataset
problem_type = [inquirer.List('problem_type', message="What kind of problem is this?", choices=['Regression', 'Classification', 'Clustering', 'Dimensionality Reduction']), ]
fe_question=[inquirer.List('fe_question',message='Do you want to eliminate features?',choices=['Yes','No']),]
regression_fe_selection = [inquirer.List('regression_selection', message='What Type of Regression would you like to fit', choices=['ElasticNet', 'Lasso', 'All']),]
regression_nfe_selection = [inquirer.List('regression_nfe_selection', message='What Type of Regression would you like to fit', choices=['ElasticNet', 'Lasso', 'SVR', 'EnsembleRegressors'])]
imputation_method = [inquirer.List('imputation_method', message='How would you like to impute the missing values?', choices=['std', 'robust', 'minmax', 'normal', 'knn', 'nn', 'mice']),]
folder_selection=[inquirer.List('working directory',message='Where is the file',choices=['Downloads','Desktop','Documents','Other']),]
problem=inquirer.prompt(folder_selection)
samples=data.shape[0]


##Regression
if samples>100000:
    sgd=True
else:
    sgd=False

while sgd:
    ##fit sgd regression

while !sgd:
    elim=inquirer.prompt(fe_question)
    if elim=='Yes':
        regression=inquirer.prompt(regression_Fe_selection)
        ##fit appropriate models - Write functions for each model





##function structure

##Check Data Cleanliness
##Fit Model 
##Evaluate Model
##Hyperparmater Tuning





 
##Classification


##Clustering


##Dimensionality Reduction


##Web scraping using BeautifulSoup and Requests libraries


##Questions



answers = inquirer.prompt(questions)


def ask_question()


###Data Cleaning

##Data Cleaning (Quality)

##Data Cleaning (Accuracy)

##Data Cleaning (Consistency)

##Uniformity

##The Workflow

##Inspection : Detect unexpected, incorrect, and inconsistent data.
##Cleaning :  Fix or remove the anomalies discovered.
##Verifying : : After cleaning, the results are inspected to verify correctness.
##A report about the changes made and the quality of the currently stored data is recorded.

##Inspection

##Data profiling
data.describe()

##Missing Values
data.isna()

##Visualizations
data.pairplot()

def get_uniqueval_count(df):
    return df.nunique().reset_index().rename(columns={'index':'feature',0:'freq'})


def get_miss_count(df):
    return df.isna().sum().reset_index(columns={'index': 'feature', 0: 'freq'})

def get_percentage(df,col):
    rows=df.shape[0]
    df[col]=df[col].apply(lambda x: round(int((x/rows)*100)),2)
    return df



def get_cat_col(df):
    return df[df['freq']<10].feature.values





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
#Library we can use is fancyimpute - How this works is it generates basic mean impute and constructs a KD-Tree, Then it uses the KD-Tree to find the 'k' nearest neighbors.
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
# start the MICE training
imputed_training = mice(train.values)

##6.) Imputing using deep neural networks (Datawig)
##This method works really well with numeric and categorical variables . It is a library that learns ML models by using DNN to impute
##missing values. It has support for both CPU and GPU for training
##Advantages are that it is quite accurate compared to other imputation techniques,it can handle categorical data with 'Feature Encoder'
##Disadvatages are that it is slow with large datasets, a requirement is that you need to specify the columns that contain information about the target column
##that will be impyuted

##Example Code for imputation using neural networks


df_train, df_test = datawig.utils.random_split(train)

#Initialize a SimpleImputer model
imputer = datawig.SimpleImputer(
    # column(s) containing information about the column we want to impute
    input_columns=['1', '2', '3', '4', '5', '6', '7', 'target'],
    output_column='0',  # the column we'd like to impute values for
    output_path='imputer_model'  # stores model data and metrics
)

train = data_split[0].copy()
##For the above data we will use various imputation methods.
train2 = train.copy()
train3 = train.copy()
train4 = train.copy()



##Regression - 
##Function to get non-numeric columns

def get_non_numeric(df,flags=[]):
    columns=df.columns
    for col in columns:
        try:
            df[col]=df[col].astype('int')
        except:
            flags.append(col)
    return flags


train_data_path = '/Users/vvithala/Documents/train_model_data.csv'
validation_data_path= '/Users/vvithala/Documents/validation_model_data.csv'


##Function to search for file into memory and return number of numeric columns and option to rename columns

dir_path1='/Users/vvithala/'
train_file_name='train_model_data.csv'
validation_file_name='validation_model_data.csv'


def ff(folder='',file_name=''):
    if file_name='':
        name=input('Enter file name ')
    if folder='':
        ans=inquirer.prompt(folder_selection)
        if ans='Other':
            ans=input('Enter folder name: ')
            folder=ans.copy()
        else:
            folder=ans.copy()
    else:
        next
    folder_path='/Users/vvithala/{}'.format(folder)
    file_path='{}/{}'.format(folder_path,file_name)
    return locals()

locals().values()
##Function to read file 
##Simple os package capabilities 
##Function to get abnormally long file names and rename 
def os_functions(todo):
    ##Move data files to cwd
    ##change current working directory
    ##output folder

    
##Compliance for every script - 1.) Specify file name , folder name , writeout , working directory

##Function to create workspace
def create_workspace(path='',workspace_name=''):
    if path=='':
        path = os.getcwd()
        print("The current working directory is {}".format(path))
        ans=make_inquiry('change_wd?',message='Do you want to change the current working directory? ',choices=['Yes','No'])
        if ans=='Yes':
            full_path=inquirer.prompt(most_frequent_directories)
            if path=='Set Custom Directory'?:
                full_path=input('Enter full path of the file! ')
            ##Change directory
            os.chidir()
    os.chrdir()
    if workspace_name='':
        name=input('Enter workspace name :')
    else:


'Most frequent directories': {'question': 'Do you want to set your current working ', 'choices': '/Users/vvithala/Desktop/', '/Users/vvithala/Downloads/', '/Users/vvithala/Desktop/','Set Custom Directory'}

#ff('train_model_data.csv')


####Exploratory Data Analysis


def make_inquiry(var_name='method', message='', choices=[]):
    questions = [inquirer.List(var_name, message, choices)]
    answer = inquirer.prompt(questions)
    return answer[var_name]


##Master dictionaries that can be updated as new changes come along.


question_dict={'imputation_method': {'question': 'How would you like to impute the missing values?',\
     'choices': ['std', 'robust', 'minmax', 'normal', 'knn', 'nn', 'mice']},\
         'problem_type': {'message': "What kind of problem is this?",\
             'choices': ['Regression', 'Classification', 'Clustering', 'Dimensionality Reduction']},\
                 'working_directory': {'message': 'Where is the file?',\
                      'choices':['Downloads', 'Desktop', 'Documents', 'Other']},\
                          'regression_selection':{'message': 'What Type of Regression would you like to fit',\
                              'choices': ['ElasticNet Regression', 'Lasso Regression', 'SVR', 'Random Forest Regressor', 'Ridge Regression', 'Regression - Stochastic Gradient Descent']},\
                                  'feature_elimination': {'message': 'Do you want to eliminate features?',\
                                      'choices': ['Yes', 'No']},\
                                          'dimensionality_reduction_choices': {'question': 'How do you want to perform dimensionality reduction?',\
                                              'choices': ['Randomized Principal Component Analysis', 'ISOMap','Spectral Embedding', 'LLE?', 'Kernel Approximation']},\
                                                  'classification_choices': {'question': 'What kind of classification algorithm would you want to apply?',\
                                                      'choices': ['Linear Classifier', 'Classification - Stochastic Gradient Descent', 'Naive Bayes', 'KNN Classifier',\
                                                           'SVC', 'RandomForestClassifier', 'Kernel Approximation']},\
                                                               'clustering_choices': {'question': 'How do you want to perform clustering?: ',\
                                                                   'choices': ['Mean Shift', 'VBGMM', 'MiniBatchKMeans', 'KMeans Clustering', 'Spectral Clustering', 'GMM'],'changing_colnames':{'question':'Do you want to give more appropriate column names?','choices':['Yes','No']}

}
                                                      




messages_dict={'exception_handling':{}}

variable_dict=locals()

exception_handling,processing,prettyprinting,decision_points
##31st January - Month End Consolidation 

##5:30 am - 08:00 am Code for Missing Value Analysis , Data Description Report , Outlier Analysis , Coding scikit learn cheatsheet workflow , Uploading to github- (Estimated time 8:00am  150min broken down into sessions of  20, 30, 40 , 50, 10)

##Success - 08:00 am - 10:30 am : Breakfast @ Westin ##Motivation hour 9:30 am - 11:00 am if having a nice breakfast , Professional Development :  1.)prepare resume , 2,)List 10 next most relevant job applications - Apply and document 3.) Follow up applied jobs!

##Failure - 08:00 - 09:00 - Go to INSOFE 

##PluggedIn - 02:00 - 04:20 - Intense Workout

##4:30 pm - 5:30 INSOFE Career Services Info Session - Prep , Job Pitch , 


##Check CPU,GPU and RAM - 1gb has 1073741824 bytes

def check_CPU_RAM():
    cpu = os.cpu_count()
    mem = virtual_memory()
    print('''This instance has {} CPU's and  has {} GB of RAM'''.format(
        os.cpu_count(), round(virtual_memory().total/1073741824,0)))
check_CPU_RAM()

def get_df(dictionary):
    return pd.DataFrame(dictionary)

##Generate Reports
##func 10
##threshold= 40%




def get_uniqueval_count(df):
    return df.nunique().reset_index().rename(columns={'index': 'feature', 0: 'freq'})

def get_miss_count(df):
    return df.isna().sum().reset_index().rename(columns={'index': 'feature', 0: 'freq'})


def get_percentage(df, col):
    rows = df.shape[0]
    df[col] = df[col].apply(lambda x: round(int((x/rows)*100)), 2)
    return df
##func 11




##func 12


def report_3(df):
    second_report = report_intermediate(df)
    num_cols = df._get_numeric_data().columns.to_list()
    binary_cat = second_report[second_report['cat_2'] == 1].feature.to_list()
    cat_10 = second_report[second_report['cat_10'] == 1].feature.to_list()
    cat_20 = second_report[second_report['cat_20'] == 1].feature.to_list()
    cat_30 = second_report[second_report['cat_30'] == 1].feature.to_list()
    cat_cols = list(set(cat_10+cat_20+cat_30+binary_cat))
    drop_keys = second_report[second_report['is_key'] == 1].feature.to_list()
    num_cols = [x for x in num_cols if x not in cat_cols+drop_keys]
    keys = second_report[second_report['is_key'] == 1].feature.to_list()
    features_miss = df.columns.to_list()
    drop_col = second_report[second_report['drop_col'] == 1].feature.to_list()
    drop_cat_col = second_report[second_report['cat_1'] == 1].feature.to_list()
    total_drop = drop_col+drop_cat_col+drop_keys
    miss_5 = second_report[second_report['miss_5'] == 1].feature.to_list()
    total_columns_dealt = set(total_drop+cat_cols+num_cols+miss_5)
    total_columns_left = set(df.columns).difference(total_columns_dealt)
    print('Total columns to be dropped is : {}' .format(total_drop), '\n')
    print('Total columns which have missing values <thres is {}:'.format(
        len(miss_5)), '\n')
    print('Total columns which are binary :', len(binary_cat), '\n')
    print('Total columns which have <10 categories are : {}' .format(len(cat_10)), '\n')
    print('Total columns which have <20 categories are : {} '.format(len(cat_20)), '\n')
    print('Total columns which have <30 categories are : {}'.format(len(cat_30)), '\n')
    #report_df = pd.DataFrame().from_dict({'To be dropped': [len(total_drop), total_drop], 'To be dropped': [len(total_drop), total_drop})})
    return locals()

##Function for final report regarding missing values and unique values



def report_final(df):
    report_3 = report_pre_final(df)
    final_df = pd.DataFrame().from_dict(dict((k, [report_3[k]]) for k in ('binary_cat', 'cat_10', 'cat_20', 'cat_30', 'drop_col', 'miss_5', 'keys', 'total_drop', 'total_columns_left')))
    final_df = final_df.T.reset_index().rename(columns={'index': 'feature', 0: 'feature_names'})
    final_df['freq'] = final_df.feature_names.apply(lambda x: len(x))
    return [locals(), report_3]


class MultiColumnLabelEncoder:
    def __init__(self):
        self.columns = None
        self.led = defaultdict(preprocessing.LabelEncoder)

    def fit(self, X):
        self.columns = X.columns
        for col in self.columns:
            cat = X[col].unique()
            cat = [x if x is not None else "None" for x in cat]
            self.led[col].fit(cat)
        return self

    def fit_transform(self, X):
        if self.columns is None:
            self.fit(X)
        return self.transform(X)

    def transform(self, X):
        return X.apply(lambda x:  self.led[x.name].transform(x.apply(lambda e: e if e is not None else "None")))

    def inverse_transform(self, X):
        return X.apply(lambda x: self.led[x.name].inverse_transform(x))


def df_prelim_clean_analysis(df):
    ##Convert all column names to lower case
    df.columns = [x.lower() for x in df.columns]
    ##Drop identical rows
    df = df[df.duplicated(keep='first')]
    ##Replace missing values with -9999
    df = df.fillna(-9999, inplace=True)
    print('Missing values have beeen replaced with -1 until further treatment!')
    columns=df.columns
    ##Process of changing column name if needed
    ans=inquirer.prompt(change)
    if ans == 'y':
        new_col = {}
        for col in df.columns:
            new_col[col]=change_col_name(col)
        df.columns = new_col.values
        print('Column names changed', '\n')
    else:
        print('Column Name not changed!')
    num_data = df._get_numeric_data()
    num_cols = num_data.columns
    cat_cols = set(df.columns.to_list()).difference(num_cols)
    print('There are {} numeric columns and they are {}'.format(len(num_cols), num_cols))
    print('There are {} categorical columns and they are {}'.format(len(cat_cols), cat_cols), '\n')
    print(df[cat_cols])
    ans = input('Do you want to lower the case of any columns (y/n) ')
    if ans == 'y':
        for col in cat_cols:
            try:
                df[col] = df[col].apply(lambda x: x.to_lower())
            except:
                next
    else:
        print('Nothing Done')
    return df

##code to select columns that need name change - **** inquirer.Checkbox
'columns_checklist':{'question':'Check the columns whose name you want to change..','choices':df.columns}




##Each step of the data science life cycle will consist

##Common strategies
##Evaluation
##Change Documentation****


def num_col(df,num=[],not_num=[]):
    columns=df.columns
    for col in columns:
        try:
            df[col]=df[col].astype('float')
            num.append(col)
        except:
            not_num.append(col)
    return num,not_num


##Getting numeric columns and categorical columns
numeric_columns,not_numeric=num_col(data)
numeric_columns=num_col(data)['num']
not_numeric=num_col(data)['not_num']

##Method to treat categorical columns - OneHotEncoding, LabelEncoder 



###Scikit Learn Cheat Sheet

##Read data file

file_name='PewDiePie.csv'

data=pd.read_csv(ff('Downloads',file_name))

##Perform preliminary data cleaning - 

#1.) Convert column names to lower case
#2.) Check which columns have numeric data and convert them to float
#3.) Display number of numeric columns and categorical columns


def clean_1(df,change_names=True):
    ##List of columns in dataframe
    columns_lower=[x.lower() for x in df.columns]
    ##Convert column names to lower case
    df.columns=columns_lower
    ##Check get numeric and categorical columns
    numeric,not_numeric=num_col(df)
    ##Convert all numeric columns to float
    df[numeric]=df[numeric].astype('float')
    print('All numeric columns have been converted into float')
    print('There are {} numeric columns and {} not numeric columns'.format(len(numeric),len(not_numeric)))
    flag=input('Do you want to change the column names')
    if flag=='yes':
        change_dict={}
        for col in col_lower:
            print('For column name : {} , what do you want to change the name to?')
            new_name=input('Enter new column name : ')
            change_dict[col]=new_name
        ##Replace existing column names with new column names 
        df.columns=change_dict.values
    return df


##Functions to create report on missing values and unique values in each column in dataframe 


clean_1(data)

num_col(data)['num']

##Function to get report with missing/unique value frequency,% and datatype
df=data.copy()
def report_1(df, first_report={}):
    negligible_threshold = round(0.05*df.shape[0])
    ##Drop columns whose missing value freq is less than threshold
    nunique= get_uniqueval_count(df)
    miss_list = get_miss_count(df)
    negligible_columns=miss_list[miss_list['freq']<=negligible_threshold].feature.unique()
    ##Drop negligible missing values
    df=df.dropna(subset=negligible_columns)
    print('Dropped missing values in columns where there is <=5% missing values')
    datatypes=df.dtypes.to_list()
    ##Get miss % for every column
    metric1=get_percentage(miss_list,freq)
    ##Get unique % for every column
    metric2=get_percentge(nunique,freq)
    ##Get missing freq for each column
    miss_features=miss_list.feature.values
    ##Get unique frequency  
    unique_features=nunique.feature.values
    return locals()

##Function to get second report that adds flags various categories in missing values 
def report_2()
report=report_1(data)
report.keys()
report['negligible']

##Report 2 buckets columns into various buckets with respect to missing values and unique values
def report_2(df):
    ##Dealing with Missing Values
    report = report_1(df)
    report_2 = report.copy()
    report_2['miss_5'] = np.where((report_2['miss_%'] > 0) & (report_2['miss_%'] <= 5), True, False).astype('int')
    report_2['miss_20'] = np.where((report_2['miss_%'] > 5) & (report_2['miss_%'] <= 20), True, False).astype('int')
    report_2['miss_<50'] = np.where((report_2['miss_%'] > 20) & (report_2['miss_%'] <= 50), True, False).astype('int')
    report_2['drop_col'] = np.where(report_2['miss_%'] > 60, True, False).astype('int')
    report_2['is_key'] = np.where(report_2['unique_%'] > 90, True, False).astype('int')
    report_2['cat_10'] = np.where((report_2['unique_freq'] > 2) & (report_2['unique_freq'] <= 10), True, False).astype('int')
    report_2['cat_20'] = np.where((report_2['unique_freq'] > 10) & (report_2['unique_freq'] <= 20), True, False).astype('int')
    report_2['cat_30'] = np.where((report_2['unique_freq'] > 20) & (report_2['unique_freq'] <= 30), True, False).astype('int')
    report_2['cat_50'] = np.where((report_2['unique_freq'] > 30) & (report_2['unique_freq'] <= 50), True, False).astype('int')
    report_2['cat_1'] = np.where(report_2['unique_freq'] <= 1, True, False).astype('int')
    report_2['cat_2'] = np.where(report_2['unique_freq'] == 2, True, False).astype('int')
    return report_2


##Actions to be done after the second report 

##Make sure the features that have less than 5% missing values are dropped
##Impute values for columns that have <20% missing values
##Make decisions for columns >20% and <=60% 

##Drop columns that have only one unique values
##Perform binary encoding for columns tht have only 2 unique values
##For columns that have between 20 and 30 categories , check whether the categories are nominal in nature - If they are ordinal use LabelEncoder, else use one hot encoding

##Analyze cat_50 , miss_<50


data=pd.read_csv('/Users/vvithala/Downloads/heart.csv')

##Output of first report should give for each column - the number of missing values,% the number of unique values,%

##Function to deal with numeric transforming numeric values - Scale , Normalize , Standardize 

##Function to deal with missing values 

##Function to deal with categorcal variables and transforming them into algorithm-ready format

##Function to fit 6 types of regression -  'ElasticNet Regression', 'Lasso Regression', 'SVR', 'Random Forest Regressor', 'Ridge Regression', 'Regression - Stochastic Gradient Descent'

##Function to perform dimensionality reduction - Randomized Principal Component Analysis , ISOMap , Kernel Approximation , Spectral Embedding

##Function to perform clustering - MeanShift , MiniBatch-KMeans , Spectral Clustering , RandomForestClassifier , GMM , VBGMM , KMeans Clustering 

##Function to perform classification - LogisticRegression , Naive Bayes , KNeighbours Classification , SGD Classifier , Linear SVC , Kernel Approximation 




##Function to fit LinearRegression, Lasso , Ridge, ElasticNet

##When function is called it fits LinearRegression , Lasso , Ridge , ElasticNet provided clean train and test data

def fit_standard_regressions(target,features):
    xtrain,xtest,ytrain,ytest=split_dataset(features,target)
    print('Fitting Simple Linear Regression','\n')
    slr=LinearRegression()
    slr.fit(xtrain,ytrain)
    predictions=slr.predict(xtest)
    slr_mse=mean_squared_error(ytest,predictions)
    slr_mae=mean_absolute_error(ytest,predictions)
    slr_rmse=np.sqrt(mse)
    slr_r2=r2_score(model)
    slr_mape=get_mape(ytest,predictions)
    ##Lasso and Ridge Regression are used to reduce model complexity - 
    print('Fitting Lasso Regression : ','\n')
    lasso=Lasso()
    lasso.fit(xtrain,ytrain)
    predictions=lasso.predict(xtest)
    lasso_mse=mean_squared_error(ytest,predictions)
    lasso_mae=mean_absolute_error(ytest,predictions)
    lasso_rmse=np.sqrt(lasso_mse)
    lasso_mape=get_mape(ytest,predictions)
    lasso_r2=r2_score(lasso)
    print('Fitting Ridge Regression','\n')
    ridge=Ridge()
    ridge.fit(xtrain,ytrain)
    predictions=ridge.predict(xtest)
    ridge_mse=mean_squared_error(ytest,predictions)
    ridge_mae=mean_absolute_error(ytest,predictions)
    ridge_r2=r2_score(ridge)
    ridge_rmse=np.sqrt(ridge_mse)
    ridge_mape=get_mape(ytest,predictions)
    return locals()





def get_summary_dict(df):
    summary=df.describe().T.reset_index().set_index('index')
    summary_dictionary = summary.to_dict()
    return summary_dict
