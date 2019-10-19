from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from prepro import get_percent
from prepro import num_cat_data_stats
from prepro import l
from prepro import s
from fancyimpute import KNN
import seaborn as sns
from prepro import missingval_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold, GridSearchCV
from sklearn import preprocessing, metrics, svm, ensemble


data = pd.read_csv('UniversalBank.csv - UniversalBank.csv.csv')
data.columns = [x.lower().replace(' ', '_') for x in data.columns]
data = data[data.experience >= 0].copy()
data = data.drop(columns=['id','zip_code'])
data[['family', 'education']] = data[['family', 'education']].astype('category')
data,miss=missingval_report(data,target='personal_loan')
target=data['personal_loan'].copy()
features=data.drop(columns=['personal_loan']).copy()
xtrain,xtest,ytrain,ytest=train_test_split(features,target,train_size=0.7,random_state=123)

xtrain.dtypes
def train_test(df,target):
    target_data=df[target]
    features=df.drop(subset=[target])
    xtrain,xtest,ytrain,ytest=train_test_split(features,target,train_size=train_size)



###Ensure data sanity by enforcing rel world rules and treat records which dont comply

##Remiving negative experience

##Dealing with cat and num data

nuni=data.nunique().reset_index().rename(columns={'index':'features',0:'freq'})
catcol=nuni[nuni.freq<10].features.unique()
numcol=set(list(data.columns)).difference(set(catcol))

num=data[numcol]
cat=data[catcol]

##Dummyfication
xtrain.dtypes
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
xtrain=scaler.fit_transform(xtrain)
ytrain=np.array(ytrain)
def train_test(df, features,target,train_size=0.7):
    xtrain, xtest, ytrain, ytest = train_test_split(features, target, train_size=train_size)
    return xtrain, xtest, ytrain, ytest

from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)

# Define the parameter grid to use for tuning the Support Vector Machine
parameters = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

# Pick the goal you're optimizing for e.g. precision if you prefer fewer false-positives
# recall if you prefer fewer false-negatives. For demonstration purposes let's pick several
# Note that the final model selection will be based on the last item in the list


xtrain.dtypes



scoringmethods = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc']
for score in scoringmethods:
    print("~~~ Hyper-parameter tuning for best %s ~~~" % score)
    # Setup for grid search with cross-validation for Support Vector Machine
    # n_jobs=-1 for parallel execution using all available cores
    svmclf = GridSearchCV(svm.SVC(C=10), parameters,cv=kf, scoring=score, n_jobs=-1)
    svmclf.fit(xtrain, ytrain)
    # Show each result from grid search
    print("Scores for different parameter combinations in the grid:")
    for params, mean_score, scores in svmclf.cv_scores_:
        print("  %0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print("")

a=GridSearchCV(svm.SVC(C=1), parameters, cv=kf, scoring='roc_auc', n_jobs=-1)
a.fit(xtrain,ytrain)
# Show classification report for the best model (set of parameters) run over the full dataset
print("Classification report:")
y_pred = svmclf.predict(X)
print(classification_report(y, y_pred))

# Show the definition of the best model
print("Best model:")
print(svmclf.best_estimator_)

# Show accuracy and area under ROC curve
print("Accuracy: %0.3f" % accuracy_score(y, y_pred, normalize=True))
print("Aucroc: %0.3f" % metrics.roc_auc_score(y, y_pred))
print("")
