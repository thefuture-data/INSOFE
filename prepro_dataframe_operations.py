#!/usr/bin/env python
# coding: utf-8
import essential_packages
import seaborn as sns
# In[8]:

##Import required packages

def sumstats(df):
    return df.describe(include='all').T

def numdata(df):
    global numcols
    numcols=df._get_numeric_data().columns
    return df._get_numeric_data()

def catdata(df):
    global catcols
    catcols=set(list(df.columns)).difference(list(df._get_numeric_data().columns))
    return df[catcols]
    

def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '200px'),
                        ('font-size', '12pt')])]


def num_cat_data_stats(df):
    global num
    global cat
    num=numdata(df)
    cat=catdata(df)
    #print("Summary Stats for numeric data: ")
    #num.describe().T.reset_index().rename(columns={'index':'numeric_features'})
    #print('Summary Stats for categorical data: ')
    #cat.describe(include='all').T.reset_index().rename(columns={'index':'cat_features'})
    return {'num_stat':sumstats(num).T.reset_index().rename(columns={'index':'numeric_features'}),'cat_stat':sumstats(cat).T.reset_index().rename(columns={'index':'categoric_features'}),'df_stat':sumstats(df).T.reset_index().rename(columns={'index':'feature_name'}),'num_data':num,'cat_data':cat,'data':df,'num_corr':num.corr(),'cat_corr':cat.corr()}
    

def s(df):
    return df.sample(10)

def get_percent(x,y):
    return round((x/y)*100,0)

def l(x):
    return list(x)


##function for finding correlation among numeric features
def corrmap(corr, cmap=sns.diverging_palette(5, 250, as_cmap=True)):
    return corr.style.background_gradient(cmap, axis=1)\
        .set_properties(**{'max-width': '80px', 'font-size': '10pt'})\
        .set_caption("Hover to magify")\
        .set_precision(2)\
        .set_table_styles(magnify())

##function for finding correlation among categorical features

##Checking missing value in target variable
def targetnacheck(df,target=''):
    return df[target].isna().sum() == 0







