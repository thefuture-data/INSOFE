import essential_packages
import prepro

def missingval_report(df, target='', drop_col=False, drop_id=False):
    ##Bucket features to categorical and numeric data.
    rows = df.shape[0]
    cols = df.shape[1]
    nuni = df.nunique().reset_index().rename(columns={'index': 'feature', 0: 'freq'})
    global catcol
    catcol = list(nuni[nuni['freq'] < 10].feature.unique())
    global numcol
    numcol = list(set(df.columns).difference(set(catcol)))
    print('There are ', len(catcol),' categorical variables and they are: ', catcol)
    print('There are ', len(numcol),' numeric columns and they are: ', numcol)
    ##Missing Data Info
    miss = df.isna().sum().reset_index().rename(
        columns={'index': 'feature', 0: 'freq'})
    #% of missing values in each column
    miss['miss_%'] = miss.freq.apply(
        lambda x: (x/rows)*100)
    # Unique values as a percentage of total number of rows
    miss['unique_%'] = round((nuni.freq/rows)*100, 2).astype('int')
    #Drop freq column
    miss = miss.drop(columns=['freq'],axis=1)
    ##Columns with missing values
    misscol = miss[miss['miss_%'] > 0].feature.unique()
    ##Calculate metrics
    metric1 = (len(misscol)/cols)*100
    metric2 = miss[miss['miss_%'] > 70].feature.unique()
    metric3 = miss[miss['unique_%'] == 100].feature.unique()
    metric4 = miss[(miss['miss_%'] >0)&(miss['miss_%'] < 5)].feature.unique()
    #print(metric4)
    print('There are ', len(metric4),' column/s that have less than 5% missing values and they are ', metric4)
    if len(metric4) > 0:
        df = df.dropna(subset=metric4, axis=0)
        print('The rows with columns less than 5% missing values has been dropped')
    ##If target has missing values , we remove the rows with missing values i.e axis=0
    if target in misscol:
        df = df.dropna(subset=[target], axis=0)
    ##if drop_id=True , we drop columns which are 100% unique
    if drop_id==True:
        df = df.drop(columns=metric3, axis=1)
    ##if drop_col=True , we drop columns which have more than 70% missing values in the column
    if drop_col==True:
        df = df.drop(columns=metric2, axis=1)
    ##Check if columns have been dropped
    cols2 = df.shape[1]
    cols3 = cols-cols2
    if cols3 > 0:
        print('There are ', cols3, ' columns that have been dropped.')
    #print(cols3)
    print('The number of columns which have missing values are:',len(misscol), ' and they are ', misscol)
    print('% of the columns have missing values: ', round(metric1, 0), '%')
    print('There are ', len(metric2),' columns which are missing more than 50% are : ', metric2)
    print('There are ', len(metric3),' columns which are a 100% unique and may not be useful for model building and they are', metric3)
    #print(df.sample(10))
    return df, miss