import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
from pandas import DataFrame
import cv2
import random
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn import svm,metrics
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
import heapq
import os
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from lifelines import CoxPHFitter, WeibullAFTFitter
import seaborn as sns

def get_traintest_data(index = 0,train_num=90,test_num=31 , csv_name='super3',random_seed=0):
    ''' Get train_test data for lasso-cox model
    Parameters
    ----------
    index: int 
        dataset index (100 bootstrap datasets)
    train_num,test_num: int
        train sample size & test sample size
    csv_name: str
        data from which csv
    random_seed: int
        random sample seed
    '''
    base_dir = '/data/Datasets/MediImgExp/csv/rfs_csv'
    csv_name = os.path.join(base_dir,csv_name+'.csv' )
    csv_data = pd.read_csv( csv_name )
    csv_data2 = pd.read_csv( base_dir + '/' + 'super1.csv',index_col=0 )

    column_names = csv_data.columns.values
    column_names2 = csv_data2.columns.values[0:13]
    # feature & output names
    feature_columns_name_ori= np.delete(column_names, [0,10,11,12,13])
    # concatenate the 9 + 13 = 22 features
    feature_columns_name_ori = np.concatenate((feature_columns_name_ori,column_names2))

    csv_data.index = csv_data['Ind'] 
    
    column_names = csv_data.columns.values
    output_columns_name = column_names[-4:]

    # train & test Index 
    # add age... features data
    csv_data = pd.concat([csv_data, csv_data2[column_names2]  ],axis=1 )  

    random.seed(random_seed)
    csv_data = shuffle(csv_data)
    Index = csv_data.index.values
    #random.shuffle(Index)
    train_Ind= Index[:train_num]
    test_Ind=Index[train_num:]
    
    # fill the nan with Multivariate Imputation
    imp = IterativeImputer(max_iter=100, random_state=0)
    imp.fit( csv_data.values ) 
    csv_data.loc[:,:] = (imp.transform(csv_data)) 
    column_names = csv_data.columns.values
    column_names = np.delete( column_names ,[ 10,11,12,13 ] )
    csv_data.loc[:,column_names] = np.round( csv_data.loc[:,column_names] )

    csv_data.drop(['Ind'],axis=1,inplace=True)
    csv_data.to_csv(base_dir + '/csv_100/' + 'super5_MIfilled_'+ str(index) +'.csv')


def selectFeaturesWithCox():
    ''' 
    Use univariable Cox to select Feature at first
    Return:
    ----------
    meaningful_columns: list(str)
        a list for meaningful column names
    '''
    base_dir = '/data/Datasets/MediImgExp/csv'
    feature_df = pd.read_csv( base_dir + '/figurefeatures.csv', index_col = 0)
    os_csv = pd.read_csv(base_dir + '/' + 'csv_100/' + 'super5_MIfilled_0.csv', index_col=0)
    column_name_list = feature_df.columns.values
    feature_df['OS'] = os_csv['OS']
    feature_df['DEATH'] = os_csv['DEATH']

    meaningful_columns = []
    for column_name in column_name_list:
        cph = CoxPHFitter()
        cph.fit(feature_df[[column_name,'OS', 'DEATH']], duration_col='OS', event_col='DEATH')
        p = cph.summary.p.values[0]
        if p <= 0.05:
            meaningful_columns.append(column_name)
            print(column_name)
    return meaningful_columns

def saveFigureFeatures(meaningful_columns):
    ''' 
    Integrate the meaningful columns into the final csv data
    Parameters:
    ----------
    meaningful_columns: list(str)
        a list for meaningful column names
    '''

    base_dir = '/data/Datasets/MediImgExp/csv'
    feature_df = pd.read_csv( base_dir + '/figurefeatures.csv', index_col = 0)
    for i in range(0, 100):
        ori_df = pd.read_csv(base_dir + '/csv_100/' + 'super5_MIfilled_' + str(i) + '.csv', index_col=0)
        figure_df = pd.DataFrame(index=ori_df.index)
        for column_name in meaningful_columns:
            figure_df[column_name] = feature_df.loc[ori_df.index.values, column_name]
        figure_df['OS'] = ori_df['OS']
        figure_df['DEATH'] = ori_df['DEATH']
        figure_df.to_csv(base_dir + '/figure_csv_100/' + 'super5_MIfilled_' + str(i) + '.csv', index=True)

def getCorrelationHeatMap():
    '''
    Draw a correlation heatmap
    '''
    base_dir = '/data/Datasets/MediImgExp/csv'
    df = pd.read_csv(base_dir + '/super1_notFillNaN.csv', index_col=0)
    column_names = df.columns.values

    # fill the nan with mean values
    for column in list(df.columns[df.isnull().sum() > 0]):
        mean_val = df[column].mean()
        df[column].fillna(mean_val, inplace=True)

    df.loc[:,column_names[0:12]] = np.round(df.loc[:,column_names[0:12]].values)
    matrix = df.loc[:, column_names[0:77]].corr()
    plt.subplots(figsize=(50, 50))
    sns.heatmap(matrix, annot=True, vmax=1, square=True, cmap="Blues")
    plt.savefig('correlation.jpg')

if __name__ == '__main__':
    for i in range(0,100):
        get_traintest_data(index=i,train_num=90,test_num=31 ,random_seed=i)

    meaningful_columns = selectFeaturesWithCox()
    saveFigureFeatures(meaningful_columns)
