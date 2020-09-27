import sys
import scipy, importlib, pprint, matplotlib.pyplot as plt, warnings
import glmnet_python
from glmnet import glmnet
from glmnetPlot import glmnetPlot
from glmnetPrint import glmnetPrint
from glmnetCoef import glmnetCoef
from glmnetPredict import glmnetPredict
from cvglmnet import cvglmnet
from cvglmnetCoef import cvglmnetCoef
from cvglmnetPlot import cvglmnetPlot
from cvglmnetPredict import cvglmnetPredict
import pandas as pd
import numpy as np
from sksurv.metrics import cumulative_dynamic_auc
from sksurv.nonparametric import CensoringDistributionEstimator
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn import metrics
import csv
from math import sqrt

def getNRIdata():
    ''' 
    Get OS predicted results(0/1) for 1/2/3 year
    in order to calculate NRI between new and old models
    '''
    feature_type = 'all'  #'ic' 'res' 'clinical' 'all' 'ic+res' 'res+clinical' 'ic+clinical'
    results_dir = 'NRI/' + feature_type + '/'

    for i in range(0, 100):
        print(i)
        csv_data = pd.read_csv('csv_100/super5_MIfilled_' + str(i) + '.csv', index_col=0)
        csv_data_figure = pd.read_csv('figure_os_csv_100_005_resnet18/super5_MIfilled_' + str(i) + '.csv', index_col=0)
        csv_data_figure_res101 = pd.read_csv('figure_os_csv_100_005_resnet101/super5_MIfilled_' + str(i) + '.csv', index_col=0)
        csv_data_figure_res152 = pd.read_csv('figure_os_csv_100_005_resnet152/super5_MIfilled_' + str(i) + '.csv', index_col=0)

        # 选择特征种类
        if feature_type == 'ic':
            column_name = csv_data.columns.values[0:21]
        elif feature_type == 'res':
            figure_column_name = csv_data_figure.columns.values[0:-2] 
            for figure_column in figure_column_name:
                csv_data[figure_column+'_res18'] = csv_data_figure[figure_column]
            
            figure_column_name2 = csv_data_figure_res101.columns.values[0:-2]
            for figure_column in figure_column_name2:
                csv_data[figure_column + '_res101'] = csv_data_figure_res101[figure_column]
            
            figure_column_name3 = csv_data_figure_res152.columns.values[0:-2]
            for figure_column in figure_column_name3:
                csv_data[figure_column + '_res152'] = csv_data_figure_res152[figure_column]

            figure_column_name = [(x + '_res18') for x in figure_column_name]
            figure_column_name2 = [(x + '_res101') for x in figure_column_name2]
            figure_column_name3 = [(x + '_res152') for x in figure_column_name3]
            column_name = np.concatenate((figure_column_name, figure_column_name2), axis=0)
            column_name = np.concatenate((column_name, figure_column_name3), axis=0)

        elif feature_type == 'clinical':
            column_name = csv_data.columns.values[26:39]

        elif feature_type == 'all':
            column_name = csv_data.columns.values
            column_name = np.delete(column_name, [21, 22, 23, 24, 25, 39]) 

            figure_column_name = csv_data_figure.columns.values[0:-2]  
            for figure_column in figure_column_name:
                csv_data[figure_column+'_res18'] = csv_data_figure[figure_column]
            
            figure_column_name2 = csv_data_figure_res101.columns.values[0:-2]  
            for figure_column in figure_column_name2:
                csv_data[figure_column + '_res101'] = csv_data_figure_res101[figure_column]
            
            figure_column_name3 = csv_data_figure_res152.columns.values[0:-2]  
            for figure_column in figure_column_name3:
                csv_data[figure_column + '_res152'] = csv_data_figure_res152[figure_column]

            figure_column_name = [(x + '_res18') for x in figure_column_name]
            figure_column_name2 = [(x + '_res101') for x in figure_column_name2]
            figure_column_name3 = [(x + '_res152') for x in figure_column_name3]

            column_name = np.concatenate((column_name, figure_column_name), axis=0)
            column_name = np.concatenate((column_name, figure_column_name2), axis=0)
            column_name = np.concatenate((column_name, figure_column_name3), axis=0)

        elif feature_type == 'ic+res':
            column_name = csv_data.columns.values[0:21]  

            figure_column_name = csv_data_figure.columns.values[0:-2]  
            for figure_column in figure_column_name:
                csv_data[figure_column+'_res18'] = csv_data_figure[figure_column]
            
            figure_column_name2 = csv_data_figure_res101.columns.values[0:-2]  
            for figure_column in figure_column_name2:
                csv_data[figure_column + '_res101'] = csv_data_figure_res101[figure_column]
            
            figure_column_name3 = csv_data_figure_res152.columns.values[0:-2] 
            for figure_column in figure_column_name3:
                csv_data[figure_column + '_res152'] = csv_data_figure_res152[figure_column]

            figure_column_name = [(x + '_res18') for x in figure_column_name]
            figure_column_name2 = [(x + '_res101') for x in figure_column_name2]
            figure_column_name3 = [(x + '_res152') for x in figure_column_name3]

            column_name = np.concatenate((column_name, figure_column_name), axis=0)
            column_name = np.concatenate((column_name, figure_column_name2), axis=0)
            column_name = np.concatenate((column_name, figure_column_name3), axis=0)

        elif feature_type == 'res+clinical':
            column_name = csv_data.columns.values[26:39] 

            figure_column_name = csv_data_figure.columns.values[0:-2] 
            for figure_column in figure_column_name:
                csv_data[figure_column+'_res18'] = csv_data_figure[figure_column]
            
            figure_column_name2 = csv_data_figure_res101.columns.values[0:-2]  
            for figure_column in figure_column_name2:
                csv_data[figure_column + '_res101'] = csv_data_figure_res101[figure_column]
            
            figure_column_name3 = csv_data_figure_res152.columns.values[0:-2] 
            for figure_column in figure_column_name3:
                csv_data[figure_column + '_res152'] = csv_data_figure_res152[figure_column]

            figure_column_name = [(x + '_res18') for x in figure_column_name]
            figure_column_name2 = [(x + '_res101') for x in figure_column_name2]
            figure_column_name3 = [(x + '_res152') for x in figure_column_name3]

            column_name = np.concatenate((column_name, figure_column_name), axis=0)
            column_name = np.concatenate((column_name, figure_column_name2), axis=0)
            column_name = np.concatenate((column_name, figure_column_name3), axis=0)

        elif feature_type == 'ic+clinical':
            column_name1 = csv_data.columns.values[0:21]  
            column_name2 = csv_data.columns.values[26:39]
            column_name = np.concatenate((column_name1, column_name2), axis=0)

        output_name = ['OS','DEATH']
        #output_name = ['RFS','RELAPSE']

        train_num = 90
        train_Ind = csv_data.index.values[:train_num]
        test_Ind = csv_data.index.values[train_num:]
        
        x = csv_data.loc[train_Ind, column_name ].values.astype(np.float64)
        y = csv_data.loc[train_Ind, output_name ].values.astype(np.float64)
        y_test = csv_data.loc[test_Ind, output_name].values.astype(np.float64)

        # define the structured array ----survival_train
        dt = np.dtype([('DEATH', 'bool'), ('OS', 'f')])
        #dt = np.dtype([('RELAPSE', 'bool'), ('RFS', 'f')])
        
        survival_train = np.array([ ( y[0,1],y[0,0] ) ],dtype=dt)
        for ii in range(1,len(train_Ind)):
            arr_train = np.array([ ( y[ii,1],y[ii,0] ) ],dtype=dt)
            survival_train = np.vstack([survival_train,arr_train])
        survival_train = np.squeeze(survival_train)

        # define the structured array ----survival_test
        survival_test = np.array([ ( y_test[0,1],y_test[0,0] ) ],dtype=dt)
        for ii in range(1,len(test_Ind)):
            arr_test = np.array([ ( y_test[ii,1],y_test[ii,0] ) ],dtype=dt)
            survival_test = np.vstack([survival_test,arr_test])
        survival_test = np.squeeze(survival_test)

        K = 5  #5-fold-cross validation
        inter = int(90/K)
        for k in range(0, K):
            list_all = np.arange(0, train_num)
            list_cut = np.arange(k * inter, (k + 1) * inter)
            list_res = list(set(list_all) - set(list_cut))
            x_ = x[list_res,:]
            y_ = y[list_res,:]
            fit = glmnet(x = x_.copy(), y = y_.copy(), family = 'cox')
            
            coef = glmnetCoef(fit, s=scipy.float64([0.05]))
            coef_list.append(coef)
            
        coef = np.mean(np.array(coef_list),axis=0)
        products = 0
        for j in range(0, len(coef)):
            products += (coef[j] * csv_data[column_name[j]])

        csv_data['score'] = products

        best_score=[0,0,0]
        auc , mean_auc, sens_list, fpr_list, order = cumulative_dynamic_auc(survival_train=survival_train,
            survival_test = survival_train, estimate = products[:train_num], times = (1, 2, 3))
        for ind in range(0, 3):
            sens = sens_list[ind]
            fpr = fpr_list[ind]
            area = []
            for ii in range(0,len(sens)):
                area.append( sens[ii]-fpr[ii] )
            max_ind = area.index(max(area))
            list_sens[ind].append(sens[max_ind]) 
            list_fpr[ind].append(fpr[max_ind])

            order_data = csv_data.loc[train_Ind,:].copy(deep=True) #train_Ind
            order_data.index = range(0,len(train_Ind))
            best_score[ind] = order_data.loc[order[max_ind], 'score']  # find the best cut-off score
        
        ordered_data = csv_data.loc[test_Ind,:].copy(deep=True)
        # save the predict & true results
        nri_df = pd.DataFrame(columns =['predY_1year','testY_1year','predY_2year','testY_2year','predY_3year','testY_3year'],index= ordered_data.index)

        for ind in range(0, 3):
            ordered_data['OS_flag_'+str(ind+1)] = ordered_data.apply(lambda x: 1 if x.score >= best_score[ind] else 0, axis=1)
            # OS DEATH -- RFS RELAPSE
            ordered_data['flag_year'] = ordered_data.apply(lambda x: 1 if (x.OS <= (ind + 1) and x.DEATH == 1) else 0, axis=1)
            nri_df['predY_' + str(ind + 1) + 'year'] = ordered_data['OS_flag_' + str(ind + 1)].values
            nri_df['testY_' + str(ind + 1) + 'year'] = ordered_data['flag_year'].values
        # save the predicted & true results(0/1) for further NRI calculating
        nri_df.to_csv(results_dir + 'super5_MIfilled_' + str(i) + '.csv', index=True)

def getNRIresults():
    '''
    calculate the NRI results between new and old models with the csv obtained by previous function
    '''
    new = 'all'  #'ic' 'res' 'clinical' 'all' 'ic+res' 'res+clinical' 'ic+clinical'
    old = 'ic+clinical'
    base_dir = 'NRI/'
    list_Z = [[],[],[]]
    list_NRI = [[], [], []]
    for i in range(0, 100):
        new_data = pd.read_csv(base_dir + new + '/super5_MIfilled_' + str(i) + '.csv', index_col=0)
        old_data = pd.read_csv(base_dir + old + '/super5_MIfilled_' + str(i) + '.csv', index_col=0)
        for j in range(0, 3):
            B1 = np.sum((new_data['testY_' + str(j + 1) + 'year'] == 1) &
                (new_data['predY_' + str(j + 1) + 'year'] == 1) &
                (old_data['predY_' + str(j + 1) + 'year'] == 0))
            C1 = np.sum((new_data['testY_' + str(j + 1) + 'year'] == 1) &
                (new_data['predY_' + str(j + 1) + 'year'] == 0) &
                (old_data['predY_' + str(j + 1) + 'year'] == 1))
            N1 = np.sum((new_data['testY_' + str(j + 1) + 'year'] == 1))

            B2 = np.sum((new_data['testY_' + str(j + 1) + 'year'] == 0) &
                (new_data['predY_' + str(j + 1) + 'year'] == 1) &
                (old_data['predY_' + str(j + 1) + 'year'] == 0))
            C2 = np.sum((new_data['testY_' + str(j + 1) + 'year'] == 0) &
                (new_data['predY_' + str(j + 1) + 'year'] == 0) &
                (old_data['predY_' + str(j + 1) + 'year'] == 1))
            N2 = np.sum((new_data['testY_' + str(j + 1) + 'year'] == 0))

            if N1 != 0 and N2 != 0 and ((B1 + C1) / (N1 * N1)) + ((B2 + C2) / (N2 * N2)) != 0:
                NRI = (B1 - C1) / N1 + (C2 - B2) / N2
                if NRI > 0:
                    Z = NRI / (sqrt(((B1 + C1) / (N1 * N1)) + ((B2 + C2) / (N2 * N2))))
                    if abs(Z) >= 1.96:  #P<0.05
                        list_NRI[j].append(NRI)
                        list_Z[j].append(Z)

    for j in range(0, 3):
        print('mean_NRI' + str(j + 1) + '= ', np.mean(np.array(list_NRI[j]), axis=0))
        print('list_Z' + str(j + 1) + '_size: ', len(list_Z[j]))

if __name__ == '__main__':
    list_auc = []
    list_auc_year = [[],[],[]]
    # all：early-stage NSCLC + late-stage NSCLC 
    # I：early-stage NSCLC 
    # II：late-stage NSCLC 
    list_p_all = [] 
    list_p_I = []
    list_p_II = []
    
    list_sens = [[],[],[]]  # sensitivity for 3 years (100 datasets)
    list_fpr = [[], [], []]  # fpr for 3 years (100 datasets)

    output_type = 'OS'  # 'OS' or 'RFS' choose the output
    feature_type = 'all'  #'ic' 'res' 'clinical' 'all' 'ic+res' 'res+clinical' 'ic+clinical'
    results_dir = 'csv_100_results/' + feature_type + '/'

    feature_weights=[[] for x in range(21)] #save the weights of IC features

    for i in range(0,100):
        print(i) 
        csv_data = pd.read_csv('csv_100/super5_MIfilled_' + str(i) + '.csv', index_col=0) 
        csv_data_figure = pd.read_csv('figure_os_csv_100_005_resnet18/super5_MIfilled_' + str(i) + '.csv', index_col=0)
        csv_data_figure_res101 = pd.read_csv('figure_os_csv_100_005_resnet101/super5_MIfilled_' + str(i) + '.csv', index_col=0)
        csv_data_figure_res152 = pd.read_csv('figure_os_csv_100_005_resnet152/super5_MIfilled_' + str(i) + '.csv', index_col=0)

        if feature_type == 'ic':
            column_name = csv_data.columns.values[0:21]
        elif feature_type == 'res':
            figure_column_name = csv_data_figure.columns.values[0:-2] 
            for figure_column in figure_column_name:
                csv_data[figure_column+'_res18'] = csv_data_figure[figure_column]
            
            figure_column_name2 = csv_data_figure_res101.columns.values[0:-2]
            for figure_column in figure_column_name2:
                csv_data[figure_column + '_res101'] = csv_data_figure_res101[figure_column]
            
            figure_column_name3 = csv_data_figure_res152.columns.values[0:-2]
            for figure_column in figure_column_name3:
                csv_data[figure_column + '_res152'] = csv_data_figure_res152[figure_column]

            figure_column_name = [(x + '_res18') for x in figure_column_name]
            figure_column_name2 = [(x + '_res101') for x in figure_column_name2]
            figure_column_name3 = [(x + '_res152') for x in figure_column_name3]
            column_name = np.concatenate((figure_column_name, figure_column_name2), axis=0)
            column_name = np.concatenate((column_name, figure_column_name3), axis=0)

        elif feature_type == 'clinical':
            column_name = csv_data.columns.values[26:39]

        elif feature_type == 'all':
            column_name = csv_data.columns.values
            column_name = np.delete(column_name, [21, 22, 23, 24, 25, 39]) 

            figure_column_name = csv_data_figure.columns.values[0:-2]  
            for figure_column in figure_column_name:
                csv_data[figure_column+'_res18'] = csv_data_figure[figure_column]
            
            figure_column_name2 = csv_data_figure_res101.columns.values[0:-2]  
            for figure_column in figure_column_name2:
                csv_data[figure_column + '_res101'] = csv_data_figure_res101[figure_column]
            
            figure_column_name3 = csv_data_figure_res152.columns.values[0:-2]  
            for figure_column in figure_column_name3:
                csv_data[figure_column + '_res152'] = csv_data_figure_res152[figure_column]

            figure_column_name = [(x + '_res18') for x in figure_column_name]
            figure_column_name2 = [(x + '_res101') for x in figure_column_name2]
            figure_column_name3 = [(x + '_res152') for x in figure_column_name3]

            column_name = np.concatenate((column_name, figure_column_name), axis=0)
            column_name = np.concatenate((column_name, figure_column_name2), axis=0)
            column_name = np.concatenate((column_name, figure_column_name3), axis=0)

        elif feature_type == 'ic+res':
            column_name = csv_data.columns.values[0:21]  

            figure_column_name = csv_data_figure.columns.values[0:-2]  
            for figure_column in figure_column_name:
                csv_data[figure_column+'_res18'] = csv_data_figure[figure_column]
            
            figure_column_name2 = csv_data_figure_res101.columns.values[0:-2]  
            for figure_column in figure_column_name2:
                csv_data[figure_column + '_res101'] = csv_data_figure_res101[figure_column]
            
            figure_column_name3 = csv_data_figure_res152.columns.values[0:-2] 
            for figure_column in figure_column_name3:
                csv_data[figure_column + '_res152'] = csv_data_figure_res152[figure_column]

            figure_column_name = [(x + '_res18') for x in figure_column_name]
            figure_column_name2 = [(x + '_res101') for x in figure_column_name2]
            figure_column_name3 = [(x + '_res152') for x in figure_column_name3]

            column_name = np.concatenate((column_name, figure_column_name), axis=0)
            column_name = np.concatenate((column_name, figure_column_name2), axis=0)
            column_name = np.concatenate((column_name, figure_column_name3), axis=0)

        elif feature_type == 'res+clinical':
            column_name = csv_data.columns.values[26:39] 

            figure_column_name = csv_data_figure.columns.values[0:-2] 
            for figure_column in figure_column_name:
                csv_data[figure_column+'_res18'] = csv_data_figure[figure_column]
            
            figure_column_name2 = csv_data_figure_res101.columns.values[0:-2]  
            for figure_column in figure_column_name2:
                csv_data[figure_column + '_res101'] = csv_data_figure_res101[figure_column]
            
            figure_column_name3 = csv_data_figure_res152.columns.values[0:-2] 
            for figure_column in figure_column_name3:
                csv_data[figure_column + '_res152'] = csv_data_figure_res152[figure_column]

            figure_column_name = [(x + '_res18') for x in figure_column_name]
            figure_column_name2 = [(x + '_res101') for x in figure_column_name2]
            figure_column_name3 = [(x + '_res152') for x in figure_column_name3]

            column_name = np.concatenate((column_name, figure_column_name), axis=0)
            column_name = np.concatenate((column_name, figure_column_name2), axis=0)
            column_name = np.concatenate((column_name, figure_column_name3), axis=0)

        elif feature_type == 'ic+clinical':
            column_name1 = csv_data.columns.values[0:21]  
            column_name2 = csv_data.columns.values[26:39]
            column_name = np.concatenate((column_name1, column_name2), axis=0)

        # judge the output OS or RELAPSE???
        if output_type == 'OS':
            output_name = ['OS', 'DEATH']  
        elif output_type == 'RFS':
            output_name = ['RFS','RELAPSE'] 

        train_num = 90
        train_Ind = csv_data.index.values[:train_num]
        test_Ind = csv_data.index.values[train_num:]
        
        x = csv_data.loc[train_Ind, column_name ].values.astype(np.float64)
        y = csv_data.loc[train_Ind, output_name ].values.astype(np.float64)
        y_test = csv_data.loc[test_Ind, output_name].values.astype(np.float64)

        # define the structured array ----survival_train
        if output_type == 'OS':
            dt = np.dtype([('DEATH', 'bool'), ('OS', 'f')])
        elif output_type == 'RFS':
            dt = np.dtype([('RELAPSE', 'bool'), ('RFS', 'f')])
        survival_train = np.array([ ( y[0,1],y[0,0] ) ],dtype=dt)
        for ii in range(1,len(train_Ind)):
            arr_train = np.array([ ( y[ii,1],y[ii,0] ) ],dtype=dt)
            survival_train = np.vstack([survival_train,arr_train])
        survival_train = np.squeeze(survival_train)

        # define the structured array ----survival_test
        survival_test = np.array([ ( y_test[0,1],y_test[0,0] ) ],dtype=dt)
        for ii in range(1,len(test_Ind)):
            arr_test = np.array([ ( y_test[ii,1],y_test[ii,0] ) ],dtype=dt)
            survival_test = np.vstack([survival_test,arr_test])
        survival_test = np.squeeze(survival_test)

        K = 5 #K-FOLD-CROSS-VALIDATION
        inter = int(90/K)
        for k in range(0, K):
            list_all = np.arange(0, train_num)
            list_cut = np.arange(k * inter, (k + 1) * inter)
            list_res = list(set(list_all) - set(list_cut))
            x_ = x[list_res,:]
            y_ = y[list_res,:]
            fit = glmnet(x = x_.copy(), y = y_.copy(), family = 'cox')
            
            coef = glmnetCoef(fit, s=scipy.float64([0.05]))
            coef_list.append(coef)
            
        coef = np.mean(np.array(coef_list),axis=0)
        products = 0
        score_ic = 0
        score_clinical = 0
        score_res = 0

        for j in range(0, len(coef)):
            products += (coef[j] * csv_data[column_name[j]])

        csv_data['score'] = products

        # caculate the AUC
        auc , mean_auc, sens_list, fpr_list, order = cumulative_dynamic_auc(survival_train=survival_train,
            survival_test=survival_test,estimate=products[train_num:] ,times=(1,2,3) )
        print('auc:',auc)
        list_auc.append(auc)
        
        if output_type == 'OS':
            OS_RFS_flag = 'OS_flag_1'
            OS_RFS_NAME = 'OS'
            Death_Relapse_Name = 'DEATH'
        elif output_type == 'RFS':
            OS_RFS_flag = 'RELAPSE_flag_1'
            OS_RFS_NAME = 'RFS'
            Death_Relapse_Name = 'RELAPSE'
        # use Log Rank Test to calculate the P values of early or late-stage NSCLC 
        test_data = ordered_data.loc[:,:] # early + late stage NSCLC
        group1 = test_data[test_data[OS_RFS_flag] == 1]
        group2 = test_data[test_data[OS_RFS_flag] == 0]
        if group1.shape[0]==0 or group2.shape[0]==0:
            continue
        T=group1[OS_RFS_NAME] 
        E=group1[Death_Relapse_Name] 
        T1=group2[OS_RFS_NAME] 
        E1=group2[Death_Relapse_Name] 
        results = logrank_test(T, T1, event_observed_A=E, event_observed_B=E1)
        if results.p_value:
            print('all_p=',results.p_value)
            list_p_all.append(results.p_value)
        
        test_data = ordered_data[ordered_data['STAGE(1=IA,2=IB)']==1] #early-stage NSCLC
        group1 = test_data[test_data[OS_RFS_flag]==1]
        group2 = test_data[test_data[OS_RFS_flag]==0]
        if group1.shape[0]==0 or group2.shape[0]==0:
            continue
        T=group1[OS_RFS_NAME] 
        E=group1[Death_Relapse_Name] 
        T1=group2[OS_RFS_NAME] 
        E1=group2[Death_Relapse_Name] 
        results=logrank_test(T,T1,event_observed_A=E, event_observed_B=E1)
        if results.p_value:
            print('early_p=',results.p_value)
            list_p_I.append(results.p_value)

        test_data = ordered_data[ordered_data['STAGE(1=IA,2=IB)']==2] #late-stage NSCLC
        group1 = test_data[test_data[OS_RFS_flag] == 1]
        group2 = test_data[test_data[OS_RFS_flag] == 0]
        if group1.shape[0]==0 or group2.shape[0]==0:
            continue
        T=group1[OS_RFS_NAME] 
        E=group1[Death_Relapse_Name] 
        T1=group2[OS_RFS_NAME] 
        E1=group2[Death_Relapse_Name] 
        results=logrank_test(T,T1,event_observed_A=E, event_observed_B=E1)
        if results.p_value:
            print('late_p=',results.p_value)
            list_p_II.append(results.p_value)

    print('mean_auc_100:', np.mean(np.array(list_auc), axis=0))

    # calculate how many times for (P<0.05 P<0.01 P<0.001)
    list_p = list_p_all
    count_P0050 = np.sum(np.array(list_p) < 0.05)
    count_P0010 = np.sum(np.array(list_p) < 0.01)
    count_P0001 = np.sum(np.array(list_p) < 0.001)
    print('early+late stage meaningful dataset num:')
    print('count_P0.05 = %d, count_P0.01 = %d, count_P0.001 = %d' % (count_P0050, count_P0010, count_P0001))
    
    list_p = list_p_I
    count_P0050 = np.sum(np.array(list_p) < 0.05)
    count_P0010 = np.sum(np.array(list_p) < 0.01)
    count_P0001 = np.sum(np.array(list_p) < 0.001)
    print('early stage meaningful dataset num:')
    print('count_P0.05 = %d, count_P0.01 = %d, count_P0.001 = %d' % (count_P0050, count_P0010, count_P0001))
    
    list_p = list_p_II
    count_P0050 = np.sum(np.array(list_p) < 0.05)
    count_P0010 = np.sum(np.array(list_p) < 0.01)
    count_P0001 = np.sum(np.array(list_p) < 0.001)
    print('late stage meaningful dataset num:')
    print('count_P0.05 = %d, count_P0.01 = %d, count_P0.001 = %d' % (count_P0050,count_P0010,count_P0001) )
