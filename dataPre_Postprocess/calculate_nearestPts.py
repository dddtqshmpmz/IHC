import numpy as np
import cv2
import csv
import os
import pandas as pd
import time

def calcuNearestPtsDis2(ptList1):
    ''' Find the nearest point of each point in ptList1 & return the mean min_distance
    Parameters
    ----------
    ptList1: numpy array
        points' array, shape:(x,2)
    Return
    ----------
    mean_Dis: float
        the mean value of the minimum distances
    '''
    if len(ptList1)<=1:
        print('error!')
        return 'error' 

    minDis_list = []
    for i in range(len(ptList1)):
        currentPt = ptList1[i,0:2]
        ptList2 = np.delete(ptList1,i,axis=0)
        disMat =  np.sqrt(np.sum(np.asarray(currentPt - ptList2)**2, axis=1).astype(np.float32) )
        minDis = disMat.min()
        minDis_list.append(minDis)

    minDisArr = np.array(minDis_list)
    mean_Dis = np.mean(minDisArr)
    return mean_Dis

def calcuNearestPtsDis(ptList1, ptList2):
    ''' Find the nearest point of each point in ptList1 from ptList2 
    & return the mean min_distance
    Parameters
    ----------
    ptList1: numpy array
        points' array, shape:(x,2)
    ptList2: numpy array
        points' array, shape:(x,2)
    Return
    ----------
    mean_Dis: float
        the mean value of the minimum distances
    '''
    if (not len(ptList2)) or (not len(ptList1)):
        print('error!')
        return 'error' 
    minDis_list = []
    for i in range(len(ptList1)):
        currentPt = ptList1[i,0:2]
        disMat =  np.sqrt(np.sum(np.asarray(currentPt - ptList2)**2, axis=1).astype(np.float32) )
        minDis = disMat.min()
        minDis_list.append(minDis)
    minDisArr = np.array(minDis_list)
    mean_Dis = np.mean(minDisArr)
    
    return mean_Dis

def calcuNearestPts(csvName1, csvName2):
    ptList1_csv = pd.read_csv(csvName1,usecols=['x_cord', 'y_cord'])
    ptList2_csv = pd.read_csv(csvName2,usecols=['x_cord', 'y_cord'])
    ptList1 = ptList1_csv.values[:,:2]
    ptList2 = ptList2_csv.values[:,:2]

    minDisInd_list = []
    for i in range(len(ptList1)):
        currentPt = ptList1[i,0:2]
        disMat =  np.sqrt(np.sum(np.asarray(currentPt - ptList2)**2, axis=1))
        minDisInd = np.argmin(disMat)
        minDisInd_list.append(minDisInd)

    minDisInd = np.array(minDisInd_list).reshape(-1,1)
    ptList1_csv = pd.concat([ptList1_csv, pd.DataFrame( columns=['nearestInd'],data = minDisInd)], axis=1)
    ptList1_csv.to_csv(csvName1,index=False)
    return minDisInd


def drawDisPic(picInd):
    picName = 'patients_dataset/image/'+ picInd +'.png'
    img = cv2.imread(picName)
    csvName1='patients_dataset/data_csv/'+picInd+'other_tumour_pts.csv'
    csvName2='patients_dataset/data_csv/'+picInd+'other_lymph_pts.csv'
    ptList1_csv = pd.read_csv(csvName1)
    ptList2_csv = pd.read_csv(csvName2)
    ptList1 = ptList1_csv.values
    ptList2 = ptList2_csv.values
    for i in range(len(ptList1)):
        img = cv2.circle(img, tuple(ptList1[i,:2]), 3 , (0, 0, 255), -1 )
        img = cv2.line(img, tuple(ptList1[i,:2]) , tuple(ptList2[ ptList1[i,2] ,:2]), (0,255,0), 1)
    for i in range(len(ptList2)):
        img = cv2.circle(img, tuple(ptList2[i,:2]), 3 , (255, 0, 0), -1 )
    
    cv2.imwrite( picInd+'_dis.png',img)

def drawDistancePic(disName1, disName2, picID):
    ''' Draw & save the distance pics
    Parameters
    ----------
    disName1,disName2: str
        such as 'positive_lymph', 'all_tumour'
    picID: str
        the patient's ID
    '''
    cellName_color = {'other_lymph': (255, 0, 0), 'positive_lymph': (255, 255, 0),
        'other_tumour': (0, 0, 255), 'positive_tumour': (0, 255, 0)}
    ptline_color = {'positive_lymph': (0,0,255), 'positive_tumour': (0,0,255),
        'ptumour_plymph': (51, 97, 235), 'other_tumour': (0, 255, 0)}

    if (disName1 == 'all_tumour' and disName2 == 'all_lymph')  or (disName1 == 'all_tumour' and disName2 == 'positive_lymph'):
        line_color = (0,255,255)

    elif disName1 == 'positive_tumour' and disName2 == 'positive_lymph':
        line_color = (51, 97, 235)
    else:
        line_color = ptline_color[disName1]

    csv_dir = '/data/Datasets/MediImgExp/data_csv'
    img_dir = '/data/Datasets/MediImgExp/image'
    if disName1 == 'all_tumour' and disName2 == 'positive_lymph':
        dis1_csv = pd.read_csv(csv_dir + '/' + picID + 'positive_tumour' + '_pts.csv', usecols=['x_cord', 'y_cord'])
        dis2_csv = pd.read_csv(csv_dir + '/' + picID + 'other_tumour' + '_pts.csv', usecols=['x_cord', 'y_cord'])
        dis3_csv = pd.read_csv(csv_dir + '/' + picID + 'positive_lymph' + '_pts.csv', usecols=['x_cord', 'y_cord'])
        ptList1 = dis1_csv.values[:,:2]
        ptList2 = dis2_csv.values[:,:2]
        ptList3 = dis3_csv.values[:,:2]

        # positive tumour: find the nearest lymph cell
        minDisInd_list = []
        for i in range(len(ptList1)):
            currentPt = ptList1[i,:]
            disMat =  np.sqrt(np.sum(np.asarray(currentPt - ptList3)**2, axis=1))
            minDisInd = np.argmin(disMat)
            minDisInd_list.append(minDisInd)

        minDisInd = np.array(minDisInd_list).reshape(-1,1)
        dis1_csv = pd.concat([dis1_csv, pd.DataFrame(columns=['nearestInd'], data=minDisInd)], axis=1)

        # other tumour: find the nearest lymph cell
        minDisInd_list = []
        for i in range(len(ptList2)):
            currentPt = ptList2[i,:]
            disMat =  np.sqrt(np.sum(np.asarray(currentPt - ptList3)**2, axis=1))
            minDisInd = np.argmin(disMat)
            minDisInd_list.append(minDisInd)

        minDisInd = np.array(minDisInd_list).reshape(-1,1)
        dis2_csv = pd.concat([dis2_csv, pd.DataFrame(columns=['nearestInd'], data=minDisInd)], axis=1)
        
        img = cv2.imread(img_dir + '/' + picID + '.jpg')
        ptList1 = dis1_csv.values
        for i in range(len(ptList1)):
            img = cv2.line(img, tuple(ptList1[i,:2]), tuple(ptList3[ptList1[i, 2],:2]), line_color, 1)
        ptList2 = dis2_csv.values
        for i in range(len(ptList2)):
            img = cv2.line(img, tuple(ptList2[i,:2]), tuple(ptList3[ptList2[i, 2],:2]), line_color, 1)
            
        for i in range(len(ptList1)):
            img = cv2.circle(img, tuple(ptList1[i,:2]), 4, (0, 255, 0), -1)
        for i in range(len(ptList2)):
            img = cv2.circle(img, tuple(ptList2[i,:2]), 4, (0, 0, 255), -1)
        for i in range(len(ptList3)):
            img = cv2.circle(img, tuple(ptList3[i,:2]), 4, (255, 255, 0), -1)
        
        cv2.imwrite(picID + disName1 + '_' + disName2 + '_dis.png', img)

    elif disName1 == 'all_tumour' and disName2 == 'all_lymph':
        dis1_csv = pd.read_csv(csv_dir + '/' + picID + 'positive_tumour' + '_pts.csv', usecols=['x_cord', 'y_cord'])
        dis2_csv = pd.read_csv(csv_dir + '/' + picID + 'other_tumour' + '_pts.csv', usecols=['x_cord', 'y_cord'])
        dis3_csv = pd.read_csv(csv_dir + '/' + picID + 'positive_lymph' + '_pts.csv', usecols=['x_cord', 'y_cord'])
        dis4_csv = pd.read_csv(csv_dir + '/' + picID + 'other_lymph' + '_pts.csv', usecols=['x_cord', 'y_cord'])
        ptList1 = dis1_csv.values[:,:2]
        ptList2 = dis2_csv.values[:,:2]
        ptList3 = dis3_csv.values[:,:2]
        ptList4 = dis4_csv.values[:,:2]
        ptList6 = np.concatenate((ptList3, ptList4), axis=0)

        minDisInd_list = []
        for i in range(len(ptList1)):
            currentPt = ptList1[i,:]
            disMat =  np.sqrt(np.sum(np.asarray(currentPt - ptList6)**2, axis=1))
            minDisInd = np.argmin(disMat)
            minDisInd_list.append(minDisInd)
 
        minDisInd = np.array(minDisInd_list).reshape(-1,1)
        dis1_csv = pd.concat([dis1_csv, pd.DataFrame(columns=['nearestInd'], data=minDisInd)], axis=1)

        minDisInd_list = []
        for i in range(len(ptList2)):
            currentPt = ptList2[i,:]
            disMat =  np.sqrt(np.sum(np.asarray(currentPt - ptList6)**2, axis=1))
            minDisInd = np.argmin(disMat)
            minDisInd_list.append(minDisInd)
        minDisInd = np.array(minDisInd_list).reshape(-1,1)
        dis2_csv = pd.concat([dis2_csv, pd.DataFrame(columns=['nearestInd'], data=minDisInd)], axis=1)
        
        img = cv2.imread(img_dir + '/' + picID + '.jpg')
        ptList1 = dis1_csv.values
        for i in range(len(ptList1)):
            img = cv2.line(img, tuple(ptList1[i,:2]), tuple(ptList6[ptList1[i, 2],:2]), line_color, 1)
        ptList2 = dis2_csv.values
        for i in range(len(ptList2)):
            img = cv2.line(img, tuple(ptList2[i,:2]), tuple(ptList6[ptList2[i, 2],:2]), line_color, 1)
            
        for i in range(len(ptList1)):
            img = cv2.circle(img, tuple(ptList1[i,:2]), 4, (0, 255, 0), -1)
        for i in range(len(ptList2)):
            img = cv2.circle(img, tuple(ptList2[i,:2]), 4, (0, 0, 255), -1)
        for i in range(len(ptList3)):
            img = cv2.circle(img, tuple(ptList3[i,:2]), 4, (255, 255, 0), -1)
        for i in range(len(ptList4)):
            img = cv2.circle(img, tuple(ptList4[i,:2]), 4, (255, 0, 0), -1)
        
        cv2.imwrite(picID + disName1 + '_' + disName2 + '_dis.png', img)

    elif disName1 != disName2:
        dis1_csv = pd.read_csv(csv_dir + '/' + picID + disName1 + '_pts.csv', usecols=['x_cord', 'y_cord'])
        dis2_csv = pd.read_csv(csv_dir + '/' + picID + disName2 + '_pts.csv', usecols=['x_cord', 'y_cord'])
        ptList1 = dis1_csv.values[:,:2]
        ptList2 = dis2_csv.values[:,:2]

        minDisInd_list = []
        for i in range(len(ptList1)):
            currentPt = ptList1[i,:]
            disMat =  np.sqrt(np.sum(np.asarray(currentPt - ptList2)**2, axis=1))
            minDisInd = np.argmin(disMat)
            minDisInd_list.append(minDisInd)

        minDisInd = np.array(minDisInd_list).reshape(-1,1)
        dis1_csv = pd.concat([dis1_csv, pd.DataFrame( columns=['nearestInd'],data = minDisInd)], axis=1)
    
        img = cv2.imread(img_dir + '/' + picID + '.jpg')
        img[:,:, 0] = 255
        img[:,:, 1] = 255
        img[:,:, 2] = 255
        ptList1 = dis1_csv.values
        for i in range(len(ptList1)):
            img = cv2.line(img, tuple(ptList1[i,:2]) , tuple(ptList2[ ptList1[i,2] ,:2]), line_color, 1)
        for i in range(len(ptList1)):
            img = cv2.circle(img, tuple(ptList1[i,:2]), 5, cellName_color[disName1], -1)
        for i in range(len(ptList2)):
            img = cv2.circle(img, tuple(ptList2[i,:2]), 5, cellName_color[disName2], -1)
        cv2.imwrite(picID + disName1 + '_' + disName2 + '_dis.png', img)
    
    elif disName1 == disName2:
        dis1_csv = pd.read_csv(csv_dir + '/' + picID + disName1 + '_pts.csv', usecols=['x_cord', 'y_cord'])
        ptList1 = dis1_csv.values[:,:2]

        minDisInd_list = []
        for i in range(len(ptList1)):
            currentPt = ptList1[i, :2]
            disMat = np.sqrt(np.sum(np.asarray(currentPt - ptList1)** 2, axis=1).astype(np.float32))
            minDisInd = np.argmin(disMat)
            disMat[minDisInd] = 1000.0
            minDisInd = np.argmin(disMat)
            minDisInd_list.append(minDisInd)

        minDisInd = np.array(minDisInd_list).reshape(-1,1)
        dis1_csv = pd.concat([dis1_csv, pd.DataFrame( columns=['nearestInd'],data = minDisInd)], axis=1)
        
        img = cv2.imread(img_dir + '/' + picID + '.jpg')
        img[:,:, 0] = 255
        img[:,:, 1] = 255
        img[:,:, 2] = 255
        ptList1 = dis1_csv.values
        for i in range(len(ptList1)):
            img = cv2.line(img, tuple(ptList1[i,:2]), tuple(ptList1[ptList1[i, 2],:2]), line_color, 1)
        for i in range(len(ptList1)):
            img = cv2.circle(img, tuple(ptList1[i,:2]), 5, cellName_color[disName1], -1)
        
        cv2.imwrite(picID + disName1 + '_dis.png', img)


def getAllPicsDisCSV():
    '''
    Get all distance data from the saved csv files (get from the above functions)
    '''
    base_dir = '/data/Datasets/MediImgExp'
    f = open( base_dir + '/' + 'AllDisData.csv','w',encoding='utf-8',newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow([ 'Ind','PosiTumourRatio','PosiLymphRatio',
        'DisTumourLymph','DisPosiTumour','DisPosiLymph',
        'DisPosiTumourPosiLymph','DisTumourPosiLymph'])

    process_dir = base_dir + '/process'
    csv_dir = base_dir + '/data_csv'
    pic_name = os.listdir(process_dir)
    picIDList = []
    for pic_name_ in pic_name:
        picIDList.append( pic_name_.split('_')[0] )
    for picID in picIDList:
        list_data = []
        list_data.append(picID)

        # PosiTumourRatio
        PosiTumourCsv = pd.read_csv( csv_dir+'/'+ picID +'positive_tumour_pts.csv')
        OtherTumourCsv = pd.read_csv( csv_dir+'/'+ picID +'other_tumour_pts.csv')
        Num_PosiTumour = PosiTumourCsv.shape[0]
        Num_OtherTumour = OtherTumourCsv.shape[0]
        if (Num_PosiTumour + Num_OtherTumour)!=0 :
            PosiTumourRatio =  Num_PosiTumour / (Num_PosiTumour + Num_OtherTumour)
        else:
            PosiTumourRatio = 'error'
        list_data.append(PosiTumourRatio)

        # PosiLymphRatio
        PosiLymphCsv = pd.read_csv( csv_dir+'/'+ picID +'positive_lymph_pts.csv')
        OtherLymphCsv = pd.read_csv( csv_dir+'/'+ picID +'other_lymph_pts.csv')
        Num_PosiLymph = PosiLymphCsv.shape[0]
        Num_OtherLymph = OtherLymphCsv.shape[0]
        if (Num_PosiLymph + Num_OtherLymph)!=0 :
            PosiLymphRatio =  Num_PosiLymph / (Num_PosiLymph + Num_OtherLymph)
        else:
            PosiLymphRatio = 'error'
        list_data.append(PosiLymphRatio)

        # DisTumourLymph
        ptList1_csv = pd.read_csv(csv_dir+'/'+ picID +'positive_tumour_pts.csv',usecols=['x_cord', 'y_cord'])
        ptList2_csv = pd.read_csv(csv_dir+'/'+ picID +'positive_lymph_pts.csv',usecols=['x_cord', 'y_cord'])
        ptList1 = ptList1_csv.values[:,:2]
        ptList2 = ptList2_csv.values[:,:2]
        ptList3_csv = pd.read_csv(csv_dir+'/'+ picID +'other_tumour_pts.csv',usecols=['x_cord', 'y_cord'])
        ptList4_csv = pd.read_csv(csv_dir+'/'+ picID +'other_lymph_pts.csv',usecols=['x_cord', 'y_cord'])
        ptList3 = ptList3_csv.values[:,:2]
        ptList4 = ptList4_csv.values[:,:2]
        ptList1 = np.concatenate((ptList1,ptList3), axis=0)
        ptList2 = np.concatenate((ptList2,ptList4), axis=0)
        DisTumourLymph = calcuNearestPtsDis(ptList1, ptList2)
        list_data.append(DisTumourLymph)

        # DisPosiTumour
        ptList1_csv = pd.read_csv(csv_dir+'/'+ picID +'positive_tumour_pts.csv',usecols=['x_cord', 'y_cord'])
        ptList1 = ptList1_csv.values[:,:2]
        DisPosiTumour = calcuNearestPtsDis2(ptList1)
        list_data.append(DisPosiTumour)

        # DisPosiLymph
        ptList1_csv = pd.read_csv(csv_dir+'/'+ picID +'positive_lymph_pts.csv',usecols=['x_cord', 'y_cord'])
        ptList1 = ptList1_csv.values[:,:2]
        DisPosiLymph = calcuNearestPtsDis2(ptList1)
        list_data.append(DisPosiLymph)

        # DisPosiTumourPosiLymph
        ptList1_csv = pd.read_csv(csv_dir+'/'+ picID +'positive_tumour_pts.csv',usecols=['x_cord', 'y_cord'])
        ptList2_csv = pd.read_csv(csv_dir+'/'+ picID +'positive_lymph_pts.csv',usecols=['x_cord', 'y_cord'])
        ptList1 = ptList1_csv.values[:,:2]
        ptList2 = ptList2_csv.values[:,:2]
        DisPosiTumourPosiLymph = calcuNearestPtsDis(ptList1, ptList2)
        list_data.append(DisPosiTumourPosiLymph)

        # DisTumourPosiLymph
        ptList1_csv = pd.read_csv(csv_dir+'/'+ picID +'positive_tumour_pts.csv',usecols=['x_cord', 'y_cord'])
        ptList2_csv = pd.read_csv(csv_dir+'/'+ picID +'positive_lymph_pts.csv',usecols=['x_cord', 'y_cord'])
        ptList1 = ptList1_csv.values[:,:2]
        ptList2 = ptList2_csv.values[:,:2]
        ptList3_csv = pd.read_csv(csv_dir+'/'+ picID +'other_tumour_pts.csv',usecols=['x_cord', 'y_cord'])
        ptList3 = ptList3_csv.values[:,:2]
        ptList1 = np.concatenate((ptList1,ptList3), axis=0)
        DisTumourPosiLymph = calcuNearestPtsDis(ptList1, ptList2)
        list_data.append(DisTumourPosiLymph)

        csv_writer.writerow(list_data)
    

def adjustToMultiCSV():
    '''
    Divide the AllDisData.csv into  6+1=7 csv
    '''
    base_dir = '/data/Datasets/MediImgExp'
    alldata = pd.read_csv( base_dir + '/' + 'AllDisData.csv' )
    IndData = alldata['Ind'].values

    patient_Ind = []
    for IndName in IndData:
        patient_Ind.append(IndName.split('-')[0])
    patient_Ind = np.unique(patient_Ind)
    patient_Ind = sorted( list(map(int,patient_Ind)) )
    column_name = ['Ind','2D','3D','GAL9','LAG3','MHC','OX40','OX40L','PD1','PDL1','TIM3']

    # stage 1 calculate the 6 csv (10 cols for each csv)
    DisPosiTumour = pd.DataFrame(columns=column_name,index= patient_Ind)
    DisPosiTumour['Ind'] = patient_Ind
    patient_Id = patient_Ind 
    column_names = ['2D','3D','GAL9','LAG3','MHC','OX40','OX40L','PD1','PDL1','TIM3']
    for patient in patient_Id:
        for column in column_names:
            combine_name = str(patient) + '-' + column 
            exist_flag = (alldata['Ind'].str[0:len(combine_name)]== combine_name).any()
            if not exist_flag:
                continue
            valid_slice = alldata[ alldata['Ind'].str[0:len(combine_name)]== combine_name ]
            arr = valid_slice['DisTumourPosiLymph'].values
            if  arr.__contains__('error'):
                arr = np.setdiff1d(arr, ['error'])
                if not arr.shape[0]:
                    continue
            valid_slice_mean = np.mean( arr.astype(np.float32))
            DisPosiTumour.loc[ patient ,column ] = valid_slice_mean
    DisPosiTumour.to_csv( base_dir + '/' + 'DisTumourPosiLymph.csv',index=False )
    
    # stage 2 add the outputs (4 cols)
    all_data_name = base_dir + '/' + 'alldata2.csv'
    all_data = pd.read_csv(all_data_name)
    all_data.index = all_data['Ind']
    valid_columns = ['RELAPSE','RFS','DEATH','OS']
    valid_slice = all_data.loc[ patient_Ind, valid_columns ]
    DisPosiTumour = pd.read_csv( base_dir + '/' + 'PosiTumourRatio.csv',index_col=0)
    DisPosiTumour = pd.concat([DisPosiTumour,valid_slice],axis = 1)
    DisPosiTumour.to_csv( base_dir + '/' + 'PosiTumourRatio.csv' )
    
    # stage 3 calculate DisTumourLymph (use all markers' mean values)
    DisTumourLymph = pd.DataFrame(columns=['mean_10markers'],index= patient_Ind)
    patient_Id = patient_Ind 
    column_names = [ 'mean_10markers']
    for patient in patient_Id:
        for column in column_names:
            combine_name = str(patient) + '-' 
            exist_flag = (alldata['Ind'].str[0:len(combine_name)]== combine_name).any()
            if not exist_flag:
                continue
            valid_slice = alldata[ alldata['Ind'].str[0:len(combine_name)]== combine_name ]
            arr = valid_slice['DisTumourLymph'].values
            if  arr.__contains__('error'):
                arr = np.setdiff1d(arr, ['error'])
                if not arr.shape[0]:
                    continue
            valid_slice_mean = np.mean( arr.astype(np.float32))
            DisTumourLymph.loc[ patient ,column ] = valid_slice_mean
    
    DisTumourLymph.to_csv( base_dir + '/' + 'DisTumourLymph.csv' )
    all_data_name = base_dir + '/' + 'alldata2.csv'
    all_data = pd.read_csv(all_data_name)
    all_data.index = all_data['Ind']
    valid_columns = ['RELAPSE','RFS','DEATH','OS']
    valid_slice = all_data.loc[ patient_Ind, valid_columns] 
    DisTumourLymph = pd.concat([DisTumourLymph,valid_slice],axis = 1)
    DisTumourLymph.to_csv( base_dir + '/' + 'DisTumourLymph.csv')
    

def getAllFeatureCSV():
    base_dir = '/data/Datasets/MediImgExp/csv'
    alldata = pd.read_csv( base_dir + '/' + 'AllDisData.csv' )
    oridata = pd.read_csv( base_dir + '/' + 'alldata2.csv',index_col=0 )
    ori_columns = oridata.columns.values
    ori_columns = ori_columns[4:-4] # original 40 feature names
    csv_name = [ 'DisPosiLymph','DisPosiTumour', 'DisPosiTumourPosiLymph',
        'DisTumourPosiLymph','PosiLymphRatio','PosiTumourRatio', 'DisTumourLymph']
    
    meaningful_feature_OS = { 'DisPosiLymph_PD1':24,'DisPosiLymph_OX40L':48.14,'DisPosiLymph_OX40':98.33,'DisPosiLymph_3D':13.44, 
        'DisPosiTumour_TIM3':546.86, 'DisPosiTumour_GAL9':85.97,'DisPosiTumour_2D':24.22,
        'DisPosiTumourPosiLymph_3D':20.88, 'DisPosiTumourPosiLymph_MHC':18.68,'DisPosiTumourPosiLymph_OX40L':40.02,
        'DisTumourPosiLymph_OX40L':173.56, 'DisTumourPosiLymph_2D':223.71,'DisTumourPosiLymph_3D':21.19,'DisTumourPosiLymph_OX40':445.89,
        'PosiLymphRatio_3D':0.97, 'PosiLymphRatio_OX40':0.44,'PosiLymphRatio_GAL9':0.23,'PosiLymphRatio_2D':0.37,
        'PosiTumourRatio_MHC':0.93,'PosiTumourRatio_GAL9':0.41,'PosiTumourRatio_3D':1.0 }

    IndData = alldata['Ind'].values
    patient_Ind = []
    for IndName in IndData:
        patient_Ind.append(IndName.split('-')[0])
    patient_Ind = np.unique(patient_Ind)
    patient_Ind = sorted( list(map(int,patient_Ind)) ) 
    
    # create a super_csv including all features 61+40=101 features / 4 outputs
    super_csv = pd.DataFrame(index= patient_Ind)
    super_csv_copy = pd.DataFrame(index= patient_Ind)

    super_csv['Ind'] = patient_Ind
    for column in ori_columns: # 40 features
        super_csv[column] = oridata.loc[ patient_Ind,column] 
    for i in range(0,7):
        csvName = os.path.join(base_dir,csv_name[i] + '.csv')
        csvData = pd.read_csv(csvName,index_col=0)
        if i == 6: # 1 feature
            column_name = csvData.columns.values[:1] 
            super_csv[ csv_name[i]+'_'+column_name[0] ] = csvData.loc[ patient_Ind, column_name[0] ]
            output_name = csvData.columns.values[-4:]
            for k in range(0,4):
                super_csv[ output_name[k] ] = csvData.loc[ patient_Ind,output_name[k] ]
            break
        column_name = csvData.columns.values[:10]
        for j in range(0,10): # 6*10 = 60 features
            super_csv[ csv_name[i]+'_'+column_name[j] ] = csvData.loc[ patient_Ind, column_name[j] ]
    
    for key, values in meaningful_feature_OS.items():
        super_csv_copy[key] = super_csv.loc[patient_Ind, key ]
        super_csv_copy.loc[ super_csv_copy[key] <= values, key] = 0
        super_csv_copy.loc[ super_csv_copy[key] > values, key ] = 1

    for k in range(0,4):
        super_csv_copy[ output_name[k] ] = csvData.loc[ patient_Ind,output_name[k] ]
    super_csv_copy = super_csv_copy.dropna(axis=0,how='any')

    #super_csv.to_csv(base_dir+'/'+ 'super1.csv',index =False)
    #super_csv.to_csv(base_dir+'/'+ 'super2.csv',index =False)
    #super_csv_copy.to_csv(base_dir+'/'+ 'super3.csv',index =True)
    #super_csv_copy.to_csv(base_dir+'/'+ 'super4.csv',index =True)


def getAllFeatureCSV2():
    base_dir = '/data/Datasets/MediImgExp/csv'
    alldata = pd.read_csv( base_dir + '/' + 'AllDisData.csv' )
    oridata = pd.read_csv( base_dir + '/' + 'alldata2.csv',index_col=0 )
    ori_columns = oridata.columns.values
    ori_columns = np.concatenate(( ori_columns[19:32],ori_columns[44:48] ) ,axis=0 ) # original 40 feature names
    csv_name = [ 'DisPosiLymph','DisPosiTumour', 'DisPosiTumourPosiLymph',
        'DisTumourPosiLymph','PosiLymphRatio','PosiTumourRatio', 'DisTumourLymph']
    
    meaningful_feature_OS = { 'DisPosiLymph_OX40L':61.55,
        'DisPosiTumour_GAL9':119.40,
        'DisPosiTumourPosiLymph_3D':20.28, 'DisPosiTumourPosiLymph_OX40L':40.02,
        'DisTumourPosiLymph_3D':21.19, 'DisTumourPosiLymph_MHC':135.56,
        'PosiLymphRatio_3D':0.97, 'PosiLymphRatio_OX40':0.52,'PosiLymphRatio_GAL9':0.2 }

    IndData = alldata['Ind'].values
    patient_Ind = []
    for IndName in IndData:
        patient_Ind.append(IndName.split('-')[0])
    patient_Ind = np.unique(patient_Ind)
    patient_Ind = sorted( list(map(int,patient_Ind)) )
    
    # create a super_csv including all features 61+xx features / 4 outputs
    super_csv = pd.DataFrame(index= patient_Ind)
    super_csv_copy = pd.DataFrame(index= patient_Ind)

    super_csv['Ind'] = patient_Ind
    for column in ori_columns: # 40 features
        super_csv[column] = oridata.loc[ patient_Ind,column] 
    for i in range(0,7):
        csvName = os.path.join(base_dir,csv_name[i] + '.csv')
        csvData = pd.read_csv(csvName,index_col=0)
        if i == 6: # 1 feature
            column_name = csvData.columns.values[:1] 
            super_csv[ csv_name[i]+'_'+column_name[0] ] = csvData.loc[ patient_Ind, column_name[0] ]
            output_name = csvData.columns.values[-4:]
            for k in range(0,4):
                super_csv[ output_name[k] ] = csvData.loc[ patient_Ind,output_name[k] ]
            break
        column_name = csvData.columns.values[:10]
        for j in range(0,10): # 6*10 = 60 features
            super_csv[ csv_name[i]+'_'+column_name[j] ] = csvData.loc[ patient_Ind, column_name[j] ]
    
    for key, values in meaningful_feature_OS.items():
        super_csv_copy[key] = super_csv.loc[patient_Ind, key ]
        super_csv_copy.loc[ super_csv_copy[key] <= values, key] = 0
        super_csv_copy.loc[ super_csv_copy[key] > values, key ] = 1

    for k in range(0,4):
        super_csv_copy[ output_name[k] ] = csvData.loc[ patient_Ind,output_name[k] ]
    #super_csv_copy = super_csv_copy.dropna(axis=0,how='any')
    #super_csv.to_csv(base_dir+'/rfs_csv'+'/'+ 'super1.csv',index =False)
    #super_csv.to_csv(base_dir+'/'+ 'super2.csv',index =False)
    super_csv_copy.to_csv(base_dir+'/rfs_csv'+'/'+ 'super3.csv',index =True)
    #super_csv_copy.to_csv(base_dir+'/'+ 'super4.csv',index =True)

if __name__ == '__main__':

    getAllPicsDisCSV()
    adjustToMultiCSV()
    getAllFeatureCSV2()
    
    # draw the distance pics
    drawDistancePic(disName1='all_tumour', disName2='positive_lymph', picID='0-GAL9-1')

    