import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
import random
import csv
import time
import pandas as pd

def processImg(picId,dataset_dir,blue_thresh, brown_thresh):
    '''Process the images with segmented masks and HSV thresholds 
    Parameters
    ----------
    picId: str
        the patient's ID
    dataset_dir: str
        the dataset's directory
    blue_thresh: numpy array
        the H,S,V threshold values of postive cells
    brown_thresh: numpy array
        the H,S,V threshold values of negative cells
    '''
    # Init the directory
    img_dir = dataset_dir+'/image'
    process_dir =  dataset_dir+'/process'
    mask_dir =  dataset_dir+'/mask'

    img_name = img_dir + '/' + picId + '.jpg'
    img = cv2.imread(img_name)
    imgCopy= img.copy()
    drawCopy = img.copy()

    #get the blue thresh (positive cells)
    gray= cv2.cvtColor(imgCopy, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(imgCopy, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv)
    blue_label = (H > blue_thresh[0]) * (H < blue_thresh[1]) * (S >blue_thresh[2] )* (S  < blue_thresh[3] ) * (V > blue_thresh[4])  #(H > 100) * (H < 115) * (S >10 )* (S  < 120 ) * (V > 110)
    blue_thresh= gray*blue_label
    blue_thresh[blue_label]=255
    
    #brown label & screen out the positive(stained) cells
    brown_label = (H > brown_thresh[0]) * (H < brown_thresh[1]) * (S >brown_thresh[2] )* (S  < brown_thresh[3] ) * (V > brown_thresh[4])* (V < int(brown_thresh[5])) # (H > 0) * (H < 30) * (S >0 )* (S  < 70 ) * (V > 130)* (V < 200)
    brown_thresh= gray*brown_label
    brown_thresh[brown_label] = 255
    brown_thresh[~brown_label] = 0
    tx=2
    ty=0
    mat_translation=np.float32([[1,0,tx],[0,1,ty]]) 
    translated_brown=cv2.warpAffine(brown_thresh,mat_translation,(brown_thresh.shape[1],brown_thresh.shape[0]) )
    brown_translated_label= (translated_brown>0)
    blue_translated_thresh= blue_thresh * brown_translated_label

    # screen out the postive tumour cells
    mask_name = mask_dir + '/' + picId + '.png'
    positive_tumour_mask = cv2.imread(mask_name,0)
    positive_tumour_label = positive_tumour_mask>0
    positive_tumour_mask = positive_tumour_label * brown_thresh 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5)) 
    positive_tumour_mask = cv2.erode(positive_tumour_mask,kernel) 

    connectedNum, connectedLabel, connected_stats, connected_centroids=cv2.connectedComponentsWithStats(positive_tumour_mask )
    f = open( dataset_dir + '/data_csv/' + picId + 'positive_tumour_pts.csv','w',encoding='utf-8',newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow([ 'x_cord','y_cord'])
    for chosen_Id in range(1,min(connectedNum,10000)):
        thePt = connected_centroids[chosen_Id,:].astype(np.int)
        csv_writer.writerow(thePt ) 
        drawCopy = cv2.circle(drawCopy, tuple(thePt), 3 , (0, 255, 0), 3 )

    # screen out the postive lymph cells
    positive_lymph_label = ~positive_tumour_label
    positive_lymph_mask = positive_lymph_label * brown_thresh 
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5)) 
    positive_lymph_mask = cv2.erode(positive_lymph_mask,kernel)
    
    connectedNum, connectedLabel, connected_stats, connected_centroids=cv2.connectedComponentsWithStats(positive_lymph_mask )
    f = open( dataset_dir + '/data_csv/' + picId + 'positive_lymph_pts.csv','w',encoding='utf-8',newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow([ 'x_cord','y_cord'])
    for chosen_Id in range(1,min(connectedNum,10000)):
        thePt = connected_centroids[chosen_Id,:].astype(np.int)
        csv_writer.writerow(thePt ) 
        drawCopy = cv2.circle(drawCopy, tuple(thePt), 3 , (255, 255, 0), 3 )

    #get all tumour cells
    mask_name = mask_dir + '/' + picId + '.png'
    mask = cv2.imread(mask_name,0)
    mask_label = mask>0
    blue_tumour_label = mask_label * blue_label
    mask[~blue_tumour_label]=0
    mask[blue_tumour_label]=255
    mask[positive_tumour_mask>0]=0
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5)) #3,3
    mask = cv2.erode(mask,kernel)

    # mean filtering + sample from the connectedComponents
    blur_mask = cv2.blur(mask,(3,3))
    blur_mask[blur_mask<255]=0
    connectedNum, connectedLabel, connected_stats, connected_centroids=cv2.connectedComponentsWithStats(blur_mask )

    f = open( dataset_dir + '/data_csv/' + picId+ 'other_tumour_pts.csv','w',encoding='utf-8',newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow([ 'x_cord','y_cord'])
    for chosen_Id in range(1,min(connectedNum,10000)):
        thePt = connected_centroids[chosen_Id,:].astype(np.int)
        csv_writer.writerow(thePt ) 
        drawCopy = cv2.circle(drawCopy, tuple(thePt), 3 , (0, 0, 255), 3 )

    #get all lymph cells
    lymph_mask = cv2.imread(mask_name,0)
    blue_lymph_label = blue_label * (~blue_tumour_label)
    lymph_mask[~blue_lymph_label]=0
    lymph_mask[blue_lymph_label]=255
    lymph_mask[positive_lymph_mask>0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5, 5)) 
    lymph_mask = cv2.erode(lymph_mask,kernel) 

    # mean filtering + sample from the connectedComponents
    blur_lymph_mask = cv2.blur(lymph_mask,(3,3))
    blur_lymph_mask[blur_lymph_mask<255]=0

    connectedNum, connectedLabel, connected_stats, connected_centroids=cv2.connectedComponentsWithStats(blur_lymph_mask )
    f = open( dataset_dir + '/data_csv/' + picId + 'other_lymph_pts.csv','w',encoding='utf-8',newline="")
    csv_writer = csv.writer(f)
    csv_writer.writerow([ 'x_cord','y_cord'])
    for chosen_Id in range(1,min(connectedNum,10000)):
        thePt = connected_centroids[chosen_Id,:].astype(np.int)
        csv_writer.writerow(thePt ) 
        drawCopy = cv2.circle(drawCopy, tuple(thePt), 3 , (255, 0, 0) , 3 )

    cv2.imwrite(process_dir+'/'+picId+'_draw_all_cells.png',drawCopy)

def getAllCellsPics():
    '''
    Get positive or negative tumour/lymph cells for all images
    The png files and csv files are saved in the related directory
    '''
    dataset_dir = '/data/Datasets/MediImgExp'
    process_dir =  dataset_dir+'/process'
    filelist = os.listdir(dataset_dir+'/image') 
    pic_id = []
    for item in filelist:
        pic_id.append(item.split('.')[0])
    blue_csv = pd.read_csv("table_blue.csv")
    brown_csv = pd.read_csv("table_brown.csv")

    for picId in pic_id:
        if os.path.exists( process_dir+'/'+picId + '_draw_all_cells.png'):
            continue
        condition1 = (blue_csv['ind']== picId)
        blue_thresh = blue_csv[ condition1 ].values[0,1:7]
        condition2 = (brown_csv['ind']== picId)
        brown_thresh = brown_csv[ condition2 ].values[0,1:7]
        print(picId)
        if blue_thresh[5] == 0 or int(brown_thresh[5]) == 0:
            continue
        processImg(picId,dataset_dir,blue_thresh ,brown_thresh)

if __name__ == '__main__':

    getAllCellsPics()
