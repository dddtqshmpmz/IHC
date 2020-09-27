import argparse
import os
from glob import glob
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import segmentation_models.segmentation_models_pytorch as smp
import torch.nn as nn
import archs
from dataset import Dataset, TestDataset
from metrics import iou_score , dice_coef
from utils import AverageMeter
from PIL import Image
from PIL import ImageFilter
from math import floor
import random
import numpy as np
import shutil

def sliding_window(image, stepSize,windowSize,height,width,savepath):
    ''' Cut out patches using sliding windows
    Parameters
    ----------
    image: ndarray
        image read using opencv
    stepSize: int
        step size in sliding windows
    height,width: int
        the height & width of the image
    savepath: str
        the directoty saving patches
    '''
    count=0
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            if (y+windowSize[1])<=height and (x+windowSize[0])<=width:
                slide=image[y:y+windowSize[1], x:x+windowSize[0],:]
                cv2.imwrite( savepath+'/'+str(count)+'.png', slide)
                count=count+1
            else:
                continue

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='patient_datasetRandom_NestedUNet_woDS',
                        help='model name')
    args = parser.parse_args()
    return args

def dice_simple(output, target):
    smooth = 1e-5
    output_ = output > 125
    target_ = target > 125
    intersection = (output_ & target_).sum()
    return (2. * intersection + smooth) / (output_.sum() + target_.sum() + smooth)

def predict_img( predict_img='78-MHC-2', cut_size=144 ):
    ''' Give the predict_img's predicted mask image based on window_size = 144
    Parameters
    ----------
    predict_img: str
        the name of the original image
    cut_size: int
        step size in sliding windows
    '''
    args = parse_args(
    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)
    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = smp.Unet('efficientnet-b3', in_channels=config['input_channels'], classes=config['num_classes'], encoder_weights='imagenet').cuda()
    model = model.cuda()
    model= nn.DataParallel(model,device_ids=[0,1,2])
    stepSize=int(0.5*cut_size)
    windowSize=[cut_size,cut_size]

    path='inputs/image'
    mask_path ='inputs/mask'
    savepath='inputs/patients_TestDataset/image'
    save_maskpath='inputs/patients_TestDataset/mask/0'
    output_path = os.path.join('outputs', config['name'], str(0))
    result_path = 'inputs/patients_TestDataset/result'
    
    if os.path.exists(savepath):
        shutil.rmtree(savepath)  
        os.makedirs(savepath)
    if os.path.exists(save_maskpath):
        shutil.rmtree(save_maskpath)  
        os.makedirs(save_maskpath)
    if os.path.exists(output_path):
        shutil.rmtree(output_path)  
        os.makedirs(output_path)
    
    image_name= path+'/'+ predict_img+'.png'
    mask_name = mask_path+'/'+ predict_img+'.png'
    image= cv2.imread(image_name)
    mask= cv2.imread(mask_name)
    height,width=image.shape[:2]
    size1 = (int(floor(width/stepSize)*stepSize), int(floor(height/stepSize)*stepSize)) 
    print(size1)
    sliding_window(image,stepSize,windowSize,height,width,savepath)
    sliding_window(mask,stepSize,windowSize,height,width,save_maskpath)

    # Data loading code
    img_ids = glob(os.path.join('inputs','patients_TestDataset', 'image', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    test_img_ids =img_ids
    model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))
    model.eval()

    test_transform = Compose([
        transforms.Resize(config['input_h'], config['input_w']),
        transforms.Normalize() ])

    test_dataset = Dataset(
        img_ids=test_img_ids,
        img_dir=os.path.join('inputs', 'patients_TestDataset', 'image'),
        mask_dir=os.path.join('inputs', 'patients_TestDataset', 'mask'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'],str(c)), exist_ok=True)
    with torch.no_grad():
        for input, target, meta in tqdm(test_loader, total=len(test_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                output = model(input)[-1]
            else:
                output = model(input)

            dice = dice_coef(output, target)
            avg_meter.update(dice, input.size(0))

            output = torch.sigmoid(output)
            zero = torch.zeros_like(output) 
            one = torch.ones_like(output)
            output = torch.where(output > 0.5, one , zero) 
            output = output.data.cpu().numpy()
            
            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
                                (output[i, c] * 255).astype('uint8') )

    print('Dice: %.4f' % avg_meter.avg)
    torch.cuda.empty_cache()

    image= cv2.imread(image_name)
    (height ,width, _)= image.shape
    width_num= floor(width*2/cut_size-1) 
    height_num= floor(height*2/cut_size-1)
    
    img_ids = glob( os.path.join('outputs', config['name'], str(0), '*.png') )
    num_patch= len(img_ids) #get the total count of patches
    assert(num_patch == (width_num*height_num))
    
    Canvas =np.zeros(( height , width ),dtype=np.uint8)
    step_size = int(cut_size/2)
    for i in range( 0 ,height_num ):
        for j in range(0,width_num):
            patchFirst = cv2.imread( os.path.join( 'outputs', config['name'], str(0), str( i * width_num + j )+'.png'),0 )
            patchFirst = cv2.resize(patchFirst, (cut_size,cut_size))
            patchFirst = (patchFirst>0).astype(np.uint8)
            if j == 0 :
                Canvas[ i* step_size : int(i*step_size + cut_size) , j* step_size : int(j*step_size + cut_size) ] = patchFirst
            
            if ( j+1 ) < width_num: #blend the results in the same row
                patchNext = cv2.imread( os.path.join( 'outputs', config['name'], str(0), str( i * width_num + j + 1 )+'.png'),0 )
                patchNext = cv2.resize(patchNext, (cut_size,cut_size))

                patchNext = (patchNext>0).astype(np.uint8)
                Canvas[ i* step_size : int(i*step_size + cut_size) , int (j* step_size + cut_size/2) :  int(j*step_size + cut_size) ] = \
                    np.bitwise_and( Canvas[ i* step_size : int(i*step_size + cut_size) , int (j* step_size + cut_size/2) :  int(j*step_size + cut_size) ] ,\
                         patchNext[ : , 0: int(cut_size/2) ] )
                Canvas[ i* step_size : int(i*step_size + cut_size) , int(j*step_size + cut_size) : int(j*step_size + 3*cut_size/2 )  ] =\
                     patchNext[ : , int(cut_size/2) : int(cut_size) ] 
            
    cv2.imwrite('inputs/patients_TestDataset/result/' + str(cut_size) +'.png', (Canvas*255).astype(np.uint8))

def mergeMultiSizeImg(img_name='78-MHC-2'):
    ''' Merge the predicted results of multi-size patches
    Parameters
    ----------
    predict_img: str
        the name of the original image
    '''
    cut_size_list = [100,144,300,500]
    for cut_size in cut_size_list:
        predict_img( predict_img=img_name,cut_size=cut_size )
    
    oriImg = cv2.imread('inputs/image/'+img_name+'.png')
    height, width, channel = oriImg.shape
    MultiSizeResult = np.zeros((height,width)).astype(np.float)
    result_dir = 'inputs/patients_TestDataset/result'
    for cut_size in cut_size_list:
        img = cv2.imread( os.path.join(result_dir, str(cut_size)+'.png'  ), 0 )
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float)
        MultiSizeResult = MultiSizeResult + img
    
    MultiSizeResult /= len(cut_size_list)
    MultiSizeResult = MultiSizeResult.astype(np.uint8)
    MultiSizeResult_label= (MultiSizeResult>125)
    MultiSizeResult[MultiSizeResult_label] = 255
    MultiSizeResult[~MultiSizeResult_label] = 0

    cv2.imwrite( result_dir + '/'+img_name+'-merged.png', MultiSizeResult ) 
    ori_mask = cv2.imread( 'inputs/mask/'+ img_name + '.png',0 )
    dice_merge = dice_simple( MultiSizeResult, ori_mask)
    print('dice_merge = ', dice_merge)

def predict_allMasks():
    ''' 
    Give all images the predicted masks
    '''
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model = smp.Unet('efficientnet-b3', in_channels=config['input_channels'], classes=config['num_classes'], encoder_weights='imagenet').cuda()

    model = model.cuda()
    model= nn.DataParallel(model,device_ids=[0,1,2])

    path='inputs/image'  
    savepath='inputs/patients_TestDataset/image'
    output_path = os.path.join('outputs', config['name'], str(0))

    image_dir = 'inputs/image'
    result_dir = 'inputs/patients_TestDataset/result/mask'
    img_name_list = os.listdir(image_dir)

    for img in img_name_list:
        predict_img = img.split('.')[0]
        cut_size_list = [100,144,300,500]
        if os.path.exists( os.path.join(result_dir, predict_img + '.png' ) ):
            continue

        for cut_size in cut_size_list:
            stepSize=int(0.5*cut_size)
            windowSize=[cut_size,cut_size]

            if os.path.exists(savepath):
                shutil.rmtree(savepath)  
                os.makedirs(savepath)
            if os.path.exists(output_path):
                shutil.rmtree(output_path)  
                os.makedirs(output_path)

            image_name= path+'/'+ predict_img+'.jpg'
            image= cv2.imread(image_name)

            height,width=image.shape[:2]
            sliding_window(image,stepSize,windowSize,height,width,savepath)

            img_ids = glob(os.path.join('inputs','patients_TestDataset', 'image', '*' + config['img_ext']))
            img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

            test_img_ids =img_ids

            model.load_state_dict(torch.load('models/%s/model.pth' % config['name'])) 
            model.eval()

            test_transform = Compose([
                transforms.Resize(config['input_h'], config['input_w']),
                transforms.Normalize() ])

            test_dataset = TestDataset( # Dataset
                img_ids=test_img_ids,
                img_dir=os.path.join('inputs', 'patients_TestDataset', 'image'),
                img_ext=config['img_ext'],
                num_classes=config['num_classes'],
                transform=test_transform) 

            test_loader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=config['batch_size'],
                shuffle=False,
                num_workers=config['num_workers'], 
                drop_last=False)
            
            for c in range(config['num_classes']):
                os.makedirs(os.path.join('outputs', config['name'],str(c)), exist_ok=True)
            with torch.no_grad():
                for input, meta in tqdm(test_loader, total=len(test_loader)):
                    input = input.cuda()

                    # compute output
                    if config['deep_supervision']:
                        output = model(input)[-1]
                    else:
                        output = model(input)

                    output = torch.sigmoid(output)  
                    zero = torch.zeros_like(output) 
                    one = torch.ones_like(output)
                    output = torch.where(output > 0.5, one , zero) 
                    output = output.data.cpu().numpy()
                    
                    for i in range(len(output)):
                        for c in range(config['num_classes']):
                            cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
                                        (output[i, c] * 255).astype('uint8') )

            torch.cuda.empty_cache()

            image= cv2.imread(image_name)
            (height ,width, _)= image.shape
            width_num= floor(width*2/cut_size-1) ]
            height_num= floor(height*2/cut_size-1)
            
            img_ids = glob( os.path.join('outputs', config['name'], str(0), '*.png') )
            num_patch= len(img_ids) 
            assert(num_patch == (width_num*height_num))
            
            Canvas =np.zeros(( height , width ),dtype=np.uint8)
            step_size = int(cut_size/2)
            for i in range( 0 ,height_num ):
                for j in range(0,width_num):
                    patchFirst = cv2.imread( os.path.join( 'outputs', config['name'], str(0), str( i * width_num + j )+'.png'),0 )
                    patchFirst = cv2.resize(patchFirst, (cut_size,cut_size))
                    patchFirst = (patchFirst>0).astype(np.uint8)
                    if j == 0 :
                        Canvas[ i* step_size : int(i*step_size + cut_size) , j* step_size : int(j*step_size + cut_size) ] = patchFirst
                    
                    if ( j+1 ) < width_num: 
                        patchNext = cv2.imread( os.path.join( 'outputs', config['name'], str(0), str( i * width_num + j + 1 )+'.png'),0 )
                        patchNext = cv2.resize(patchNext, (cut_size,cut_size))

                        patchNext = (patchNext>0).astype(np.uint8)
                        Canvas[ i* step_size : int(i*step_size + cut_size) , int (j* step_size + cut_size/2) :  int(j*step_size + cut_size) ] = \
                            np.bitwise_and( Canvas[ i* step_size : int(i*step_size + cut_size) , int (j* step_size + cut_size/2) :  int(j*step_size + cut_size) ] ,\
                                patchNext[ : , 0: int(cut_size/2) ] )
                        Canvas[ i* step_size : int(i*step_size + cut_size) , int(j*step_size + cut_size) : int(j*step_size + 3*cut_size/2 )  ] =\
                            patchNext[ : , int(cut_size/2) : int(cut_size) ] 
                    
            cv2.imwrite('inputs/patients_TestDataset/result/' + str(cut_size) +'.png', (Canvas*255).astype(np.uint8))

        oriImg = cv2.imread('inputs/image/'+predict_img+'.jpg')
        height, width, channel = oriImg.shape
        MultiSizeResult = np.zeros((height,width)).astype(np.float)
        result_dir = 'inputs/patients_TestDataset/result'
        for cut_size in cut_size_list:
            img = cv2.imread( os.path.join(result_dir, str(cut_size)+'.png'  ), 0 )
            img = cv2.resize(img, (width, height))
            img = img.astype(np.float)
            MultiSizeResult = MultiSizeResult + img
        
        MultiSizeResult /= len(cut_size_list)
        MultiSizeResult = MultiSizeResult.astype(np.uint8)
        MultiSizeResult_label= (MultiSizeResult>125)
        MultiSizeResult[MultiSizeResult_label] = 255
        MultiSizeResult[~MultiSizeResult_label] = 0

        cv2.imwrite( result_dir + '/mask/' + predict_img+'.png', MultiSizeResult )

def produce_randomCropDataset(output_size=[100,144,300,500], cropPicName=['0-2D-1','0-2D-2'],cropNumEach=3000 ):
    ''' Produce random crop dataset (crop size = output_size)
    Parameters
    ----------
    output_size: list(int)
        the cropped images' size
    cropPicName: list(str)
        the names of the cropped images
    cropNumEach: int
        the num of cropped patches for each whole image
    '''
    cropPicDir = 'inputs/image'
    cropMastDir= 'inputs/mask'
    savePicDir= 'inputs/patient_datasetRandom/images'
    saveMaskDir='inputs/patient_datasetRandom/masks/0'
    if not os.path.exists(savePicDir):
        os.makedirs(savePicDir)
    if not os.path.exists(saveMaskDir):
        os.makedirs(saveMaskDir)
    
    imglist=[cv2.imread(cropPicDir+'/'+p+'.png') for p in cropPicName]
    masklist=[cv2.imread(cropMastDir+'/'+p+'.png') for p in cropPicName]

    for ind in range(len(imglist)):
        (h, w, c ) = imglist[ind].shape
        print('ind:',ind)
        for i in range(cropNumEach):
            pos1, pos2 = random.uniform(0,1), random.uniform(0,1)
            outputInd = random.randint(0,3)
            w1 = int(pos1 * (w - output_size[outputInd]))
            sw = output_size[outputInd]
            h1 = int(pos2 * (h - output_size[outputInd]))
            sh = output_size[outputInd]

            mask_gray = masklist[ind][h1:h1+sh, w1:w1+sw]
            cv2.imwrite(saveMaskDir+'/'+cropPicName[ind]+'-'+str(i)+'.png',(mask_gray).astype('uint8') )
            image = imglist[ind][ h1:h1+sh, w1:w1+sw, :]
            cv2.imwrite(savePicDir+'/'+cropPicName[ind]+'-'+str(i)+'.png', image )


if __name__ == '__main__':

    # produce the random-crop dataset
    cropPicName = ['0-2D-1','0-2D-2','0-3D-1','2-GAL9-2','42-PD1-2','44-PDL1-1','67-TIM3-2']
    produce_randomCropDataset(output_size=[100,144,300,500], cropPicName= cropPicName,cropNumEach=3000)

    # use the efficientUnet model to predict masks for all images
    predict_allMasks()

    