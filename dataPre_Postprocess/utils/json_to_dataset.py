import argparse
import base64
import json
import os
import os.path as osp
import imgviz
import PIL.Image
from labelme.logger import logger
from labelme import utils
import cv2
import numpy as np

def json2mask( json_file, jsonDir, imgDir,maskDir ):
    '''Convert the json files to mask images
    Parameters
    ----------
    json_file: str
        the name of json file
    jsonDir: str
        the name of json file's directory
    imgDir: str
        the name of image file's directory
    maskDir: str
        the name of mask file's directory
    '''

    jsonID = json_file.split('.')[0]
    data = json.load(open(json_Dir+'/'+json_file))
    imageData = data.get("imageData")
    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = utils.img_b64_to_arr(imageData)

    label_name_to_value = {"_background_": 0}
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(
        img.shape, data["shapes"], label_name_to_value
    )

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    lbl_viz = imgviz.label2rgb(
        label=lbl, img=imgviz.asgray(img), label_names=label_names, loc="rb"
    )

    PIL.Image.fromarray(img).save(osp.join(imgDir, jsonID + ".png"))
    utils.lblsave(osp.join(maskDir, jsonID + ".png"), lbl)

def masks_to_graypic( path ):
    '''Convert the mask image to uint8
    Parameters
    ----------
    path: str
        the name of mask image
    '''
    filelist = os.listdir(path) # list the names of images
    for item in filelist:
        image_name= path+'/'+item
        image= cv2.imread(image_name)
        mask_image = (image[:,:,2]>0)
        mask= np.zeros( (image.shape[0],image.shape[1]) )
        mask[mask_image]=1
        cv2.imwrite(path+'/'+item ,(mask * 255).astype('uint8') )

if __name__ == "__main__":
    json_Dir = 'patients_dataset/json'
    imgDir = 'patients_dataset/image'
    maskDir = 'patients_dataset/mask'
    filelist = os.listdir(json_Dir)
    for file_ in filelist:
        json2mask( file_, json_Dir, imgDir,maskDir)
    
    masks_to_graypic(maskDir)