# IHC

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [How to Run](#how-to-run)
- [License](#license)

## Overview

This is a project which analyses both quantitative and spatial features of immune checkpoints. It demonstrates a stable and potent potential to predict prognosis in lung cancers. Moreover, we also proposed a new approach to analyzing conventional IHC via EfficientUnet and ResNet.

## System Requirements

### Hardware Requirements

This project requires a standard computer with at least one GPU to support the neural networks.

### Software Requirements

#### OS Requirements

This project is supported for Linux, and it has been tested on the following system:

+ Linux: Ubuntu 16.04 

#### Python Dependencies

The code is compatible with Python 3.7. The following dependencies are needed to run the training or testing tasks:

```python
numpy
pandas
glmnet-py
scikit-survival
efficientnet-pytorch
opencv-contrib-python
```
Additionally, the code requires PyTorch (>= 1.4.0).

## Installation Guide

You can get our basic codes from Github.

```python
git clone https://github.com/dddtqshmpmz/IHC.git
```

## How to Run

1. Install all the  dependencies.
	+ `pip install -r requirements.txt`
	
2. Segment the tumor regions and lymph regions using EfficientUnet.

   + `cd efficientUnet `
   + `python train.py` Train an EfficientUnet model for semantic segmentation. The model will be saved in `/models`  directory. It takes about 3 hours with 3* GTX 1080Ti.
   + `python predict.py` Produce all segmented masks for IHC images. There are some outputs in `/inputs/patients_TestDataset/result`  directory.

3. Pre-process and post-process the data.
   + `cd dataPre_Postprocess` 
   + `python cell_extraction.py`  Extract and classify the 4 types of cells. Save the images and  points in a local directory.
   + `python calculate_nearestPts.py` Calculate some quantitative and spatial features. You can see one demo image in `/distancePic/all_tumour_positive_lymph_dis`  directory, and some point coordinates in `/patients_dataset/data_csv` directory.
   + `python get_data.py` Organize and merge the data.

4. Use lasso-cox model to predict prognosis in lung cancers.
   + `cd lasso_cox` The outputs of `Step 3 ` are in `/csv_100`,`/figure_os_csv_100_005_resnet*` ,`/rfs_csv_100`,etc.
   + `python train_test_cox.py`  Train and test in the datasets using lasso-cox. Some outputs are in `/csv_100_auc`,`/NRI`,`/rfs_csv_100_auc`,etc. It takes about 5 minutes on a standard PC.

## License

This project is covered under the **Apache 2.0 License**.