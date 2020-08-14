import os
import cv2
#from tensorflow.keras.layers.core import *
#from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D, Add
#from tensorflow.keras.layers import BatchNormalization
#from tensorflow.keras.models import Model,Sequential,load_model
#from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.optimizers import Adadelta, RMSprop,SGD,Adam
#from tensorflow.keras import regularizers
#from tensorflow.keras import backend as K
import numpy as np
import scipy
import numpy.random as rng
#from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split
from skimage.transform import resize
#from skimage.io import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from tensorflow.keras.layers import concatenate
%matplotlib inline
import pandas as pd



#loading the training images
name = os.listdir("leftImg8bit/train")
name2 = os.listdir("gtFine/train")
input_images = []
output_images = []

print("loading training images")
count = 0
for i in name :
    print("input image")
    img_x = cv2.imread("leftImg8bit/train/"+i)
    print (i)
    #img_x = cv2.resize(img_x, (y_shape,x_shape)) 
    #img_x1 = cv2.cvtColor(img_x, cv2.COLOR_BGR2GRAY)
    #img_x = img_x[:,:,np.newaxis]
    im = img_x[0:512,0:512,:]
    input_images.append(im)
    im = img_x[0:512,512:1024,:]
    input_images.append(im)
    im = img_x[0:512,1024:1536,:]
    input_images.append(im)
    im = img_x[0:512,1536:2048,:]
    input_images.append(im)
    im = img_x[512:1024,0:512,:]
    input_images.append(im)
    im = img_x[512:1024,512:1024,:]
    input_images.append(im)
    im = img_x[512:1024,1024:1536,:]
    input_images.append(im)
    im = img_x[512:1024,1536:2048,:]
    input_images.append(im)
    im = img_x[256:768,256:768,:]
    input_images.append(im)
    im = img_x[256:768,768:1280,:]
    input_images.append(im)
    im = img_x[256:768,1280:1792,:]
    input_images.append(im)
    k='_'.join(i.strip().split('.')[0].split('_')[:-1])
    #(mn,mb)=k

    print("output image")
    temp_3d=[]
    tem_3d=[]
    #img_y = cv2.resize(img_y, (y_shape, x_shape))
    for x in range(1,34):
        img_y = cv2.imread("gtFine/train/"+k+"_gtfine_labelIds.png")
        img_y[img_y!=x]=0
        img_y[img_y==x]=255
        temp_3d.append(img_y[:,:,0])
        #for y in range(0,1024):
        #    for z in range (0,2048):
        #        if img_y[y][z][0]==x:
        #            tem_3d[y][z][x-1] = 255
        #        else:
        #            tem_3d[y][z][x-1] = 0
    tem_3d = np.swapaxes(temp_3d,2,0)
    tem_3d = np.swapaxes(tem_3d,0,1)
    tem_arr_3d = np.array(tem_3d)
    im = tem_arr_3d[0:512,0:512,:]
    output_images.append(im)
    im = tem_arr_3d[0:512,512:1024,:]
    output_images.append(im)
    im = tem_arr_3d[0:512,1024:1536,:]
    output_images.append(im)
    im = tem_arr_3d[0:512,1536:2048,:]
    output_images.append(im)
    im = tem_arr_3d[512:1024,0:512,:]
    output_images.append(im)
    im = tem_arr_3d[512:1024,512:1024,:]
    output_images.append(im)
    im = tem_arr_3d[512:1024,1024:1536,:]
    output_images.append(im)
    im = tem_arr_3d[512:1024,1536:2048,:]
    output_images.append(im)
    im = tem_arr_3d[256:768,256:768,:]
    output_images.append(im)
    im = tem_arr_3d[256:768,768:1280,:]
    output_images.append(im)
    im = tem_arr_3d[256:768,1280:1792,:]
    output_images.append(im)
    #if count == 1:
    #    break
    count+=1
    print (count)


#loading the validation images
name3 = os.listdir("leftImg8bit/val")
input_val_images = []
output_val_images = []

print("loading validation images")
count = 0
for j in name3 :
    print("input image")
    img_x = cv2.imread("leftImg8bit/val/"+j)
    print (j)
    #img_x = cv2.resize(img_x, (y_shape,x_shape)) 
    #img_x1 = cv2.cvtColor(img_x, cv2.COLOR_BGR2GRAY)
    #img_x = img_x[:,:,np.newaxis]
    im = img_x[0:512,0:512,:]
    input_val_images.append(im)
    im = img_x[0:512,512:1024,:]
    input_val_images.append(im)
    im = img_x[0:512,1024:1536,:]
    input_val_images.append(im)
    im = img_x[0:512,1536:2048,:]
    input_val_images.append(im)
    im = img_x[512:1024,0:512,:]
    input_val_images.append(im)
    im = img_x[512:1024,512:1024,:]
    input_val_images.append(im)
    im = img_x[512:1024,1024:1536,:]
    input_val_images.append(im)
    im = img_x[512:1024,1536:2048,:]
    input_val_images.append(im)
    im = img_x[256:768,256:768,:]
    input_val_images.append(im)
    im = img_x[256:768,768:1280,:]
    input_val_images.append(im)
    im = img_x[256:768,1280:1792,:]
    input_val_images.append(im)
    k='_'.join(j.strip().split('.')[0].split('_')[:-1])
    #(mn,mb)=k

    print("output image")
    temp_3d=[]
    tem_3d=[]
    #img_y = cv2.resize(img_y, (y_shape, x_shape))
    for x in range(1,34):
        img_y = cv2.imread("gtFine/val/"+k+"_gtfine_labelIds.png")
        img_y[img_y!=x]=0
        img_y[img_y==x]=255
        temp_3d.append(img_y[:,:,0])
        #for y in range(0,1024):
        #    for z in range (0,2048):
        #        if img_y[y][z][0]==x:
        #            tem_3d[y][z][x-1] = 255
        #        else:
        #            tem_3d[y][z][x-1] = 0
    tem_3d = np.swapaxes(temp_3d,2,0)
    tem_3d = np.swapaxes(tem_3d,0,1)
    tem_arr_3d = np.array(tem_3d)
    im = tem_arr_3d[0:512,0:512,:]
    output_val_images.append(im)
    im = tem_arr_3d[0:512,512:1024,:]
    output_val_images.append(im)
    im = tem_arr_3d[0:512,1024:1536,:]
    output_val_images.append(im)
    im = tem_arr_3d[0:512,1536:2048,:]
    output_val_images.append(im)
    im = tem_arr_3d[512:1024,0:512,:]
    output_val_images.append(im)
    im = tem_arr_3d[512:1024,512:1024,:]
    output_val_images.append(im)
    im = tem_arr_3d[512:1024,1024:1536,:]
    output_val_images.append(im)
    im = tem_arr_3d[512:1024,1536:2048,:]
    output_val_images.append(im)
    im = tem_arr_3d[256:768,256:768,:]
    output_val_images.append(im)
    im = tem_arr_3d[256:768,768:1280,:]
    output_val_images.append(im)
    im = tem_arr_3d[256:768,1280:1792,:]
    output_val_images.append(im)
    #if count == 1:
    #    break
    count+=1
    print (count)

