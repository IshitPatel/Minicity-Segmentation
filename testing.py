import os
import cv2
#from keras.layers.core import *
#from keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D, Add
#from keras.layers import BatchNormalization
#from keras.models import Model,Sequential,load_model
#from keras.callbacks import ModelCheckpoint
#from keras.optimizers import Adadelta, RMSprop,SGD,Adam
#from keras import regularizers
#from keras import backend as K
import numpy as np
import scipy
import numpy.random as rng
#from sklearn.utils import shuffle
#from sklearn.model_selection import train_test_split
from skimage.transform import resize
#from skimage.io import imsave
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from tensorflow.keras.layers import concatenate
#%matplotlib inline
import pandas as pd
from keras.models import load_model
from PIL import Image
from HGUNet import HGUNet



name = os.listdir("leftImg8bit/test")
input_test_images = []

print("loading test images")
count = 0
for i in name :
    img_x = cv2.imread("leftImg8bit/test/"+i)
    print (i)
    #img_x = cv2.resize(img_x, (y_shape,x_shape)) 
    #img_x1 = cv2.cvtColor(img_x, cv2.COLOR_BGR2GRAY)
    #img_x = img_x[:,:,np.newaxis]
    #im = img_x[0:512,0:512,:]
    #input_images.append(im)
    #im = img_x[0:512,512:1024,:]
    #input_images.append(im)
    #im = img_x[0:512,1024:1536,:]
    #input_images.append(im)
    #im = img_x[0:512,1536:2048,:]
    #input_images.append(im)
    #im = img_x[512:1024,0:512,:]
    #input_images.append(im)
    #im = img_x[512:1024,512:1024,:]
    #input_images.append(im)
    #im = img_x[512:1024,1024:1536,:]
    #input_images.append(im)
    #im = img_x[512:1024,1536:2048,:]
    #input_images.append(im)
    #im = img_x[256:768,256:768,:]
    #input_images.append(im)
    #im = img_x[256:768,768:1280,:]
    #input_images.append(im)
    #im = img_x[256:768,1280:1792,:]
    input_test_images.append(img_x)
    count = count+1
    print(count)

X_test = np.asarray(input_test_images, np.float32)/255
out = []
image = []

#creating the model
input_size = (1024,2048,3)
test_model = HGUNet(input_size)

test_model.load_weights("weights/37.hdf5")

for i in range(0,int(count/4)):
    out.append(test_model.predict(X_test[4*i:4*i+4]))
    print("Batch completed")
    y_test = np.asarray(out)
    for j in range(0,count):
        im = np.argmax(y_test[0],axis=-1)
        im = im[:,:,np.newaxis]
        img = np.concatenate((im,im,im),axis=-1)
        image.append(img)

predicitons = np.asarray(image)

p = 0
for k in name:
    image_to_save = Image.fromarray(predictions[p],'RGB')
    image_to_save.save("predicfolder/"k+"_labelIDs.png")
    

    

    

