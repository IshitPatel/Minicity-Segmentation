import os
import cv2
#from tensorflow.keras.layers.core import *
from tensorflow.keras.layers import Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D, Add
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model,Sequential,load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adadelta, RMSprop,SGD,Adam
from tensorflow.keras import regularizers
from tensorflow.keras import backend as K
import numpy as np
import scipy
import numpy.random as rng
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from skimage.transform import resize
#from skimage.io import imsave
import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
from tensorflow.keras.layers import concatenate
#%matplotlib inline
import pandas as pd

#importing the training and validation variables
from load_images import input_images,output_images,input_val_images,output_val_images
#importing model file
import HGUNet

print("imported the libraries")
K.set_image_data_format('channels_last')

def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def weighted_cxe(y_true,y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    yt=np.asarray(y_true_f)
    w=sum(yt>0)
    t=len(yt)
    b=t-w
    d=(t/w) + (t/b)
    c1=((t/w)/d)*(1-y_pred_f )
    c2=(t/b)/d*(y_pred_f)
    loss=K.sum(c1*K.log(1-y_pred_f ) + c2*K.log(y_pred_f))
    return loss

x_shape = 512
y_shape = 512
channels = 3
input_img = Input(shape = (x_shape, y_shape,channels))

model= HGUNet(input_image)
model.summary()
model.compile(optimizer = Adam(0.0005), loss="binary_crossentropy", metrics = ["accuracy"])

X_train = input_images
y_train = output_images

X_val = input_val_images
y_val = output_val_images

X_train = np.asarray(X_train, np.float32)/255
X_val = np.asarray(X_val, np.float32)/255
y_train = np.asarray(y_train, np.float32)/255
y_val = np.asarray(y_val, np.float32)/255
saveModel = "UNet_Hg_dice_2018_split.h5"

#numEpochs = 100
batch_size = 8
num_batches = int(len(X_train)/batch_size)
print ("Number of batches: %d\n" % num_batches)
saveDir = '/'
loss=[]
val_loss=[]
acc=[]
val_acc=[]
epoch=0;
best_loss=1000
r_c=0

#model.load_weights('UNet.h5', by_name = True)

while epoch <100 :
    
    history = model.fit(X_train, y_train, batch_size=batch_size, epochs=1, validation_data=(X_val,y_val), shuffle=True, verbose=1) 
#    if best_loss>=float(history.history['val_loss'][0]) :
#        
#        best_loss=float(history.history['val_loss'][0])
#        model.save_weights(saveModel)
#        
    epoch=epoch+1
#    print ("EPOCH NO. : "+str(epoch)+"\n")
#    loss.append(float(history.history['loss'][0]))
#    val_loss.append(float(history.history['val_loss'][0]))
#    acc.append(float(history.history['acc'][0]))
#    val_acc.append(float(history.history['val_acc'][0]))
#    loss_arr=np.asarray(loss)
#    
#    loss1=np.asarray(loss)
#    acc1=np.asarray(acc)
#    val_acc1=np.asarray(val_acc)
#    print ('loss=',loss1,'Acc = ',acc1,'Validation Loss = ',val_loss1,'Validation Accuracy = ',val_acc1)
    
print("training Done.")
