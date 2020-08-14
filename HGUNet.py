import os
import cv2
from keras.layers.core import *
from keras.layers import  Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D, Add
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential,load_model
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import backend as K
import numpy as np

K.set_image_data_format('channels_last')

def HGUNet(input_):

	input_img=Input(input_)
	###########################################  Encoder  ####################################################

	Econv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "block1_conv1")(input_img)
	Econv1_1 = BatchNormalization()(Econv1_1)
	Econv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "block1_conv2")(Econv1_1)
	Econv1_2 = BatchNormalization()(Econv1_2)
	pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "block1_pool1")(Econv1_2)
	
	Econv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv1")(pool1)
	Econv2_1 = BatchNormalization()(Econv2_1)
	Econv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv2")(Econv2_1)
	Econv2_2 = BatchNormalization()(Econv2_2)
	pool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(Econv2_2)

	Econv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv1")(pool2)
	Econv3_1 = BatchNormalization()(Econv3_1)
	Econv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv2")(Econv3_1)
	Econv3_2 = BatchNormalization()(Econv3_2)
	pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(Econv3_2)

	###########################################  HG  ####################################################


	conv_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_conv1")(pool3)

	conv1_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_block1_conv1")(conv_1)
	conv1_1 = BatchNormalization()(conv1_1)
	conv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_block1_conv2")(conv1_1)
	conv1_2 = BatchNormalization()(conv1_2)
	conv1_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_block1_conv3")(conv1_2)
	conv1_3 = BatchNormalization()(conv1_3)
	residual1 = Add(name = "hg_block1_add")([conv_1,conv1_3])

	pool1_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "hg_block1_pool1")(residual1) #56

	branch1_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_branch_block1_conv1")(residual1)
	branch1_1 = BatchNormalization()(branch1_1)
	branch1_2= Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_branch_block1_conv2")(branch1_1)
	branch1_2 = BatchNormalization()(branch1_2)
	branch1_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_branch_block1_conv3")(branch1_2)
	branch1_3 = BatchNormalization()(branch1_3)
	bresidual1 = Add(name = "hg_branch_block1_add")([residual1,branch1_3])

	conv2_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_block2_conv1")(pool1_1)
	conv2_1 = BatchNormalization()(conv2_1)
	conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_block2_conv2")(conv2_1)
	conv2_2 = BatchNormalization()(conv2_2)
	conv2_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_block2_conv3")(conv2_2)
	conv2_3 = BatchNormalization()(conv2_3)
	residual2 = Add( name = "hg_block2_add")([pool1_1,conv2_3])

	pool2_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "hg_block2_pool1")(residual2) #28

	branch2_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_branch_block2_conv1")(residual2)
	branch2_1 = BatchNormalization()(branch2_1)
	branch2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_branch_block2_conv2")(branch2_1)
	branch2_2 = BatchNormalization()(branch2_2)
	branch2_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_branch_block2_conv3")(branch2_2)
	branch2_3 = BatchNormalization()(branch2_3)
	bresidual2 = Add(name = "hg_branch_block2_add")([residual2,branch2_3])

	conv3_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_block3_conv1")(pool2_1)
	conv3_1 = BatchNormalization()(conv3_1)
	conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_block3_conv2")(conv3_1)
	conv3_2 = BatchNormalization()(conv3_2)
	conv3_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_block3_conv3")(conv3_2)
	conv3_3 = BatchNormalization()(conv3_3)
	residual3 = Add(name = "hg_block3_add")([pool2_1,conv3_3])

	pool3_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "hg_block3_pool1")(residual3) #14

	branch3_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_branch_block3_conv1")(residual3)
	branch3_1 = BatchNormalization()(branch3_1)
	branch3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_branch_block3_conv2")(branch3_1)
	branch3_2 = BatchNormalization()(branch3_2)
	branch3_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_branch_block3_conv3")(branch3_2)
	branch3_3 = BatchNormalization()(branch3_3)
	bresidual3 = Add(name = "hg_branch_block3_add")([residual3,branch3_3])
	
	conv4_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_block4_conv1")(pool3_1)
	conv4_1 = BatchNormalization()(conv4_1)
	conv4_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_block4_conv2")(conv4_1)
	conv4_2 = BatchNormalization()(conv4_2)
	conv4_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_block4_conv3")(conv4_2)
	conv4_3 = BatchNormalization()(conv4_3)
	residual4 = Add(name = "hg_block4_add")([pool3_1,conv4_3])
	
	conv5_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_block5_conv1")(residual4)
	conv5_1 = BatchNormalization()(conv5_1)
	conv5_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_block5_conv2")(conv5_1)
	conv5_2 = BatchNormalization()(conv5_2)
	conv5_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_block5_conv3")(conv5_2)
	conv5_3 = BatchNormalization()(conv5_3)
	residual5 = Add(name = "hg_block5_add")([residual4,conv5_3])
	
	up1_1 = Conv2DTranspose(256,(2,2),strides = (2,2), activation = 'relu', padding = 'same',name = "hg_up1")(residual5)
	up1_1 = BatchNormalization()(up1_1) #28
	add1 = Add(name = "hg_up1_add")([up1_1,bresidual3])

	uconv1_1 = Conv2D(128, (1, 1), activation='relu', padding='same',name = "hg_upconv1_1")(add1)
	uconv1_1 = BatchNormalization()(uconv1_1)
	uconv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same',name = "hg_upconv1_2")(uconv1_1)
	uconv1_2 = BatchNormalization()(uconv1_2)
	uconv1_3 = Conv2D(256, (1, 1), activation='relu', padding='same',name = "hg_upconv1_3")(uconv1_2)
	uconv1_3 = BatchNormalization()(uconv1_3)
	uresidual1 = Add(name = "hg_upblock1_add")([add1,uconv1_3])

	up2_1 = Conv2DTranspose(256,(2,2),strides = (2,2), activation = 'relu', padding = 'same',name = "hg_up2")(uresidual1)
	up2_1 = BatchNormalization()(up2_1) #56
	add2 = Add()([up2_1,bresidual2])

	uconv2_1 = Conv2D(128, (1, 1), activation='relu', padding='same',name = "hg_upconv2_1")(add2)
	uconv2_1 = BatchNormalization()(uconv2_1)
	uconv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same',name = "hg_upconv2_2")(uconv2_1)
	uconv2_2 = BatchNormalization()(uconv2_2)
	uconv2_3 = Conv2D(256, (1, 1), activation='relu', padding='same',name = "hg_upconv2_3")(uconv2_2)
	uconv2_3 = BatchNormalization()(uconv2_3)
	uresidual2 = Add(name = "hg_upblock2")([add2,uconv2_3])

	up3_1 = Conv2DTranspose(256,(2,2),strides = (2,2), activation = 'relu', padding = 'same',name = "hg_up3")(uresidual2)
	up3_1 = BatchNormalization()(up3_1) #112
	add3 = Add()([up3_1,bresidual1])

	uconv3_1 = Conv2D(128, (1, 1), activation='relu', padding='same',name = "hg_upconv3_1")(add3)
	uconv3_1 = BatchNormalization()(uconv3_1)
	uconv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same',name = "hg_upconv3_2")(uconv3_1)
	uconv3_2 = BatchNormalization()(uconv3_2)
	uconv3_3 = Conv2D(256, (1, 1), activation='relu', padding='same',name = "hg_upconv3_3")(uconv3_2)
	uconv3_3 = BatchNormalization()(uconv3_3)
	uresidual3 = Add()([add3,uconv3_3])

	##########################################  Decoder   ##################################################

	up1 = Conv2DTranspose(128,(3,3),strides = (2,2), activation ='relu', padding = 'same', name = "upsample_1")(uresidual3)
	up1 = BatchNormalization()(up1)
	up1 = merge([up1, Econv3_2], mode='concat', concat_axis=3, name = "merge_1")
	Upconv1_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_1")(up1)
	Upconv1_1 = BatchNormalization()(Upconv1_1)
	Upconv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_2")(Upconv1_1)
	Upconv1_2 = BatchNormalization()(Upconv1_2)

	up2 = Conv2DTranspose(64,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_2")(Upconv1_2)
	up2 = BatchNormalization()(up2)
	up2 = merge([up2, Econv2_2], mode='concat', concat_axis=3, name = "merge_2")
	Upconv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_1")(up2)
	Upconv2_1 = BatchNormalization()(Upconv2_1)
	Upconv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_2")(Upconv2_1)
	Upconv2_2 = BatchNormalization()(Upconv2_2)
	
	up3 = Conv2DTranspose(16,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_3")(Upconv2_2)
	up3 = BatchNormalization()(up3)
	up3 = merge([up3, Econv1_2], mode='concat', concat_axis=3, name = "merge_3")
	Upconv3_1 = Conv2D(32, (3, 3), activation='relu', padding='same', name = "Upconv3_1")(up3)
	Upconv3_1 = BatchNormalization()(Upconv3_1)
	Upconv3_2 = Conv2D(32, (3, 3), activation='relu', padding='same', name = "Upconv3_2")(Upconv3_1)
	Upconv3_2 = BatchNormalization()(Upconv3_2)
	   
	decoded = Conv2D(33, (3, 3), activation='sigmoid', padding='same', name = "Ouput_layer")(Upconv3_2)

	convnet = Model(input_img, decoded)

	return convnet
'''
#Sample example
t=HGUNet((512,512,3))
t.summary()
'''