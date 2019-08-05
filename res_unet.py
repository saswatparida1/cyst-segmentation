

!pip install medpy

import numpy as np
from keras.models import Model, load_model
from keras.layers import BatchNormalization,Convolution2D,SeparableConv2D,Conv2D,concatenate,add,Concatenate,Conv2DTranspose, Input, Add, UpSampling2D, Activation, merge, MaxPooling2D, Deconvolution2D, Reshape, Permute, Dropout
from keras.optimizers import SGD, Adam, Adadelta,RMSprop
#from scipy.misc import imresize, imsave, imread
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from medpy.metric import dc, precision, recall
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers

import numpy as np
from keras.models import Model, load_model
from keras.layers import BatchNormalization,Convolution2D, Input,ZeroPadding2D, UpSampling2D, Activation, merge, MaxPooling2D, Deconvolution2D, Reshape, Permute

from keras.optimizers import SGD, Adam
#from scipy.misc import imresize, imsave, imread
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from medpy.metric import dc, precision, recall
import matplotlib.pyplot as plt
from keras.regularizers import l2
from sklearn.metrics import log_loss
from keras.preprocessing.image import ImageDataGenerator
import sys
lrate1=3e-4  #changed lrate
lrate=float(lrate1)
print ('learning rate:',lrate)
#print('JBHI_w8_'+str(lrate1)+'.h5')
#print('JBHI_model_best'+lrate1)

import numpy as np
import warnings

from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
#from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

import re
from scipy import linalg
import scipy.ndimage as ndi
# from six.moves import range
import os
import threading
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def dice_coef(y_true, y_pred):
    
    #y_pred = K.round(y_pred)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return 1. -dice_coef(y_true, y_pred)



#TEST1_DATA

# 1w88Hj79XAbDr-nbQM5cWjLbqCi4sjRZW    --cutgrader1
# 1Y4XcQrdQ8qthROnMJ2-Krj7LUO0XWEAm    --cutgrader2
# 1DjNwEWftZtn_o0SUqRupGeR0eIk3S7qe    --cutgrader3


file7 = drive.CreateFile({'id': '1dhJjIXM_W4fm0U72h48Jfy2y3Eogl8ZJ'})
file7.GetContentFile('test1_cutimages_conall.npy')
test1_X = np.load('test1_cutimages_conall.npy')
print (test1_X.shape)

file8 = drive.CreateFile({'id': '1DjNwEWftZtn_o0SUqRupGeR0eIk3S7qe '})
file8.GetContentFile('test1_cutgrader3.npy')
test1_y = np.load('test1_cutgrader3.npy')
print (test1_y.shape)

#########

# test1_cutimages_conall     1dhJjIXM_W4fm0U72h48Jfy2y3Eogl8ZJ
# test1_cutgt1               1w88Hj79XAbDr-nbQM5cWjLbqCi4sjRZW
# test1_cutgt2               1Y4XcQrdQ8qthROnMJ2-Krj7LUO0XWEAm
# test1_cutgt3               1DjNwEWftZtn_o0SUqRupGeR0eIk3S7qe


#TEST1_DATA



file7 = drive.CreateFile({'id': '1dhJjIXM_W4fm0U72h48Jfy2y3Eogl8ZJ'})
file7.GetContentFile('test1_cutimages_conall.npy')
test1_X = np.load('test1_cutimages_conall.npy')
print (test1_X.shape)

#test1_cutgrader1
file8 = drive.CreateFile({'id': '1w88Hj79XAbDr-nbQM5cWjLbqCi4sjRZW'})
file8.GetContentFile('test1_cutgrader1.npy')
test1_y1= np.load('test1_cutgrader1.npy')
print (test1_y1.shape)


#test1_cutgrader2

file9 = drive.CreateFile({'id': '1Y4XcQrdQ8qthROnMJ2-Krj7LUO0XWEAm'})
file9.GetContentFile('test1_cutgrader2.npy')
test1_y2= np.load('test1_cutgrader2.npy')
print (test1_y2.shape)


#test1_cutgrader3


file10 = drive.CreateFile({'id': '1DjNwEWftZtn_o0SUqRupGeR0eIk3S7qe'})
file10.GetContentFile('test1_cutgrader3.npy')
test1_y3= np.load('test1_cutgrader3.npy')
print (test1_y3.shape)

#TEST2_DATA

#  1B_xojbfI3xpjWw_Wqpr89cIVkEp4aF2y  --cutgrader1
#  1dNvcUtc9G7TafH7ZQ0_lTkLANf1Va1be  --cutgrader2
#  1roXTQkkH46hWo6AU1UWJe4J3zhHSqfNr  --cutgrader3



file7 = drive.CreateFile({'id': '1vYOErUtk1aT7XnV9EcFXa1wI0b5twRE2'})
file7.GetContentFile('test2_cutimages_conall.npy')
test2_X = np.load('test2_cutimages_conall.npy')
print (test2_X.shape)

file8 = drive.CreateFile({'id': '1roXTQkkH46hWo6AU1UWJe4J3zhHSqfNr'})
file8.GetContentFile('test2_cutgrader1.npy')
test2_y = np.load('test2_cutgrader1.npy')
print (test2_y.shape)


##############

# test2_cutimage_conall   1vYOErUtk1aT7XnV9EcFXa1wI0b5twRE2
# test2_cutgt1            1B_xojbfI3xpjWw_Wqpr89cIVkEp4aF2y
# test2_cutgt2            1dNvcUtc9G7TafH7ZQ0_lTkLANf1Va1be
# test2_cutgt3            1roXTQkkH46hWo6AU1UWJe4J3zhHSqfNr



#TEST2_DATA



file1 = drive.CreateFile({'id': '1vYOErUtk1aT7XnV9EcFXa1wI0b5twRE2'})
file1.GetContentFile('test2_cutimages_conall.npy')
test2_X = np.load('test2_cutimages_conall.npy')
print (test2_X.shape)

#Test2_cutgrader1

file2 = drive.CreateFile({'id': '1B_xojbfI3xpjWw_Wqpr89cIVkEp4aF2y'})
file2.GetContentFile('test2_cutgrader1.npy')
test2_y1 = np.load('test2_cutgrader1.npy')
print (test2_y1.shape)

#test2_cutgrader2

file3 = drive.CreateFile({'id': '1dNvcUtc9G7TafH7ZQ0_lTkLANf1Va1be'})
file3.GetContentFile('test2_cutgrader2.npy')
test2_y2 = np.load('test2_cutgrader2.npy')
print (test2_y2.shape)


#test3_cutgrader3


file4 = drive.CreateFile({'id': '1roXTQkkH46hWo6AU1UWJe4J3zhHSqfNr'})
file4.GetContentFile('test2_cutgrader3.npy')
test2_y3 = np.load('test2_cutgrader3.npy')
print (test2_y3.shape)

train_X = np.reshape(train_X, (-1,256,512,1))
train_y = np.reshape(train_y*1, (-1,256,512,1))
test1_X = np.reshape(test1_X, (-1,256,512,1))
#test1_y = np.reshape(test1_y*1, (-1,256,512,1))


###########


test1_y1 = np.reshape(test1_y1*1, (-1,256,512,1))
test1_y2 = np.reshape(test1_y2*1, (-1,256,512,1))
test1_y3 = np.reshape(test1_y3*1, (-1,256,512,1))


test1_y1 = test1_y1.astype('float32')
test1_y2 = test1_y2.astype('float32')
test1_y3 = test1_y3.astype('float32')


###########




test2_X = np.reshape(test2_X, (-1,256,512,1))
test2_y = np.reshape(test2_y*1, (-1,256,512,1))


###########33


test2_y1 = np.reshape(test2_y1*1, (-1,256,512,1))
test2_y2 = np.reshape(test2_y2*1, (-1,256,512,1))
test2_y3 = np.reshape(test2_y3*1, (-1,256,512,1))


test2_y1 = test2_y1.astype('float32')
test2_y2 = test2_y2.astype('float32')
test2_y3 = test2_y3.astype('float32')


print(test2_X.shape)
print(test2_y1.shape)
print(test2_y2.shape)
print(test2_y3.shape)
############3

train_X = train_X.astype('float32')
train_y = train_y.astype('float32')
test1_X = test1_X.astype('float32')
#test1_y = test1_y.astype('float32')
test2_X = test2_X.astype('float32')
test2_y = test2_y.astype('float32')

print(train_X.shape)
print(train_y.shape)
#print(test1_X.shape)
#print(test1_y.shape)
print(test2_X.shape)
print(test2_y.shape)

mean = np.mean(train_X)
std = np.std(train_X)

print(mean)
print(std)

train_X -= mean
train_X /= std



# In[8]:

res1 = test1_X - mean
res1 = res1 / std

res2 = test2_X - mean
res2 = res2 / std

test1_X -= mean
test1_X /= std

test2_X -= mean
test2_X /= std

datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True, 
                             width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2,fill_mode='constant')

datagen.fit(train_X)
weight_decay = 0.0001

def conv_block(input_img, filters, stage, block,strides=(1,1), wt='he_normal'):
    
    
    
    """if stage==5:
      strides=(1,1) 
     """ 
    
    """if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:"""
    bn_axis = 1
    
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
   
    
    x = Conv2D(filters, (1, 1), padding='same',strides=strides,name=conv_name_base + '2a',init=wt)(input_img)
    x = BatchNormalization(name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)
    
    
    x = Conv2D(filters,(3, 3), padding='same',name=conv_name_base + '2b',init=wt)(x)
    x = BatchNormalization( name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)
    
    
    x = Conv2D(filters,(1,1), name=conv_name_base + '2c',init=wt)(x)
    x = BatchNormalization( name=bn_name_base + '2c')(x)
    
    
    shortcut = Conv2D(filters,(1, 1), strides=strides, 
                      name=conv_name_base + '1')(input_img)
   
    shortcut = BatchNormalization( name=bn_name_base + '1')(shortcut)

    x = layers.add([x,shortcut])
    x = Activation('relu')(x)
    
    return x

#model_l1

def ResNet50(learn = 3e-4, wt='he_normal',act='relu'):

    inputs = Input((256, 512,1)) 
    conv1 = Convolution2D(16, 3, 3, activation=act, border_mode='same',init=wt)(inputs) 
    conv1 = BatchNormalization()(conv1) 
    conv1 = Convolution2D(16, 3, 3, activation=act, border_mode='same',init=wt)(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = conv_block(conv1,16, stage=1, block='a1')
    conv1 = conv_block(conv1,16, stage=1, block='a2')
    
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1) 

    conv2 = Convolution2D(32, 3, 3, activation=act, border_mode='same',init=wt)(pool1)
    conv2 = BatchNormalization()(conv2) 
    conv2 = Convolution2D(32, 3, 3, activation=act, border_mode='same',init=wt)(conv2)    
    conv2 = BatchNormalization()(conv2) 
    conv2 = conv_block(conv2,32, stage=1, block='b1')
    conv2 = conv_block(conv2,32, stage=1, block='b2')
    #conv2 = conv_block(conv2,32, stage=1, block='b3')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2) 

    conv3 = Convolution2D(64, 3, 3, activation=act, border_mode='same',init=wt)(pool2)
    conv3 = BatchNormalization()(conv3) 
    conv3 = Convolution2D(64, 3, 3, activation=act, border_mode='same',init=wt)(conv3) 
    conv3 = BatchNormalization()(conv3) 
    conv3 = conv_block(conv3,64, stage=1, block='c1')
    conv3 = conv_block(conv3,64, stage=1, block='c2')
    #conv3 = conv_block(conv3,64, stage=1, block='c3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3) 

    conv4 = Convolution2D(128, 3, 3, activation=act, border_mode='same',init=wt)(pool3) 
    conv4 = BatchNormalization()(conv4) 
    conv4 = Convolution2D(128, 3, 3, activation=act, border_mode='same',init=wt)(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = conv_block(conv4,128, stage=1, block='d1')
    conv4 = conv_block(conv4,128, stage=1, block='d2')
    #conv4 = conv_block(conv4,128, stage=1, block='d3')
    #conv4 = Convolution2D(128, 3, 3, activation=act, border_mode='same',init=wt,dilation_rate=(8,8))(conv4)
    #conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4) 

    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',init='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5) 
    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',init='he_normal')(conv5) 
    conv5 = BatchNormalization()(conv5)
    conv5 = conv_block(conv5,256, stage=1, block='e1')
    conv5 = conv_block(conv5,256, stage=1, block='e2')
    # conv5 = conv_block(conv5,256, stage=1, block='e3')
   

    input7 = UpSampling2D(size=(2, 2))(conv5) 
    input7 = Convolution2D(128,2,2,border_mode='same')(input7) 
    
    up7 = add([input7, conv4]) 
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',init='he_normal')(up7) 
    conv7 = BatchNormalization()(conv7) 
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',init='he_normal')(conv7) 
    conv7 = BatchNormalization()(conv7) 
    conv7 = conv_block(conv7,128, stage=1, block='f1')
    conv7 = conv_block(conv7,128, stage=1, block='f2')
    #conv7 = conv_block(conv7,128, stage=1, block='f3')

    input8 = UpSampling2D(size=(2, 2))(conv7) 
    input8 = Convolution2D(64,2,2,border_mode='same')(input8) 
    #up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)

    up8 = add([input8, conv3]) 
    conv8 = Convolution2D(64, 3, 3, activation=act, border_mode='same',init=wt)(up8) 
    conv8 = BatchNormalization()(conv8) 
    conv8 = Convolution2D(64, 3, 3, activation=act, border_mode='same',init=wt)(conv8) 
    conv8 = BatchNormalization()(conv8)
    conv8 = conv_block(conv8,64, stage=1, block='g1')
    conv8 = conv_block(conv8,64, stage=1, block='g2')
    #conv8 = conv_block(conv8,64, stage=1, block='g3')

    input9 = UpSampling2D(size=(2, 2))(conv8) 
    input9 = Convolution2D(32,2,2,border_mode='same')(input9) 
    #up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv2], axis=3)

    up9 = add([input9, conv2]) 
    conv9 = Convolution2D(32, 3, 3, activation=act, border_mode='same',init=wt)(up9) 
    conv9 = BatchNormalization()(conv9) 
    conv9 = Convolution2D(32, 3, 3, activation=act, border_mode='same',init=wt)(conv9) 
    conv9 = BatchNormalization()(conv9) 
    conv9 = conv_block(conv9,32, stage=1, block='h1')
    conv9 = conv_block(conv9,32, stage=1, block='h2')
    #conv9 = conv_block(conv9,32, stage=1, block='h3')

    input10 = UpSampling2D(size=(2, 2))(conv9) 
    input10 = Convolution2D(16,2,2,border_mode='same')(input10) 
    #up10 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv9), conv1], axis=3)

    up10 = add([input10, conv1]) 
    conv10 = Convolution2D(16, 3, 3, activation=act, border_mode='same',init=wt)(up10)
    conv10 = BatchNormalization()(conv10) 
    conv10 = Convolution2D(16, 3, 3, activation=act, border_mode='same',init=wt)(conv10) 
    conv10 = BatchNormalization()(conv10) 
    conv10 = conv_block(conv10,16, stage=1, block='i1')
    conv10 = conv_block(conv10,16, stage=1, block='i2')
    #conv10 = conv_block(conv10,16, stage=1, block='i3')

    #conv11 = Dropout(0.5)(conv10)
    conv12 = Convolution2D(1, 1, 1,border_mode='same')(conv10) 

    conv13 = Activation('sigmoid')(conv12) 

    model = Model(input=inputs, output=conv13) 

    model.compile(Adam(lr=learn), loss='binary_crossentropy', metrics=[dice_coef,'accuracy']) 

    return model



K.clear_session()
model = ResNet50()

model.summary()

from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

mc = ModelCheckpoint('model_loss', monitor='val_loss', save_best_only=True)
#tb = TensorBoard(write_images=True)
cv = CSVLogger('/content/drive/My Drive/Colab/results/latest/smriti_resnet_test1_l3.csv',append=True)

from google.colab import drive
drive.mount('/content/drive')

os.listdir('/content/drive/My Drive/Colab/results/best/')

epochs =160
bestdc=0
total=0
for i in range(epochs):
    print(i)
    history= model.fit_generator(datagen.flow(train_X, train_y, batch_size=4), nb_epoch=1, samples_per_epoch=len(train_X), callbacks=[cv,mc])                           
    #history= model.fit(train_X, train_y, batch_size=4, nb_epoch=1, validation_data=(valid_X,valid_y), callbacks=[cv,mc])
    total=0
    for j in range(8):
        val_pred = model.predict(val_x[j], batch_size=4)
        val_result = np.zeros(val_pred.shape)
            #print test1_result.shape
        val_result[val_pred>0.5] = 1
            #np.save('patients/patient_' + 'str(j)' + '_' + 'str(i)' + '_.npy',val_result)
            #np.save('patients_sfu_dilaconv+dropout(0.2)/patient_' + str(j) + '_' + str(i) + '_.npy',val_result)
        dice = dc(val_result,val_gt[j])
        # print("Volume "+str(j)+": "+str(dice))
        total+= dice
    mean = total/8.0
    print("Average dc: "+str(mean))
    if mean>bestdc:
        bestdc=mean
        model.save_weights('/content/drive/My Drive/Colab/results/latest/model_resnet_test1_weights_l3.h5') 
    print("Best dc: "+str(bestdc))

model.load_weights('/content/drive/My Drive/Colab/results/best/final.h5')

#TEST1
dcoefarr=[]


  
val_pred = model.predict(test1_X, batch_size=4)
score = model.evaluate(test1_X, test1_y)
print(score)
val_result = np.zeros(val_pred.shape)                                                                                                                                                        
val_result[val_pred>0.5] = 1
dcoef = dc(val_result,test1_y)
prec  = precision(val_result,test1_y)
rec   = recall(val_result,test1_y)
dcoefarr.append(dcoef)
print ("Dice: ",dcoef)
print("Precision: ",prec)
print("recall: ",rec)

#TEST2
dcoefarr=[]


  
val_pred = model.predict(test2_X, batch_size=4)
score = model.evaluate(test2_X, test2_y)
print(score)
val_result = np.zeros(val_pred.shape)                                                                                                                                                        
val_result[val_pred>0.5] = 1
dcoef = dc(val_result,test2_y)
prec  = precision(val_result,test2_y)
rec   = recall(val_result,test2_y)
dcoefarr.append(dcoef)
print ("Dice: ",dcoef)
print("Precision: ",prec)
print("recall: ",rec)

### TEST1_PREDICTION

#cirrus 1 nd 2 ,nidek 3 nd 4,spectralis 5 nd 6,topcon 7 nd 8

def test1results(model):
    
    frames = [(0,128),(128,256),(256,261),(261,266),(266,315),(315,364),(364,492),(492,620)]
    test1resarray = []
    
    
    for item in frames:
        x = item[0]
        y = item[1]
        
        test1_pred = model.predict(res1[x:y], batch_size=4)
        test1_result = np.zeros(test1_pred.shape)
        #print test1_result.shape
        test1_result[test1_pred>0.5] = 1
        test1resarray.append(test1_result)
        
        a = test1_result
        b = test1_y1[x:y]
        c = test1_y2[x:y]
        d = test1_y3[x:y]
        
        print("")
        print(dc(a, b))
        print(dc(a, c))
        print(dc(a, d))

        print(recall(a, b))
        print(recall(a, c))
        print(recall(a, d))

        print(precision(a, b))
        print(precision(a, c))
        print(precision(a, d))
        
    '''  
    for i in xrange(test1_pred.shape[0]):
            imsave('test1_results/'+str(i+x)+'.jpg',test1_X[i+x,:,:,0])
            imsave('test1_results/'+str(i+x)+'g.jpg',test1_y3[i+x,:,:,0])
            imsave('test1_results/'+str(i+x)+'p.jpg',test1_result[i,:,:,0]*255)
    
    test1resarray = np.array(test1resarray)
    np.save('test1results')
    '''

test1results(model)
#cirrus 1 nd 2 ,nidek 3 nd 4,spectralis 5 nd 6,topcon 7 nd 8

##TEST2_PREDICTION

#cirrus1 nd 2 ,nidek 3 ,spectralis 4 nd 5,topcon 6 nd 7
def test2results(model):
    
    frames = [(0,128),(128,256),(256,261),(261,268),(268,275),(275,282),(282,289)]
    test2resarray = []
    
    print("\nTEST 2 RESULTS: ")
    for item in frames:
        x = item[0]
        y = item[1]
        
        test2_pred = model.predict(res2[x:y], batch_size=4)
        test2_result = np.zeros(test2_pred.shape)
        #print test1_result.shape
        test2_result[test2_pred>0.5] = 1
        test2resarray.append(test2_result)
        
        a = test2_result
        b = test2_y1[x:y]
        c = test2_y2[x:y]
        d = test2_y3[x:y]
        
        print("")
        print(dc(a, b))
        print(dc(a, c))
        print(dc(a, d))

        print(recall(a, b))
        print(recall(a, c))
        print(recall(a, d))

        print(precision(a, b))
        print(precision(a, c))
        print(precision(a, d))
        #for i in xrange(test2_pred.shape[0]):
        #    imsave('test2_results/'+str(i+x)+'.jpg',test2_X[i+x,:,:,0])
        #   imsave('test2_results/'+str(i+x)+'g.jpg',test2_y3[i+x,:,:,0])
        #  imsave('test2_results/'+str(i+x)+'p.jpg',test2_result[i,:,:,0]*255)
    
    #test2resarray = np.array(test2resarray)
    #np.save('test2results')

test2results(model)
#cirrus 1 nd 2 ,nidek 3 ,spectralis 4 nd 5,topcon 6 nd 7



for i in range(0,287):
    imsave('/content/drive/My Drive/Colab/images/'+str('i')+'.jpg',test1_X[i,:,:,0])
    imsave('/content/drive/My Drive/Colab/images/'+str('i1')+'.jpg',test1_y1[i,:,:,0])
    imsave('/content/drive/My Drive/Colab/images/'+str('i2')+'.jpg',test1_y2[i,:,:,0])
    imsave('/content/drive/My Drive/Colab/images/'+str('i3')+'.jpg',test1_y3[i,:,:,0])
    #imsave('/content/drive/My Drive/results/smriti_resnet_with_identity/G1/test1/'+str(i+1)+'a.jpg',test_y[i,:,:,0])
    #imsave('/content/drive/My Drive/Colab/results/G2/test2/'+str(i+1)+'b.jpg',val_result[i,:,:,0]*255)
