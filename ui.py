
import numpy as np
from keras.models import Model, load_model
from keras.layers import BatchNormalization,Convolution2D,SeparableConv2D,Conv2D,concatenate,add,Concatenate,Conv2DTranspose, Input, Add, UpSampling2D, Activation, merge, MaxPooling2D, Deconvolution2D, Reshape, Permute, Dropout
from keras.optimizers import SGD, Adam, Adadelta,RMSprop
from scipy.misc import imresize, imsave, imread
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler,ReduceLROnPlateau
from medpy.metric import dc, precision, recall
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
import matplotlib

import numpy as np
from keras.models import Model, load_model
from keras.layers import BatchNormalization,Convolution2D, Input,ZeroPadding2D, UpSampling2D, Activation, merge, MaxPooling2D, Deconvolution2D, Reshape, Permute

from keras.optimizers import SGD, Adam
from scipy.misc import imresize, imsave, imread
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
import warnings
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


    x = Conv2D(filters,(5, 5), padding='same',name=conv_name_base + '2d',init=wt)(x)
    x = BatchNormalization( name=bn_name_base + '2d')(x)
    x = Activation('relu')(x)


    x = Conv2D(filters,(3, 3), padding='same',name=conv_name_base + '2e',init=wt)(x)
    x = BatchNormalization( name=bn_name_base + '2e')(x)
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
    conv1_1 = BatchNormalization()(conv1)
    conv1 = conv_block(conv1_1,16, stage=1, block='a1')
    conv1 = conv_block(conv1,16, stage=1, block='a2')

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(32, 3, 3, activation=act, border_mode='same',init=wt)(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Convolution2D(32, 3, 3, activation=act, border_mode='same',init=wt)(conv2)
    conv2_2 = BatchNormalization()(conv2)
    conv2 = conv_block(conv2_2,32, stage=1, block='b1')
    conv2 = conv_block(conv2,32, stage=1, block='b2')
    #conv2 = conv_block(conv2,32, stage=1, block='b3')
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(64, 3, 3, activation=act, border_mode='same',init=wt)(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Convolution2D(64, 3, 3, activation=act, border_mode='same',init=wt)(conv3)
    conv3_3 = BatchNormalization()(conv3)
    conv3 = conv_block(conv3_3,64, stage=1, block='c1')
    conv3 = conv_block(conv3,64, stage=1, block='c2')
    #conv3 = conv_block(conv3,64, stage=1, block='c3')
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(128, 3, 3, activation=act, border_mode='same',init=wt)(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Convolution2D(128, 3, 3, activation=act, border_mode='same',init=wt)(conv4)
    conv4_4 = BatchNormalization()(conv4)
    conv4 = conv_block(conv4_4,128, stage=1, block='d1')
    conv4 = conv_block(conv4,128, stage=1, block='d2')
    #conv4 = conv_block(conv4,128, stage=1, block='d3')
    #conv4 = Convolution2D(128, 3, 3, activation=act, border_mode='same',init=wt,dilation_rate=(8,8))(conv4)
    #conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',init='he_normal')(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Convolution2D(256, 3, 3, activation='relu', border_mode='same',init='he_normal')(conv5)
    conv5_5 = BatchNormalization()(conv5)
    conv5 = conv_block(conv5_5,256, stage=1, block='e1')
    conv5 = conv_block(conv5,256, stage=1, block='e2')
    # conv5 = conv_block(conv5,256, stage=1, block='e3')


    input7 = UpSampling2D(size=(2, 2))(conv5)
    input7 = Convolution2D(128,2,2,border_mode='same')(input7)

    up7 = add([input7, conv4,conv4_4])
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',init='he_normal')(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same',init='he_normal')(conv7)
    conv7_7 = BatchNormalization()(conv7)
    conv7 = conv_block(conv7_7,128, stage=1, block='f1')
    conv7 = conv_block(conv7,128, stage=1, block='f2')
    #conv7 = conv_block(conv7,128, stage=1, block='f3')

    input8 = UpSampling2D(size=(2, 2))(conv7)
    input8 = Convolution2D(64,2,2,border_mode='same')(input8)
    #up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv4), conv3], axis=3)

    up8 = add([input8, conv3,conv3_3])
    conv8 = Convolution2D(64, 3, 3, activation=act, border_mode='same',init=wt)(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Convolution2D(64, 3, 3, activation=act, border_mode='same',init=wt)(conv8)
    conv8_8 = BatchNormalization()(conv8)
    conv8 = conv_block(conv8_8,64, stage=1, block='g1')
    conv8 = conv_block(conv8,64, stage=1, block='g2')
    #conv8 = conv_block(conv8,64, stage=1, block='g3')

    input9 = UpSampling2D(size=(2, 2))(conv8)
    input9 = Convolution2D(32,2,2,border_mode='same')(input9)
    #up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv2], axis=3)

    up9 = add([input9, conv2,conv2_2])
    conv9 = Convolution2D(32, 3, 3, activation=act, border_mode='same',init=wt)(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Convolution2D(32, 3, 3, activation=act, border_mode='same',init=wt)(conv9)
    conv9_9 = BatchNormalization()(conv9)
    conv9 = conv_block(conv9_9,32, stage=1, block='h1')
    conv9 = conv_block(conv9,32, stage=1, block='h2')
    #conv9 = conv_block(conv9,32, stage=1, block='h3')

    input10 = UpSampling2D(size=(2, 2))(conv9)
    input10 = Convolution2D(16,2,2,border_mode='same')(input10)
    #up10 = concatenate([Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv9), conv1], axis=3)

    up10 = add([input10, conv1,conv1_1])
    conv10 = Convolution2D(16, 3, 3, activation=act, border_mode='same',init=wt)(up10)
    conv10 = BatchNormalization()(conv10)
    conv10 = Convolution2D(16, 3, 3, activation=act, border_mode='same',init=wt)(conv10)
    conv10_10 = BatchNormalization()(conv10)
    conv10 = conv_block(conv10_10,16, stage=1, block='i1')
    conv10 = conv_block(conv10,16, stage=1, block='i2')
    #conv10 = conv_block(conv10,16, stage=1, block='i3')

    #conv11 = Dropout(0.5)(conv10)
    conv12 = Convolution2D(1, 1, 1,border_mode='same')(conv10)

    conv13 = Activation('sigmoid')(conv12)

    model = Model(input=inputs, output=conv13)

    model.compile(Adam(lr=learn), loss='binary_crossentropy', metrics=[dice_coef,'accuracy'])

    return model

def load():
    
    test1_X = np.load('sas.npy')
    test1_X = np.reshape(test1_X, (-1,256,512,1))
    test1_X = test1_X.astype('float32')
    
#    test1_X=(test1_X-np.min(test1_X))/(np.max(test1_X)-np.min(test1_X))
    mean=np.mean(test1_X)
    std=np.std(test1_X)
    test1_X=(test1_X-mean)/std
    model = ResNet50()
    model.load_weights('/home/saswat/PycharmProjects/saswat/l2.h5')
    val_pred = model.predict(test1_X)
    val_pred = (val_pred>0.5)*1

    val_pred=np.reshape(val_pred,(256,512))

    print(val_pred.shape)
    matplotlib.image.imsave('/home/saswat/PycharmProjects/saswat/name1.png',val_pred)
    np.save("va.npy",val_pred)


if __name__ == '__main__':
    load()


