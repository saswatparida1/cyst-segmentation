
import numpy as np
from keras.models import Model, load_model
from keras.layers import BatchNormalization,Convolution2D, Input, UpSampling2D, Activation, merge, MaxPooling2D, Deconvolution2D, Reshape, Permute
from keras.optimizers import SGD, Adam
from scipy.misc import imresize, imsave, imread
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from medpy.metric import dc, precision, recall
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import sys
lrate=float(3e-4)
print ('learning rate:',lrate)
import numpy as np
from keras.models import Model, load_model, Sequential
#from keras.layers import BatchNormalization,Convolution2D,Convolution1D, Input, UpSampling2D, Activation, merge, MaxPooling2D, Deconvolution2D, Reshape, Permute, SeparableConvolution2D
from keras import layers
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import UpSampling2D,Conv2DTranspose
from medpy.metric import dc, precision, recall
from keras.optimizers import SGD, Adam, Adadelta
from scipy.misc import imresize, imsave, imread
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
#from medpy.metric import dc, precision, recall
import matplotlib.pyplot as plt
import matplotlib.cm

import tensorflow as tf

from keras.models import Model
from keras import layers
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Conv2D
from keras.layers import DepthwiseConv2D
from keras.layers import ZeroPadding2D
from keras.layers import GlobalAveragePooling2D
from keras.utils.layer_utils import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.activations import relu
from keras.applications.imagenet_utils import preprocess_input

from keras.layers import AtrousConvolution2D,Convolution2D


def dice_coef(y_true, y_pred):
    
    #y_pred = K.round(y_pred)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def deeplab(learn = 3e-4):
    
    inputs = Input((256,512,1))
    
        ########################### ENTRY FLOW #######################################
    
  
    ############ Block 1
    

    layer = Conv2D(32, (3, 3), strides=(2, 2),padding='same', use_bias=False, name='block1_conv1')(inputs)
    #print layer.shape
    
    layer = BatchNormalization(name='block1_conv1_bn')(layer)
    #keras.layers.LeakyReLU(alpha=0.3)(layer)
    layer = Activation('relu', name='block1_relu1')(layer)
    #layer = Dropout(0.5)(layer)
    #print layer.shape
    layer = Conv2D(64, (3, 3),padding='same', use_bias=False, name='block1_conv2')(layer)
       
    #print layer.shape
    layer = BatchNormalization(name='block1_bn2')(layer)
    layer = Activation('relu', name='block1_relu2')(layer)
    #layer = Dropout(0.5)(layer)
    residual = Conv2D(128, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='block1_res')(layer)
    residual = BatchNormalization(name='block1_res_bn')(residual)
    
    #print residual.shape
    
    ################ Block 2
    
    layer = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(layer)
    #print layer.shape
    layer = BatchNormalization(name='block2_bn1')(layer)
    layer = Activation('relu', name='block2_relu1')(layer)
    #layer = Dropout(0.5)(layer)
    layer = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(layer)
    #print layer.shape
    layer = BatchNormalization(name='block2_bn2')(layer)
    #print "before"
    #print layer.shape
    #layer = Dropout(0.5)(layer)
    layer = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(layer)
    #print layer
    layer = layers.add([layer, residual], name='block2_add')
    #print layer.shape
    residual = Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='block2_res')(layer)
    skip = residual
   # residual = BatchNormalization(name='block2_res_bn')(residual)
    
    ############### Block 3
    
    layer = Activation('relu', name='block3_relu1')(layer)
    layer = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(layer)
    
    layer = BatchNormalization(name = 'block3_bn1')(layer)
    layer = Activation('relu', name='block3_relu2')(layer)
    #layer = Dropout(0.5)(layer)
    layer = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(layer)
    
    layer = BatchNormalization(name='block3_bn2')(layer)
    #layer = Dropout(0.5)(layer)
    layer = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(layer)
    layer = layers.add([layer, residual], name='block3_add')
    
    residual = Conv2D(512, (1, 1), strides=(2,2),padding='same', use_bias=False, name='block3_res')(layer)
    residual = BatchNormalization(name='block3_res_bn')(residual)
    
    ############### Block 4
    
    layer = Activation('relu', name='block4_relu1')(layer)
    layer = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(layer)
    
    layer = BatchNormalization(name='block4_bn1')(layer)
    layer = Activation('relu',name='block4_relu2')(layer)
    #layer = Dropout(0.5)(layer)
    layer = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(layer)
    
    layer = BatchNormalization(name='block4_bn2')(layer)
    #layer = Dropout(0.5)(layer)
    layer = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(layer)
    layer = layers.add([layer, residual], name='block4_add')
    skip_1=[]
    #print (layer.shape)
    #################################### Middle Flow #################################
    
    ############## Block 5
    
    for i in range(12):
        prefix = 'block' + str(i + 115)

        residual = layer
        layer = Activation('relu',name=prefix+'relu1')(layer)
        layer = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(layer)
        
        layer = BatchNormalization(name=prefix+'bn1')(layer)
        layer = Activation('relu', name=prefix+'relu2')(layer)
        #layer = Dropout(0.5)(layer)
        layer = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(layer)
        
        layer = BatchNormalization(name=prefix+'bn2')(layer)
        layer = Activation('relu',name=prefix+'relu3')(layer)
        #layer = Dropout(0.5)(layer)
        layer = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(layer)
        
        layer = BatchNormalization(name=prefix+'bn3')(layer)
        #layer = Dropout(0.5)(layer)
        layer = layers.add([layer, residual] , name=prefix+'add')
	skip_1.append(layer)

############################################ Exit Flow ################################

   ########## $$$$$$ Block 13

    residual = Conv2D(512, (1, 1), strides=(2, 2), padding='same', use_bias=False, name='block12_res')(layer)
    residual = BatchNormalization(name='block12_res_bn')(residual)
    
    layer = Activation('relu',name='block13_relu1')(layer)
    layer = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(layer)
    layer = BatchNormalization(name='block13_bn1')(layer)
    layer = Activation('relu', name='block13_relu2')(layer)
    #layer = Dropout(0.5)(layer)
    layer = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(layer)
    
    layer = BatchNormalization(name='block13_bn2')(layer)
    #layer = Dropout(0.5)(layer)
    layer = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(layer)
    layer = layers.add([layer, residual], name='block13_add')
    
    
    layer = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(layer)
    
    layer = BatchNormalization(name='block14_bn1')(layer)
    layer = Activation('relu',name='block14_relu1')(layer)
    #layer = Dropout(0.5)(layer)
    layer = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(layer)
    
    layer = BatchNormalization(name='block14_bn2')(layer)
    layer = Activation('relu', name='block14_relu2')(layer)
    #layer = Dropout(0.5)(layer)
   
    #layer = GlobalAveragePooling2D(name='block15_avg_pool')(layer)


   #############################feature extract#######################
    #b41 = Lambda(lambda layer: K.expand_dims(layer, 1))(layer)
    #b41 = Lambda(lambda layer: K.expand_dims(layer, 1))(b41)
#################################  ASPP   ###########################################
    b41=layer
    layer=Convolution2D(48, 1, 1, activation='relu', name='layer_1')(layer)
    b1 = SeparableConv2D(1024, (3, 3), dilation_rate=(6,6) ,activation='relu', name='fc6_1',padding='same',use_bias=False)(b41)
    b1 = BatchNormalization()(b1)
    #b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_1')(b1)
    b1 = BatchNormalization()(b1)
    #b1 = Dropout(0.5)(b1)
    b1 = Convolution2D(48, 1, 1, activation='relu', name='fc8_voc12_1')(b1)
    # hole = 12
    b2 = SeparableConv2D(1024, (3, 3), dilation_rate=(12,12) ,activation='relu', name='fc7_1_2',padding='same',use_bias=False)(b41)
    b2 = BatchNormalization()(b2)
    #b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_2')(b2)
    b2 = BatchNormalization()(b2)
    #b2 = Dropout(0.5)(b2)
    b2 = Convolution2D(48, 1, 1, activation='relu', name='fc8_voc12_2')(b2)

    # hole = 18
    b3 = SeparableConv2D(1024, (3, 3), dilation_rate=(18,18), activation='relu', name='fc7_1_1',padding='same',use_bias=False)(b41)
    b3 = BatchNormalization()(b3)
    #b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_3')(b3)
    b3 = BatchNormalization()(b3)
    #b3 = Dropout(0.5)(b3)
    b3 = Convolution2D(48, 1, 1, activation='relu', name='fc8_voc12_3')(b3)

    # hole = 24
    b4 = SeparableConv2D(1024, (3, 3), dilation_rate=(24,24), activation='relu', name='fc6_1_2',padding='same',use_bias=False)(b41)
    b4 = BatchNormalization()(b4)
    #b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(1024, 1, 1, activation='relu', name='fc7_4')(b4)
    b4 = BatchNormalization()(b4)
    #b4 = Dropout(0.5)(b4)
    b4 = Convolution2D(48, 1, 1, activation='relu', name='fc8_voc12_4')(b4)

    s = Concatenate()([b1, b2, b3, b4,layer])
    
   # s = Conv2D(1, (1,1), activation='relu', name='last------')(s)
   # x = Lambda(lambda xx: tf.image.resize_bilinear(s, (256,512), align_corners=True))(s)
    
    
###########################################################################################

    ############################## Reverse (De-conv) Entry ##############################3

    layer = Conv2D(1024, (3,3), padding='same', use_bias=False, name='block-14_conv')(s)
    #layer = Dropout(0.2)(layer)
    #print (layer.shape)
    #layer = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block-14_sepconv1')(layer)
    layer = Activation('relu', name='block-14_relu1')(layer)

    
    
    layer = UpSampling2D(size=(2,2))(layer)
    residual = Conv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False)(layer)
    layer = Dropout(0.2)(layer)
    residual = BatchNormalization(name='block-14_re_bn')(residual)
    
    layer = Conv2D(512, (3, 3), padding='same', use_bias=False, name='block-13_sepconv2')(layer)
    #layer = Dropout(0.2)(layer)
    layer = Activation('relu', name='block-13_relu1')(layer)
    layer = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block-13_sepconv1')(layer)
    #layer = Dropout(0.2)(layer)
    layer = Activation('relu', name='block-13_relu2')(layer)
    layer = layers.add([layer, residual], name='block-13_add')
    
    ############## ############ Reverse Middle ###############################
    
    for i in range(12):
        prefix = 'block-' + str(1137 - i)
        
        residual = layer
        #layer = Activation('relu',name=block_prefix+'relu1')(layer)
        layer = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(layer)
        #layer = Dropout(0.2)(layer)
        layer = BatchNormalization(name=prefix+'bn1')(layer)
        layer = Activation('relu', name=prefix+'relu2')(layer)
        layer = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(layer)
        #layer = Dropout(0.2)(layer)
        layer = BatchNormalization(name=prefix+'bn2')(layer)
        layer = Activation('relu',name=prefix+'relu3')(layer)
        layer = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(layer)
        #layer = Dropout(0.2)(layer)
        layer = BatchNormalization(name=prefix+'bn3')(layer)
        layer = Activation('relu',name=prefix+'relu1')(layer)
        layer = layers.add([layer, residual,skip_1[i]] , name=prefix+'add')
     
    ################## Reverse Exit ##############################################
    
    ########################## Block-4 #######################
    
   
    
    layer = UpSampling2D(size=(2,2))(layer)
    residual = SeparableConv2D(512, (1, 1), strides=(1, 1), padding='same', use_bias=False)(layer)
    residual = BatchNormalization(name='block-4_res_bn')(residual)
    layer = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block-4_sepconv1')(layer)
    #layer = Dropout(0.2)(layer)
    layer = BatchNormalization(name='block-4_bn1')(layer)
    layer = Activation('relu', name='block-4_relu1')(layer)
    layer = SeparableConv2D(512, (3, 3), padding='same', use_bias=False, name='block-4_sepconv2')(layer)
    #layer = Dropout(0.2)(layer)
    layer = BatchNormalization(name='block-4_bn2')(layer)
    layer = Activation('relu', name='block-4_relu2')(layer)
    layer = layers.add([layer,residual], name='block-4_add')
    
    ######################### Block-3 ########################
    layer = UpSampling2D(size=(2,2))(layer)

    residual = Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=False)(layer)
    residual = BatchNormalization(name='block-3_res_bn')(residual)
    
    #layer = UpSampling2D(size=(2,2))(layer)
    layer = Conv2D(256, (3, 3), padding='same', use_bias=False, name='block-3_sepconv1')(layer)
    #layer = Dropout(0.2)(layer)
    layer = BatchNormalization(name='block-3_bn1')(layer)
    layer = Activation('relu',name='block-3_relu1')(layer)
    layer = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block-3_sepconv2')(layer)
    #layer = Dropout(0.2)(layer)
    layer = BatchNormalization(name='block-3_bn2')(layer)
    layer = Activation('relu', name='block-3_relu2')(layer)
    layer = layers.add([layer,residual], name='block-3_add')
    
    ########################## Block-2 #######################
    layer = UpSampling2D(size=(2,2))(layer)
    residual = Conv2D(128, (1, 1), strides=(1, 1), padding='same', use_bias=False)(layer)
    residual = BatchNormalization(name='block-2_bn')(residual)
    
    
    layer = Conv2D(128, (3, 3), padding='same', use_bias=False, name='block-2_sepconv1')(layer)
    #layer = Dropout(0.2)(layer)
    layer = BatchNormalization(name='block-2_bn1')(layer)
    layer = Activation('relu',name='block-2_relu1')(layer)
    layer = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block-2_sepconv2')(layer)
    #layer = Dropout(0.2)(layer)
    layer = BatchNormalization(name='block-2_bn2')(layer)
    layer = layers.add([layer,residual], name='block-2_add')
    
    ############################ Block-1 ########################
    
    layer = Activation('relu', name='block-1_relu1')(layer)
    #layer = Conv2D(64, (3, 3), use_bias=False, name='block-11_conv1')(layer)
    layer = Conv2D(64, (3, 3),padding='same', use_bias=False, name='block-1_conv1',dilation_rate=1)(layer)
    #layer = Dropout(0.2)(layer)
    layer = BatchNormalization(name='block-1_bn1')(layer)
    layer = Activation('relu', name='block-1_relu2')(layer)
    layer = UpSampling2D(size=(2,2))(layer)
    layer = Conv2D(32, (3, 3), padding='same', use_bias=False, name='block-1_conv2',dilation_rate=1)(layer)
    #layer = Dropout(0.2)(layer)
    layer = BatchNormalization(name='block-1_bn2')(layer)
    #layer = Dropout(0.1)(layer)
    layer = Conv2D(1, (1, 1) ,border_mode='same')(layer)
    
###############################################################################################
    out = Activation('sigmoid')(layer)

    model = Model(input=inputs, output=out)
    
    
    
    return model
    

train_X = np.load('train_cutimages_conall.npy')
train_y = np.load('train_cutground.npy')

print (train_X.shape)
#print np.unique(train_y)


# In[42]:

train_X = np.reshape(train_X, (-1,256,512,1))
train_y = np.reshape(train_y*1, (-1,256,512,1))

train_X = train_X.astype('float32')
train_y = train_y.astype('float32')

print (train_X.shape)
print (train_y.shape)


# In[43]:

print (np.max(train_X))





# In[45]:

mean = np.mean(train_X)
std = np.std(train_X)
print (mean, std)


# In[46]:

train_X = train_X - mean
train_X = train_X / std
print (train_X.shape)

# In[44]:

datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, horizontal_flip=True, 
                             width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, fill_mode='constant')

datagen.fit(train_X)
# In[47]:

test1_X = np.load('test1_cutimages_conall.npy')
test1_y1 = np.load('test1_cutgrader1.npy')
test1_y2 = np.load('test1_cutgrader2.npy')
test1_y3 = np.load('test1_cutgrader3.npy')

print (test1_y1.shape)
print (np.max(test1_X))


# In[48]:

test1_X = np.reshape(test1_X, (-1,256,512,1))
test1_y1 = np.reshape(test1_y1*1, (-1,256,512,1))
test1_y2 = np.reshape(test1_y2*1, (-1,256,512,1))
test1_y3 = np.reshape(test1_y3*1, (-1,256,512,1))

test1_X = test1_X.astype('float32')
test1_y1 = test1_y1.astype('float32')
test1_y2 = test1_y2.astype('float32')
test1_y3 = test1_y3.astype('float32')

print (test1_X.shape)
print (test1_y1.shape)
print (test1_y2.shape)
print (test1_y3.shape)


# In[49]:

res1 = test1_X - mean
res1 = res1 / std
print (res1.shape)


# In[50]:

#Validation Data Creation

val_x = []
val_gt = []
val_x.append(test1_X[0:128])
val_gt.append(test1_y3[0:128])

val_x.append(test1_X[128:256])
val_gt.append(test1_y3[128:256])

val_x.append(test1_X[256:261])
val_gt.append(test1_y3[256:261])

val_x.append(test1_X[261:266])
val_gt.append(test1_y3[261:266])

val_x.append(test1_X[266:315])
val_gt.append(test1_y3[266:315])

val_x.append(test1_X[315:364])
val_gt.append(test1_y3[315:364])

val_x.append(test1_X[364:492])
val_gt.append(test1_y3[364:492])

val_x.append(test1_X[492:620])
val_gt.append(test1_y3[492:620])

val_x = np.array(val_x)
val_gt = np.array(val_gt)



val_x = val_x - mean
val_x = val_x / std

print (val_x.shape)
print (val_gt.shape)


# In[ ]:




# In[51]:


K.clear_session()
model = deeplab()
model.summary()

model.compile(optimizer=Adam(lr=3e-4), loss='binary_crossentropy', metrics=[dice_coef,'accuracy'])

# In[52]:

from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger

mc = ModelCheckpoint('deeplab_time_bestvalloss3_exp5', monitor='val_loss', save_best_only=True)
#tb = TensorBoard(write_images=True)
cv = CSVLogger('deeplab_time_logs_exp5',append=True)


# In[53]:

bestdc = 0




# In[ ]:




# In[ ]:

epochs = 500

for i in xrange(epochs):
    print (i)
    history= model.fit_generator(datagen.flow(train_X, train_y, batch_size=4), nb_epoch=1, samples_per_epoch=len(train_X),callbacks=[cv])                           
    total=0
    for j in xrange(8):
        val_pred = model.predict(val_x[j], batch_size=4)
        val_result = np.zeros(val_pred.shape)
        #print test1_result.shape
        val_result[val_pred>0.5] = 1
        dice = dc(val_result,val_gt[j])
        print ("Volume "+str(j)+": "+str(dice))
        total += dice
        
    
    mean = total/8.0
    print ("Average dc: "+str(mean))
    if mean > bestdc:
        model.save_weights('deeplab_exp5.h5')
        bestdc = mean
        #test1results(model)
        #test2results(model)
        
    print ("Best dc: "+str(bestdc))
    print ("")


model.save('deeplab_exp5_best_time')


# In[32]:

#test1_pred = model.predict(res1, batch_size=8)
#test1_result = np.zeros(test1_pred.shape)
#print test1_result.shape
#test1_result[test1_pred>0.5] = 1


# In[33]:

#volumes = [(0,128),(128,256),(256,261),(261,266),(266,315),(315,364),(364,492),(492,620)]


# In[34]:
'''
diceg1=[]
diceg2=[]
diceg3=[]
for i in volumes:
    print ("Volume",i)
    d1 = dice(test1_result[i[0]:i[1]],test1_y1[i[0]:i[1]])
    d2 = dice(test1_result[i[0]:i[1]],test1_y2[i[0]:i[1]])
    d3 = dice(test1_result[i[0]:i[1]],test1_y3[i[0]:i[1]])
    
    print (d1)
    print (d2)
    print (d3)
    diceg1.append(d1)
    diceg2.append(d2)
    diceg3.append(d3)


# In[35]:

print (np.mean(diceg1))
print (np.mean(diceg2))
print (np.mean(diceg3))

'''
