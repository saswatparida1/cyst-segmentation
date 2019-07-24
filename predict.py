
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



import re
from scipy import linalg
import scipy.ndimage as ndi
import os
import threading
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def random_channel_shift(x, intensity, channel_index=0):
    x = np.rollaxis(x, channel_index, 0)
    min_x, max_x = np.min(x), np.max(x)
    channel_images = [np.clip(x_channel + np.random.uniform(-intensity, intensity), min_x, max_x)
                      for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform_matrix, channel_index=0, fill_mode='nearest', cval=0.):
    x = np.rollaxis(x, channel_index, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(x_channel, final_affine_matrix,
                      final_offset, order=0, mode=fill_mode, cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_index+1)
    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def array_to_img(x, dim_ordering='default', scale=True):
    from PIL import Image
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x /= np.max(x)
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[2])


def img_to_array(img, dim_ordering='default'):
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x


class ImageDataGenerator(object):
    '''Generate minibatches with
    real-time data augmentation.
    Assume X is train img, Y is train label (same size as X with only 0 and 255 for values)
    # Arguments
        featurewise_center: set input mean to 0 over the dataset. Only to X
        samplewise_center: set each sample mean to 0. Only to X
        featurewise_std_normalization: divide inputs by std of the dataset. Only to X
        samplewise_std_normalization: divide each input by its std. Only to X
        zca_whitening: apply ZCA whitening. Only to X
        rotation_range: degrees (0 to 180). To X and Y
        width_shift_range: fraction of total width. To X and Y
        height_shift_range: fraction of total height. To X and Y
        shear_range: shear intensity (shear angle in radians). To X and Y
        zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
            in the range [1-z, 1+z]. A sequence of two can be passed instead
            to select this range. To X and Y
        channel_shift_range: shift range for each channels. Only to X
        fill_mode: points outside the boundaries are filled according to the
            given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
            is 'nearest'. For Y, always fill with constant 0
        cval: value used for points outside the boundaries when fill_mode is
            'constant'. Default is 0.
        horizontal_flip: whether to randomly flip images horizontally. To X and Y
        vertical_flip: whether to randomly flip images vertically. To X and Y
        rescale: rescaling factor. If None or 0, no rescaling is applied,
            otherwise we multiply the data by the value provided (before applying
            any other transformation). Only to X
        dim_ordering: 'th' or 'tf'. In 'th' mode, the channels dimension
            (the depth) is at index 1, in 'tf' mode it is at index 3.
            It defaults to the `image_dim_ordering` value found in your
            Keras config file at `~/.keras/keras.json`.
            If you never set it, then it will be "th".
    '''
    def __init__(self,
                 featurewise_center=False,
                 samplewise_center=False,
                 featurewise_std_normalization=False,
                 samplewise_std_normalization=False,
                 zca_whitening=False,
                 rotation_range=0.,
                 width_shift_range=0.,
                 height_shift_range=0.,
                 shear_range=0.,
                 zoom_range=0.,
                 channel_shift_range=0.,
                 fill_mode='nearest',
                 cval=0.,
                 horizontal_flip=False,
                 vertical_flip=False,
                 rescale=None,
                 
                 dim_ordering='default'):
        
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.__dict__.update(locals())
        self.mean = None
        self.std = None
        self.principal_components = None
        self.rescale = rescale

        if dim_ordering not in {'tf', 'th'}:
            raise Exception('dim_ordering should be "tf" (channel after row and '
                            'column) or "th" (channel before row and column). '
                            'Received arg: ', dim_ordering)
        self.dim_ordering = dim_ordering
        if dim_ordering == 'th':
            self.channel_index = 1
            self.row_index = 2
            self.col_index = 3
        if dim_ordering == 'tf':
            self.channel_index = 3
            self.row_index = 1
            self.col_index = 2

        if np.isscalar(zoom_range):
            self.zoom_range = [1 - zoom_range, 1 + zoom_range]
        elif len(zoom_range) == 2:
            self.zoom_range = [zoom_range[0], zoom_range[1]]
        else:
            raise Exception('zoom_range should be a float or '
                            'a tuple or list of two floats. '
                            'Received arg: ', zoom_range)

    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)

    def standardize(self, x):
        # Only applied to X
        if self.rescale:
            x *= self.rescale
        # x is a single image, so it doesn't have image number at index 0
        img_channel_index = self.channel_index - 1
        if self.samplewise_center:
            x -= np.mean(x, axis=img_channel_index, keepdims=True)
        if self.samplewise_std_normalization:
            x /= (np.std(x, axis=img_channel_index, keepdims=True) + 1e-7)

        if self.featurewise_center:
            x -= self.mean
        if self.featurewise_std_normalization:
            x /= (self.std + 1e-7)

        if self.zca_whitening:
            flatx = np.reshape(x, (x.size))
            whitex = np.dot(flatx, self.principal_components)
            x = np.reshape(whitex, (x.shape[0], x.shape[1], x.shape[2]))

        return x

    def random_transform(self, x, y):
        # Need to modify to transform both X and Y ---- to do
        # x is a single image, so it doesn't have image number at index 0
        img_row_index = self.row_index - 1
        img_col_index = self.col_index - 1
        img_channel_index = self.channel_index - 1

        # use composition of homographies to generate final transform that needs to be applied
        if self.rotation_range:
            theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
        else:
            theta = 0
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta), np.cos(theta), 0],
                                    [0, 0, 1]])
        if self.height_shift_range:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range) * x.shape[img_row_index]
        else:
            tx = 0

        if self.width_shift_range:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range) * x.shape[img_col_index]
        else:
            ty = 0

        translation_matrix = np.array([[1, 0, tx],
                                       [0, 1, ty],
                                       [0, 0, 1]])
        if self.shear_range:
            shear = np.random.uniform(-self.shear_range, self.shear_range)
        else:
            shear = 0
        shear_matrix = np.array([[1, -np.sin(shear), 0],
                                 [0, np.cos(shear), 0],
                                 [0, 0, 1]])

        if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
            zx, zy = 1, 1
        else:
            zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
        zoom_matrix = np.array([[zx, 0, 0],
                                [0, zy, 0],
                                [0, 0, 1]])

        transform_matrix = np.dot(np.dot(np.dot(rotation_matrix, translation_matrix), shear_matrix), zoom_matrix)

        h, w = x.shape[img_row_index], x.shape[img_col_index]
        transform_matrix = transform_matrix_offset_center(transform_matrix, h, w)
        x = apply_transform(x, transform_matrix, img_channel_index,
                            fill_mode=self.fill_mode, cval=self.cval)
        # For y, mask data, fill mode constant, cval = 0
        y = apply_transform(y, transform_matrix, img_channel_index,
                            fill_mode="constant", cval=0)

        if self.channel_shift_range != 0:
            x = random_channel_shift(x, self.channel_shift_range, img_channel_index)

        if self.horizontal_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_col_index)
                y = flip_axis(y, img_col_index)

        if self.vertical_flip:
            if np.random.random() < 0.5:
                x = flip_axis(x, img_row_index)
                y = flip_axis(y, img_row_index)

        # TODO:
        # channel-wise normalization
        # barrel/fisheye
        return x, y

    def fit(self, X,
            augment=False,
            rounds=1,
            seed=None):
        '''Required for featurewise_center, featurewise_std_normalization
        and zca_whitening.
        # Arguments
            X: Numpy array, the data to fit on.
            augment: whether to fit on randomly augmented samples
            rounds: if `augment`,
                how many augmentation passes to do over the data
            seed: random seed.
        # Only applied to X
        '''
        X = np.copy(X)
        if augment:
            aX = np.zeros(tuple([rounds * X.shape[0]] + list(X.shape)[1:]))
            for r in range(rounds):
                for i in range(X.shape[0]):
                    aX[i + r * X.shape[0]] = self.random_transform(X[i])
            X = aX

        if self.featurewise_center:
            self.mean = np.mean(X, axis=0)
            X -= self.mean

        if self.featurewise_std_normalization:
            self.std = np.std(X, axis=0)
            X /= (self.std + 1e-7)

        if self.zca_whitening:
            flatX = np.reshape(X, (X.shape[0], X.shape[1] * X.shape[2] * X.shape[3]))
            sigma = np.dot(flatX.T, flatX) / flatX.shape[1]
            U, S, V = linalg.svd(sigma)
            self.principal_components = np.dot(np.dot(U, np.diag(1. / np.sqrt(S + 10e-7))), U.T)


class Iterator(object):

    def __init__(self, N, batch_size, shuffle, seed):
        self.N = N
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_generator = self._flow_index(N, batch_size, shuffle, seed)

    def reset(self):
        self.batch_index = 0

    def _flow_index(self, N, batch_size=32, shuffle=False, seed=None):
        # ensure self.batch_index is 0
        self.reset()
        while 1:
            if self.batch_index == 0:
                index_array = np.arange(N)
                if shuffle:
                    if seed is not None:
                        np.random.seed(seed + self.total_batches_seen)
                    index_array = np.random.permutation(N)

            current_index = (self.batch_index * batch_size) % N
            if N >= current_index + batch_size:
                current_batch_size = batch_size
                self.batch_index += 1
            else:
                current_batch_size = N - current_index
                self.batch_index = 0
            self.total_batches_seen += 1
            yield (index_array[current_index: current_index + current_batch_size],
                   current_index, current_batch_size)

    def __iter__(self):
        # needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        # ?
        return self.next(*args, **kwargs)


class NumpyArrayIterator(Iterator):

    def __init__(self, X, y, image_data_generator,
                 batch_size=32, shuffle=False, seed=None,
                 dim_ordering='default',
                 save_to_dir=None, save_prefix='', save_format='jpeg'):
        if len(X) != len(y):
            raise Exception('X (images tensor) and y (labels) '
                            'should have the same length. '
                            'Found: X.shape = %s, y.shape = %s' % (np.asarray(X).shape, np.asarray(y).shape))
        if dim_ordering == 'default':
            dim_ordering = K.image_dim_ordering()
        self.X = X
        self.y = y
        self.image_data_generator = image_data_generator
        self.dim_ordering = dim_ordering
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format
        super(NumpyArrayIterator, self).__init__(X.shape[0], batch_size, shuffle, seed)

    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        batch_y = np.zeros(tuple([current_batch_size] + list(self.y.shape)[1:]))
        for i, j in enumerate(index_array):
            x = self.X[j]
            label = self.y[j]
            x, label = self.image_data_generator.random_transform(x.astype('float32'), label.astype("float32"))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
            batch_y[i] = label
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
                mask = array_to_img(batch_y[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}_mask.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                mask.save(os.path.join(self.save_to_dir, fname))
        return batch_x, batch_y




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
    


train_X = np.load('../train_cutimages_conall.npy')
train_y = np.load('../train_cutground.npy')

print (train_X.shape)
#print np.unique(train_y)


train_X = np.reshape(train_X, (-1,256,512,1))
train_y = np.reshape(train_y*1, (-1,256,512,1))

train_X = train_X.astype('float64')
train_y = train_y.astype('float64')

print (train_X.shape)
print (train_y.shape)



print (np.max(train_X))





mean = np.mean(train_X)
std = np.std(train_X)
print (mean, std)


train_X = train_X - mean
train_X = train_X / std
print (train_X.shape)


test1_X = np.load('../test1_cutimages_conall.npy')
test1_y1 = np.load('../test1_cutgrader1.npy')
test1_y2 = np.load('../test1_cutgrader2.npy')
test1_y3 = np.load('../test1_cutgrader3.npy')


test2_X = np.load('../test2_cutimages_conall.npy')
test2_y1 = np.load('../test2_cutgrader1.npy')
test2_y2 = np.load('../test2_cutgrader2.npy')
test2_y3 = np.load('../test2_cutgrader3.npy')

print (np.max(test1_X))


test1_X =np.reshape(test1_X, (-1,256,512,1))
test1_y1 = np.reshape(test1_y1*1, (-1,256,512,1))
test1_y2 = np.reshape(test1_y2*1, (-1,256,512,1))
test1_y3 = np.reshape(test1_y3*1, (-1,256,512,1))

test1_X = test1_X.astype('float64')
test1_y1 = test1_y1.astype('float64')
test1_y2 = test1_y2.astype('float64')
test1_y3 = test1_y3.astype('float64')


test2_X =np.reshape(test2_X, (-1,256,512,1))
test2_y1 = np.reshape(test2_y1*1, (-1,256,512,1))
test2_y2 = np.reshape(test2_y2*1, (-1,256,512,1))
test2_y3 = np.reshape(test2_y3*1, (-1,256,512,1))


test2_X = test2_X.astype('float64')
test2_y1 = test2_y1.astype('float64')
test2_y2 = test2_y2.astype('float64')
test2_y3 = test2_y3.astype('float64')


print (test1_X.shape)
print (test1_y1.shape)
print (test1_y2.shape)
print (test1_y3.shape)



res1 = test1_X - mean
res1 = res1 / std
print (res1.shape)

res2 = test2_X - mean
res2 = res2 / std
print (res2.shape)



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



from medpy.metric import dc, precision, recall


K.clear_session()
model = deeplab()
model.summary()

model.load_weights('deeplab_exp5.h5')

model.compile(optimizer=Adam(lr=3e-4,amsgrad=True), loss='binary_crossentropy', metrics=[dice_coef,'accuracy'])




precisionavg1=[]
recavg1=[]
dcoefarr1=[]


precisionavg2=[]
recavg2=[]
dcoefarr2=[]


precisionavg3=[]
recavg3=[]
dcoefarr3=[]


print
print("TEST___1---------------------------------------- ------------------------")

############################################################test1

cir=np.concatenate((res1[0:255],res2[0:255]),axis=0)
cir_l=np.concatenate((test1_y1[0:255],test2_y1[0:255]),axis=0)

print
print("cirrus ------------------------")
val_pred1 = model.predict(cir, batch_size=4)

val_pred1=np.reshape(val_pred1*1,(-1,256,512))
val_pred=np.zeros(val_pred1.shape)
val_pred[val_pred1>0.5] = 1
cir=np.reshape(cir*1,(-1,256,512))
from scipy.misc import imsave
for i in range(cir.shape[0]):
    imsave("cirrus/{}_image.png".format(i),cir[i])
    imsave("cirrus/{}_label.png".format(i),val_pred[i])

############################

nid=np.concatenate((res1[256:265],res2[256:260]),axis=0)
nid_l=np.concatenate((test1_y1[256:265],test2_y1[256:260]),axis=0)



print
print(" nidek------------------------- ")
  
val_pred1 = model.predict(nid, batch_size=4)

val_pred1=np.reshape(val_pred1*1,(-1,256,512))
val_pred=np.zeros(val_pred1.shape)
val_pred[val_pred1>0.5] = 1

nid=np.reshape(nid*1,(-1,256,512))
from scipy.misc import imsave
for i in range(nid.shape[0]):
    imsave("nidek/{}_image.png".format(i),nid[i])
    imsave("nidek/{}_label.png".format(i),val_pred[i])
################################################
spe=np.concatenate((res1[266:363],res2[261:274]),axis=0)
spe_l=np.concatenate((test1_y1[266:363],test2_y1[261:274]),axis=0)


print
print(" spectralis------------------ ")
  
val_pred1 = model.predict(spe, batch_size=4)

val_pred1=np.reshape(val_pred1*1,(-1,256,512))
val_pred=np.zeros(val_pred1.shape)
val_pred[val_pred1>0.5] = 1

spe=np.reshape(spe*1,(-1,256,512))
from scipy.misc import imsave
for i in range(spe.shape[0]):
    imsave("spectralis/{}_image.png".format(i),spe[i])
    imsave("spectralis/{}_label.png".format(i),val_pred[i])

#####################################################

top=np.concatenate((res1[364:619],res2[275:281]),axis=0)
top_l=np.concatenate((test1_y1[364:619],test2_y1[275:281]),axis=0)


print
print(" topcon------------------")
  
val_pred1 = model.predict(top, batch_size=4)

val_pred1=np.reshape(val_pred1*1,(-1,256,512))
val_pred=np.zeros(val_pred1.shape)
val_pred[val_pred1>0.5] = 1
top=np.reshape(top*1,(-1,256,512))
from scipy.misc import imsave
for i in range(top.shape[0]):
    imsave("topcon/{}_image.png".format(i),top[i])
    imsave("topcon/{}_label.png".format(i),val_pred[i])

###################################################
