

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
