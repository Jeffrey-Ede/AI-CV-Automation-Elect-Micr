from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse

import numpy as np
import tensorflow as tf

import cv2
from scipy.misc import imread

import time

import os, random

from PIL import Image

import functools
import itertools

import collections
import six

from tensorflow.python.platform import tf_logging as logging
from tensorflow.core.framework import node_def_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.training import basic_session_run_hooks
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training import device_setter
from tensorflow.contrib.learn.python.learn import run_config

slim = tf.contrib.slim

tf.logging.set_verbosity(tf.logging.DEBUG)

features1 = 16
features2 = 32
features3 = 64
features4 = 128
features5 = 256
features6 = 768

gen_features1 = 32 #256
gen_features2 = 64 #128
gen_features3 = 128 #64
gen_features4 = 256 #32
gen_features5 = 512 #32
gen_features6 = 768 #16
gen_features7 = 1024 #8
gen_features8 = 1536 #8
gen_features9 = 2048 #8

dec_features1 = 512 #8
dec_features2 = 384 #16
dec_features3 = 256 #32
dec_features4 = 128 #64
dec_features5 = 64 #128
dec_features6 = 32 #256

num_global_enhancer_blocks = 8
num_local_enhancer_blocks = 5

data_dir = "X:/Jeffrey-Ede/stills_all/"
#data_dir = "E:/stills_hq-mini/"

modelSavePeriod = 4 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "X:/Jeffrey-Ede/models/gan-unsupervised-latent-3/"

shuffle_buffer_size = 5000
num_parallel_calls = 8
num_parallel_readers = 8
prefetch_buffer_size = 20
batch_size = 1
num_gpus = 1

def num_examples_per_epoch(subset='train'):
    if subset == 'train':
        return 65536
    elif subset == 'validation':
        return 4096
    elif subset == 'eval':
        return 16384
    else:
        raise ValueError('Invalid data subset "%s"' % subset)

#batch_size = 8 #Batch size to use during training
num_epochs = 1000000 #Dataset repeats indefinitely

logDir = "C:/dump/train/"
log_file = model_dir+"log.txt"
val_log_file = model_dir+"val_log.txt"
discr_pred_file = model_dir+"discr_pred.txt"
log_every = 1 #Log every _ examples
cumProbs = np.array([]) #Indices of the distribution plus 1 will be correspond to means

numMeans = 64 // batch_size
scaleMean = 4 #Each means array index increment corresponds to this increase in the mean
numDynamicGrad = 1 #Number of gradients to calculate for each possible mean when dynamically updating training
lossSmoothingBoxcarSize = 5

channels = 1 #Greyscale input image

#Sidelength of images to feed the neural network
cropsize = 256
generator_input_size = cropsize
height_crop = width_crop = cropsize

#hparams = experiment_hparams(train_batch_size=batch_size, eval_batch_size=16)

weight_decay = 0.0
initial_learning_rate = 0.001
initial_discriminator_learning_rate = 0.001
num_workers = 1

increase_batch_size_by_factor = 1
effective_batch_size = increase_batch_size_by_factor*batch_size

save_result_every_n_batches = 25000

val_skip_n = 10
trainee_switch_skip_n = 1

max_num_since_change = 0

flip_weight = 42

def _tf_fspecial_gauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function
    """
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]

    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)

    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)

    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)


def tf_ssim(img1, img2, cs_map=False, mean_metric=True, size=11, sigma=1.5):
    window = _tf_fspecial_gauss(size, sigma) # window shape [size, size]
    K1 = 0.01
    K2 = 0.03
    L = 1  # depth of image (255 in case the image has a differnt scale)
    C1 = (K1*L)**2
    C2 = (K2*L)**2
    mu1 = tf.nn.conv2d(img1, window, strides=[1,1,1,1], padding='VALID')
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1], padding='VALID')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides=[1,1,1,1],padding='VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides=[1,1,1,1],padding='VALID') - mu2_sq
    sigma12 = tf.nn.conv2d(img1*img2, window, strides=[1,1,1,1],padding='VALID') - mu1_mu2
    if cs_map:
        value = (((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2)),
                (2.0*sigma12 + C2)/(sigma1_sq + sigma2_sq + C2))
    else:
        value = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*
                    (sigma1_sq + sigma2_sq + C2))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value


def tf_ms_ssim(img1, img2, mean_metric=True, level=5):
    weight = tf.constant([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=tf.float32)
    mssim = []
    mcs = []
    for l in range(level):
        ssim_map, cs_map = tf_ssim(img1, img2, cs_map=True, mean_metric=False)
        mssim.append(tf.reduce_mean(ssim_map))
        mcs.append(tf.reduce_mean(cs_map))
        filtered_im1 = tf.nn.avg_pool(img1, [1,2,2,1], [1,2,2,1], padding='SAME')
        filtered_im2 = tf.nn.avg_pool(img2, [1,2,2,1], [1,2,2,1], padding='SAME')
        img1 = filtered_im1
        img2 = filtered_im2

    # list to tensor of dim D+1
    mssim = tf.pack(mssim, axis=0)
    mcs = tf.pack(mcs, axis=0)

    value = (tf.reduce_prod(mcs[0:level-1]**weight[0:level-1])*
                            (mssim[level-1]**weight[level-1]))

    if mean_metric:
        value = tf.reduce_mean(value)
    return value

def tf_median(v):
    v = tf.reshape(v, [-1])
    m = v.get_shape()[0]//2
    return tf.nn.top_k(v, m).values[m-1]

def generator_architecture(inputs, phase=False, params=None):
    """Generates fake data to try and fool the discrimator"""

    with tf.variable_scope("GAN/Gen"):

        concat_axis = 3

        def _instance_norm(net, train=phase):
            batch, rows, cols, channels = [i.value for i in net.get_shape()]
            var_shape = [channels]
            mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
            shift = tf.Variable(tf.zeros(var_shape), trainable=False)
            scale = tf.Variable(tf.ones(var_shape), trainable=False)
            epsilon = 1e-3
            normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
            return scale * normalized + shift

        def instance_then_activ(input):
            batch_then_activ = _instance_norm(input)
            batch_then_activ = tf.nn.relu(batch_then_activ)
            return batch_then_activ

        def pad(tensor, size):
            d1_pad = size[0]
            d2_pad = size[1]

            paddings = tf.constant([[0, 0], [d1_pad, d1_pad], [d2_pad, d2_pad], [0, 0]], dtype=tf.int32)
            padded = tf.pad(tensor, paddings, mode="REFLECT")
            return padded

        ##Reusable blocks
        def _batch_norm_fn(input):
            batch_norm = tf.contrib.layers.batch_norm(
                input,
                decay=0.9997,
                center=True, scale=True,
                is_training=True,
                fused=True,
                zero_debias_moving_mean=False,
                renorm=False)
            return batch_norm

        def batch_then_activ(input):
            batch_then_activ = _batch_norm_fn(input)
            batch_then_activ = tf.nn.relu(batch_then_activ)
            return batch_then_activ

        def conv_block_not_sep(input, filters, kernel_size=3, phase=phase, pad_size=None, batch_and_activ=True):

            if pad_size:
                input = pad(input, pad_size)

            conv_block = slim.conv2d(
                inputs=input,
                num_outputs=filters,
                kernel_size=kernel_size,
                activation_fn=None,
                padding='SAME' if not pad_size else 'VALID')

            if batch_and_activ:
                conv_block = batch_then_activ(conv_block)

            return conv_block

        def conv_block(input, filters, phase=phase, pad_size=None):
            """
            Convolution -> batch normalisation -> activation
            phase defaults to true, meaning that the network is being trained
            """

            conv_block = strided_conv_block(input, filters, 1, 1, pad_size=pad_size)

            return conv_block

        def strided_conv_block(input, filters, stride, rate=1, phase=phase, 
                               extra_batch_norm=True, kernel_size=3, pad_size=None):
            if pad_size:
                strided_conv = pad(input, pad_size)
            else:
                strided_conv = input

            strided_conv = slim.separable_convolution2d(
                inputs=strided_conv,
                num_outputs=filters,
                kernel_size=kernel_size,
                depth_multiplier=1,
                stride=stride,
                data_format='NHWC',
                padding='SAME' if not pad_size else 'VALID',
                rate=rate,
                activation_fn=None,#tf.nn.relu,
                normalizer_fn=_batch_norm_fn if extra_batch_norm else None,
                normalizer_params=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True,
                scope=None)
            strided_conv = batch_then_activ(strided_conv)

            return strided_conv

        def residual_conv(input, filters, pad_size=None):

            if pad_size:
                input = pad(input, pad_size)

            residual = slim.conv2d(
                inputs=input,
                num_outputs=filters,
                kernel_size=1,
                stride=2,
                activation_fn=None,
                padding='SAME' if not pad_size else 'VALID')
            residual = batch_then_activ(residual)

            return residual

        def deconv_block(input, filters, new_size, pad_size=None):
            '''Transpositionally convolute a feature space to upsample it'''

            deconv = tf.image.resize_images(images=input, size=new_size)
            deconv = conv_block(deconv, filters, pad_size)

            return deconv

        def xception_encoding_block(input, features):
        
            cnn = conv_block(
                input=input, 
                filters=features)
            cnn = conv_block(
                input=cnn1, 
                filters=features)
            cnn = strided_conv_block(
                input=cnn,
                filters=features,
                stride=2)

            residual = residual_conv(input, features)
            cnn += residual

            return cnn

        def xception_middle_block(input, features, pad_size=None):
        
            main_flow = strided_conv_block(
                input=input,
                filters=features,
                stride=1, 
                pad_size=pad_size)
            main_flow = strided_conv_block(
                input=main_flow,
                filters=features,
                stride=1,
                pad_size=pad_size)
            main_flow = strided_conv_block(
                input=main_flow,
                filters=features,
                stride=1,
                pad_size=pad_size)

            return main_flow + input

        def xception_encoding_block_diff(input, features_start, features_end):
        
            cnn = conv_block(
                input=input, 
                filters=features_start)
            cnn = conv_block(
                input=cnn, 
                filters=features_start)
            cnn = strided_conv_block(
                input=cnn,
                filters=features_end,
                stride=2)

            residual = residual_conv(input, features_end)
            cnn += residual

            return cnn

        ##Model building
        input_layer = tf.reshape(inputs, [-1, generator_input_size, generator_input_size, channels])

        #256
        enc = strided_conv_block(input=input_layer,
                                 filters=gen_features1,
                                 stride=2,
                                 kernel_size=3)

        #128
        enc = strided_conv_block(enc, gen_features3, 1, 1)
        enc = strided_conv_block(input=input_layer,
                                 filters=gen_features2,
                                 stride=2,
                                 kernel_size=3)

        #64
        enc = strided_conv_block(enc, gen_features3, 1, 1)
        enc = strided_conv_block(input=input_layer,
                                 filters=gen_features3,
                                 stride=2,
                                 kernel_size=3)

        #32
        enc = strided_conv_block(enc, gen_features4, 1, 1)
        enc = strided_conv_block(input=input_layer,
                                 filters=gen_features4,
                                 stride=2,
                                 kernel_size=3)

        #16
        enc = xception_encoding_block_diff(enc, gen_features5, gen_features6)

        #8
        enc = strided_conv_block(enc, gen_features7, 1, 1)
        enc = strided_conv_block(enc, gen_features8, 1, 1)
        enc = strided_conv_block(enc, gen_features9, 1, 1)

        global_avg = tf.reduce_mean(enc, [1,2])

        fc = tf.reshape(global_avg, (-1, gen_features9))
        fc = tf.contrib.layers.fully_connected(inputs=fc,
                                               num_outputs=gen_features9,
                                               activation_fn=None,
                                               trainable=phase)
        fc = batch_then_activ(fc)

        fc = tf.contrib.layers.dropout(fc, keep_prob=0.75, is_training=phase)

        ### End of encoding. Start of decoding ###
        
        fc = tf.contrib.layers.fully_connected(inputs=fc,
                                               num_outputs=gen_features9,
                                               activation_fn=None,
                                               trainable=phase)
        fc = batch_then_activ(fc)
        
        dec = tf.reshape(fc, [-1, 4, 4, gen_features9//(4*4)])

        #4
        dec = deconv_block(dec, dec_features1, (8, 8), pad_size=(1,1))

        #8
        dec = deconv_block(dec, dec_features2, (16, 16), pad_size=(1,1))

        #16
        dec = deconv_block(dec, dec_features3, (32, 32), pad_size=(1,1))
        dec = conv_block(dec, dec_features3, pad_size=(1,1))

        #32
        dec = deconv_block(dec, dec_features4, (64, 64), pad_size=(1,1))
        dec = conv_block(dec, dec_features4, pad_size=(1,1))

        #64
        dec = deconv_block(dec, dec_features5, (128, 128), pad_size=(1,1))
        for _ in range(5):
            dec = xception_middle_block(dec, dec_features5, pad_size=(1,1))

        #128
        dec = deconv_block(dec, dec_features6, (256, 256), pad_size=(1,1))
        dec = conv_block(dec, dec_features6, pad_size=(1,1))

        dec = pad(dec, (1,1))
        dec = slim.conv2d(
                inputs=dec,
                num_outputs=1,
                kernel_size=3,
                padding="VALID",
                activation_fn=None)
        dec = _instance_norm(dec)

        dec = tf.tanh(dec)

        return dec

def discriminator_architecture(inputs, phase=False, params=None, gen_loss=0., reuse=False):
    """Discriminates between real and fake data"""

    with tf.variable_scope("GAN/Discr", reuse=False):

        #phase = mode == tf.estimator.ModeKeys.TRAIN #phase is true during training
        concat_axis = 3

        def _instance_norm(net, train=phase):
            batch, rows, cols, channels = [i.value for i in net.get_shape()]
            var_shape = [channels]
            mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
            shift = tf.Variable(tf.zeros(var_shape), trainable=False)
            scale = tf.Variable(tf.ones(var_shape), trainable=False)
            epsilon = 1e-3
            normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
            return scale * normalized + shift

        def instance_then_activ(input):
            batch_then_activ = _instance_norm(input)
            batch_then_activ = tf.nn.relu(batch_then_activ)
            return batch_then_activ

        ##Reusable blocks
        def _batch_norm_fn(input):
            batch_norm = tf.contrib.layers.batch_norm(
                input,
                decay=0.9997,
                center=True, scale=True,
                is_training=True,
                fused=True,
                zero_debias_moving_mean=False,
                renorm=False)
            return batch_norm

        def batch_then_activ(input):
            batch_then_activ = _batch_norm_fn(input)
            batch_then_activ = tf.nn.leaky_relu(batch_then_activ)
            return batch_then_activ

        def conv_block_not_sep(input, filters, kernel_size=3, phase=phase):
            """
            Convolution -> batch normalisation -> leaky relu
            phase defaults to true, meaning that the network is being trained
            """

            conv_block = slim.conv2d(
                inputs=input,
                num_outputs=filters,
                kernel_size=kernel_size,
                padding="SAME",
                activation_fn=None)
            conv_block = batch_then_activ(conv_block)

            return conv_block

        def conv_block(input, filters, phase=phase):
            """
            Convolution -> batch normalisation -> leaky relu
            phase defaults to true, meaning that the network is being trained
            """

            conv_block = strided_conv_block(input, filters, 1, 1)

            return conv_block

        def strided_conv_block(input, filters, stride, rate=1, phase=phase, 
                               extra_batch_norm=True):
        
            strided_conv = slim.separable_convolution2d(
                inputs=input,
                num_outputs=filters,
                kernel_size=3,
                depth_multiplier=1,
                stride=stride,
                padding='SAME',
                data_format='NHWC',
                rate=rate,
                activation_fn=None,#tf.nn.relu,
                normalizer_fn=_batch_norm_fn if extra_batch_norm else False,
                normalizer_params=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=None,
                biases_initializer=tf.zeros_initializer(),
                biases_regularizer=None,
                reuse=None,
                variables_collections=None,
                outputs_collections=None,
                trainable=True)
            strided_conv = batch_then_activ(strided_conv)

            return strided_conv

        def residual_conv(input, filters):

            residual = slim.conv2d(
                inputs=input,
                num_outputs=filters,
                kernel_size=1,
                stride=2,
                padding="SAME",
                activation_fn=None)
            residual = batch_then_activ(residual)

            return residual

        def xception_encoding_block(input, features):
        
            cnn = conv_block(
                input=input, 
                filters=features)
            cnn = conv_block(
                input=cnn, 
                filters=features)
            cnn = strided_conv_block(
                input=cnn,
                filters=features,
                stride=2)

            residual = residual_conv(input, features)
            cnn += residual

            return cnn

        def xception_encoding_block_diff(input, features_start, features_end):
        
            cnn = conv_block(
                input=input, 
                filters=features_start)
            cnn = conv_block(
                input=cnn, 
                filters=features_start)
            cnn = strided_conv_block(
                input=cnn,
                filters=features_end,
                stride=2)

            residual = residual_conv(input, features_end)
            cnn += residual

            return cnn

        def xception_middle_block(input, features):
        
            main_flow = strided_conv_block(
                input=input,
                filters=features,
                stride=1)
            main_flow = strided_conv_block(
                input=main_flow,
                filters=features,
                stride=1)
            main_flow = strided_conv_block(
                input=main_flow,
                filters=features,
                stride=1)

            return main_flow + input

        def shared_flow(input, layers):

            shared = xception_encoding_block_diff(input, features2, features3)
            layers.append(shared)
            shared = xception_encoding_block_diff(shared, features3, features4)
            layers.append(shared)
        
            shared = xception_encoding_block(shared, features5)
            layers.append(shared)

            shared = xception_middle_block(shared, features5)
            layers.append(shared)
            shared = xception_middle_block(shared, features5)
            layers.append(shared)
            shared = xception_middle_block(shared, features5)
            layers.append(shared)
            shared = xception_middle_block(shared, features5)
            layers.append(shared)

            return shared, layers

        def terminating_fc(input):

            fc = tf.reduce_mean(input, [1,2])
            fc = tf.reshape(fc, (-1, features5))
            fc = tf.contrib.layers.fully_connected(inputs=fc,
                                                   num_outputs=1,
                                                   activation_fn=None)
            return fc

        '''Model building'''        
        layers = []

        with tf.variable_scope("small-start", reuse=reuse) as small_start_scope:
            small = inputs[0]
            small = strided_conv_block(small, features1, 1, 1)
            layers.append(small)
            small = strided_conv_block(small, features2, 1, 1)
            layers.append(small)

        with tf.variable_scope("medium-start", reuse=reuse):
            medium = inputs[1]
            medium = tf.nn.avg_pool(medium,
                                    [1, 2, 2, 1],
                                    strides=[1, 2, 2, 1],
                                    padding='SAME')
            medium = strided_conv_block(medium, features1, 1, 1)
            layers.append(medium)
            medium = strided_conv_block(medium, features2, 1, 1)
            layers.append(medium)

        with tf.variable_scope("large-start", reuse=reuse):
            large = inputs[2]
            #large = tf.nn.avg_pool(large,
            #                        [1, 4, 4, 1],
            #                        strides=[1, 4, 4, 1],
            #                        padding='SAME')
            large = strided_conv_block(large, features1, 1, 1)
            layers.append(large)
            large = strided_conv_block(large, features2, 1, 1)
            layers.append(large)

        with tf.variable_scope("shared", reuse=reuse) as shared_scope:
            small, layers = shared_flow(small, layers)
        with tf.variable_scope(shared_scope, reuse=True):
            medium, layers = shared_flow(medium, layers)
        with tf.variable_scope(shared_scope, reuse=True):
            large, layers = shared_flow(large, layers)

        with tf.variable_scope("small-end", reuse=reuse) as small_end_scope:
            small = xception_middle_block(small, features5)
            layers.append(small)
            small = xception_middle_block(small, features5)
            layers.append(small)
            small = slim.conv2d(
                inputs=large,
                num_outputs=features5,
                kernel_size=3,
                padding="SAME",
                activation_fn=tf.nn.leaky_relu)
            small = instance_then_activ(small)
            small = terminating_fc(small)

        with tf.variable_scope("medium-end", reuse=reuse):
            medium = xception_middle_block(medium, features5)
            layers.append(medium)
            medium = xception_middle_block(medium, features5)
            layers.append(medium)
            medium = slim.conv2d(
                inputs=medium,
                num_outputs=features5,
                kernel_size=3,
                padding="SAME",
                activation_fn=tf.nn.leaky_relu)
            medium = instance_then_activ(medium)
            medium = terminating_fc(medium)

        with tf.variable_scope("large-end", reuse=reuse):
            large = xception_middle_block(large, features5)
            layers.append(large)
            large = xception_middle_block(large, features5)
            layers.append(large)
            large = slim.conv2d(
                inputs=large,
                num_outputs=features5,
                kernel_size=3,
                padding="SAME",
                activation_fn=tf.nn.leaky_relu)
            large = instance_then_activ(large)
            large = terminating_fc(large)

        output = tf.sigmoid(tf.reduce_max(tf.concat([small, medium, large], axis=1), axis=1))

    return [output] + layers

class ExamplesPerSecondHook(session_run_hook.SessionRunHook):
    """Hook to print out examples per second.
        Total time is tracked and then divided by the total number of steps
        to get the average step time and then batch_size is used to determine
        the running average of examples per second. The examples per second for the
        most recent interval is also logged.
    """

    def __init__(
        self,
        batch_size,
        every_n_steps=100,
        every_n_secs=None,):
        """Initializer for ExamplesPerSecondHook.
          Args:
          batch_size: Total batch size used to calculate examples/second from
          global time.
          every_n_steps: Log stats every n steps.
          every_n_secs: Log stats every n seconds.
        """
        if (every_n_steps is None) == (every_n_secs is None):
            raise ValueError('exactly one of every_n_steps and every_n_secs should be provided.')
        self._timer = basic_session_run_hooks.SecondOrStepTimer(
            every_steps=every_n_steps, every_secs=every_n_secs)

        self._step_train_time = 0
        self._total_steps = 0
        self._batch_size = batch_size

    def begin(self):
        self._global_step_tensor = training_util.get_global_step()
        if self._global_step_tensor is None:
            raise RuntimeError('Global step should be created to use StepCounterHook.')

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return basic_session_run_hooks.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        _ = run_context

        global_step = run_values.results
        if self._timer.should_trigger_for_step(global_step):
            elapsed_time, elapsed_steps = self._timer.update_last_triggered_step(
                global_step)
            if elapsed_time is not None:
                steps_per_sec = elapsed_steps / elapsed_time
                self._step_train_time += elapsed_time
                self._total_steps += elapsed_steps

                average_examples_per_sec = self._batch_size * (
                    self._total_steps / self._step_train_time)
                current_examples_per_sec = steps_per_sec * self._batch_size
                # Average examples/sec followed by current examples/sec
                logging.info('%s: %g (%g), step = %g', 'Average examples/sec',
                             average_examples_per_sec, current_examples_per_sec,
                             self._total_steps)

def local_device_setter(num_devices=1,
                        ps_device_type='cpu',
                        worker_device='/cpu:0',
                        ps_ops=None,
                        ps_strategy=None):
    if ps_ops == None:
        ps_ops = ['Variable', 'VariableV2', 'VarHandleOp']

    if ps_strategy is None:
        ps_strategy = device_setter._RoundRobinStrategy(num_devices)
    if not six.callable(ps_strategy):
        raise TypeError("ps_strategy must be callable")

    def _local_device_chooser(op):
        current_device = pydev.DeviceSpec.from_string(op.device or "")

        node_def = op if isinstance(op, node_def_pb2.NodeDef) else op.node_def
        if node_def.op in ps_ops:
            ps_device_spec = pydev.DeviceSpec.from_string(
                '/{}:{}'.format(ps_device_type, ps_strategy(op)))

            ps_device_spec.merge_from(current_device)
            return ps_device_spec.to_string()
        else:
            worker_device_spec = pydev.DeviceSpec.from_string(worker_device or "")
            worker_device_spec.merge_from(current_device)
            return worker_device_spec.to_string()

    return _local_device_chooser


def get_model_fn(num_gpus, variable_strategy, num_workers, component=""):
    """Returns a function that will build the model."""
    if component == "generator":
        def _model_fn(features, mode=None, params=None):
            """Model body.

            Support single host, one or more GPU training. Parameter distribution can
            be either one of the following scheme.
            1. CPU is the parameter server and manages gradient updates.
            2. Parameters are distributed evenly across all GPUs, and the first GPU
            manages gradient updates.
    
            Args:
                features: a list of tensors, one for each tower
                mode: ModeKeys.TRAIN or EVAL
                params: Hyperparameters suitable for tuning
            Returns:
                An EstimatorSpec object.
            """
            is_training = mode#(mode == tf.estimator.ModeKeys.TRAIN)
            momentum = params.momentum

            tower_features = features
            tower_grads = []
            tower_preds = []
            discr_preds = []

            # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
            # on CPU. The exception is Intel MKL on CPU which is optimal with
            # channels_last.
            data_format = params.data_format
            if not data_format:
                if num_gpus == 0:
                    data_format = 'channels_last'
                else:
                    data_format = 'channels_first'

            if num_gpus == 0:
                num_devices = 1
                device_type = 'cpu'
            else:
                num_devices = num_gpus
                device_type = 'gpu'

            for i in range(num_devices):
                worker_device = '/{}:{}'.format(device_type, i)
                if variable_strategy == 'CPU':
                    device_setter = local_device_setter(
                        worker_device=worker_device)
                elif variable_strategy == 'GPU':
                    device_setter = local_device_setter(
                        ps_device_type='gpu',
                        worker_device=worker_device,
                        ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                            num_gpus, tf.contrib.training.byte_size_load_fn))
                with tf.variable_scope('nn', reuse=bool(i != 0)):
                    with tf.name_scope('tower_%d' % i) as name_scope:
                        with tf.device(device_setter):
                            grads, preds, preds_d = _generator_tower_fn(
                                is_training, tower_features[i])

                            tower_grads.append(grads)
                            tower_preds.append(preds)
                            discr_preds.append(preds_d)
                            if i == 0:
                                # Only trigger batch_norm moving mean and variance update from
                                # the 1st tower. Ideally, we should grab the updates from all
                                # towers but these stats accumulate extremely fast so we can
                                # ignore the other stats from the other towers without
                                # significant detriment.
                                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

            _tower_preds_tmp = tf.stack(preds)
            _discr_preds_tmp = tf.stack(preds_d)
        
            return [_tower_preds_tmp, _discr_preds_tmp, update_ops] + tower_grads

    if component =="discriminator":
        def _model_fn(features, labels=None, mode=None, params=None):
            """Model body.

            Support single host, one or more GPU training. Parameter distribution can
            be either one of the following scheme.
            1. CPU is the parameter server and manages gradient updates.
            2. Parameters are distributed evenly across all GPUs, and the first GPU
            manages gradient updates.
    
            Args:
                features: a list of tensors, one for each tower
                mode: ModeKeys.TRAIN or EVAL
                params: Hyperparameters suitable for tuning
            Returns:
                An EstimatorSpec object.
            """
            is_training = mode#(mode == tf.estimator.ModeKeys.TRAIN)
            momentum = params.momentum

            tower_features = features
            tower_labels = labels
            tower_losses = []
            tower_grads = []
            tower_preds = []
            tower_params = []

            # channels first (NCHW) is normally optimal on GPU and channels last (NHWC)
            # on CPU. The exception is Intel MKL on CPU which is optimal with
            # channels_last.
            data_format = params.data_format
            if not data_format:
                if num_gpus == 0:
                    data_format = 'channels_last'
                else:
                    data_format = 'channels_first'

            if num_gpus == 0:
                num_devices = 1
                device_type = 'cpu'
            else:
                num_devices = num_gpus
                device_type = 'gpu'

            for i in range(num_devices):
                worker_device = '/{}:{}'.format(device_type, i)
                if variable_strategy == 'CPU':
                    device_setter = local_device_setter(
                        worker_device=worker_device)
                elif variable_strategy == 'GPU':
                    device_setter = local_device_setter(
                        ps_device_type='gpu',
                        worker_device=worker_device,
                        ps_strategy=tf.contrib.training.GreedyLoadBalancingStrategy(
                            num_gpus, tf.contrib.training.byte_size_load_fn))
                with tf.variable_scope('nn', reuse=bool(i != 0)):
                    with tf.name_scope('tower_%d' % i) as name_scope:
                        with tf.device(device_setter):
                            loss, grads, preds = _discriminator_tower_fn(
                                is_training, tower_features[i], tower_labels[i])

                            tower_losses.append(loss)
                            tower_grads.append(grads)
                            tower_preds.append(preds)
                            if i == 0:
                                # Only trigger batch_norm moving mean and variance update from
                                # the 1st tower. Ideally, we should grab the updates from all
                                # towers but these stats accumulate extremely fast so we can
                                # ignore the other stats from the other towers without
                                # significant detriment.
                                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

            _tower_losses_tmp = tf.tuple(tower_losses)
            _tower_preds_tmp = tf.stack(preds)
        
            return [_tower_losses_tmp, _tower_preds_tmp, update_ops] + tower_grads

    return _model_fn

def get_multiscale_crops(input, multiscale_channels=1):

    small = tf.random_crop(
                input,
                size=(batch_size, cropsize//4, cropsize//4, multiscale_channels))
    medium = tf.random_crop(
                input,
                size=(batch_size, cropsize//2, cropsize//2, multiscale_channels))
    large = tf.random_crop(
                input,
                size=(batch_size, (3*cropsize)//4, (3*cropsize)//4, multiscale_channels))
    large = tf.image.resize_images(large, (cropsize//4, cropsize//4))

    return small, medium, large

def _generator_tower_fn(is_training, feature):
    """Build computation tower.
        Args:
        is_training: true if is training graph.
        feature: a Tensor.
        Returns:
        A tuple with the loss for the tower, the gradients and parameters, and
        predictions.
    """

    #phase = tf.estimator.ModeKeys.TRAIN if is_training else tf.estimator.ModeKeys.EVAL
    feature = tf.reshape(feature, [-1, cropsize, cropsize, channels])
    output = generator_architecture(feature, is_training)
    model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="nn/GAN/Gen")

    concat = tf.concat([output, feature], axis=3)
    shapes = [(batch_size, cropsize//4, cropsize//4, channels),
              (batch_size, cropsize//2, cropsize//2, channels),
              (batch_size, cropsize//4, cropsize//4, channels)]
    multiscale_crops = get_multiscale_crops(concat, multiscale_channels=2)
    multiscale_crops = [tf.unstack(crop, axis=3) for crop in multiscale_crops]
    multiscale = [tf.reshape(unstacked[0], shape) 
                  for unstacked, shape in zip(multiscale_crops, shapes)]
    multiscale_natural = [tf.reshape(unstacked[1], shape) 
                          for unstacked, shape in zip(multiscale_crops, shapes)]

    discrimination = discriminator_architecture(multiscale, is_training, reuse=False)
    output_d = discrimination[0]

    #Compare discrimination features for generation against those for a real image
    discrimination_natural = discriminator_architecture(multiscale_natural, is_training, reuse=True)

    natural_stat_losses = []
    for i in range(1, len(discrimination)):
        natural_stat_losses.append(
            tf.reduce_mean(
                tf.losses.absolute_difference(
                    discrimination[i], discrimination_natural[i])))
    natural_stat_loss = tf.add_n(natural_stat_losses)

    weight_natural_stats = 7.

    loss = -tf.log(tf.maximum(output_d, 1e-9))
    loss += weight_natural_stats*natural_stat_loss
    loss += 1.e-5 * tf.add_n(
        [tf.nn.l2_loss(v) for v in model_params])
    
    tower_grad = tf.gradients(loss, model_params)

    return tower_grad, output, output_d


def _discriminator_tower_fn(is_training, feature, ground_truth):
    """Build computation tower.
        Args:
        is_training: true if is training graph.
        feature: a Tensor.
        Returns:
        A tuple with the loss for the tower, the gradients and parameters, and
        predictions.
    """

    feature = tf.reshape(feature, [-1, cropsize, cropsize, channels])
    shapes = [(batch_size, cropsize//4, cropsize//4, channels),
              (batch_size, cropsize//2, cropsize//2, channels),
              (batch_size, cropsize//4, cropsize//4, channels)]
    multiscale_crops = get_multiscale_crops(feature)
    multiscale_crops = [tf.reshape(crop, shape) 
                         for crop, shape in zip(multiscale_crops, shapes)]

    output = discriminator_architecture(multiscale_crops, is_training, reuse=True)[0]

    model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="nn/GAN/Discr")

    tower_loss = -tf.cond(ground_truth[0] > 0.5, #Only works for batch size 1
                          lambda: tf.log(tf.maximum(output, 1e-9)), 
                          lambda: tf.log(tf.maximum(1.-output, 1e-9)))
    tower_loss += 5.e-5 * tf.add_n([tf.nn.l2_loss(v) for v in model_params])

    tower_grad = tf.gradients(tower_loss, model_params)

    return tower_loss, tower_grad, output


def flip_rotate(img):
    """Applies a random flip || rotation to the image, possibly leaving it unchanged"""

    choice = int(8*np.random.rand())
    
    if choice == 0:
        return img
    if choice == 1:
        return np.rot90(img, 1)
    if choice == 2:
        return np.rot90(img, 2)
    if choice == 3:
        return np.rot90(img, 3)
    if choice == 4:
        return np.flip(img, 0)
    if choice == 5:
        return np.flip(img, 1)
    if choice == 6:
        return np.flip(np.rot90(img, 1), 0)
    if choice == 7:
        return np.flip(np.rot90(img, 1), 1)


def load_image(addr, resizeSize=None, imgType=np.float32):
    """Read an image and make sure it is of the correct type. Optionally resize it"""
    
    try:
        img = imread(addr, mode='F')
    except:
        img = 0.5*np.ones((512,512))
        print("Image read failed")

    if resizeSize:
        img = cv2.resize(img, resizeSize, interpolation=cv2.INTER_AREA)

    return img.astype(imgType)


def scale0to1(img):
    """Rescale image between 0 and 1"""

    min = np.min(img)
    max = np.max(img)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def norm_img(img):
    
    min = np.min(img)
    max = np.max(img)

    if min == max:
        img.fill(0.)
    else:
        a = 0.5*(min+max)
        b = 0.5*(max-min)

        img = (img-a) / b

    return img.astype(np.float32)

def preprocess(img):

    img[np.isnan(img)] = 0.
    img[np.isinf(img)] = 0.

    img = cv2.resize(img, (cropsize, cropsize))

    img = norm_img(img)

    return img

def record_parser(record):
    """Parse files and generate lower quality images from them"""
    return preprocess(flip_rotate(load_image(record)))

def reshaper(img):
    img = tf.reshape(img, [cropsize, cropsize, channels])
    return img


def input_fn(dir, subset, batch_size, num_shards):
    """Create a dataset from a list of filenames and shard batches from it"""

    with tf.device('/cpu:0'):

        dataset = tf.data.Dataset.list_files(dir+subset+"/"+"*.tif")
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(
            lambda file: tf.py_func(record_parser, [file], [tf.float32]),
            num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(reshaper, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

        iter = dataset.make_one_shot_iterator()
        img_batch = iter.get_next()

        return [img_batch]

        #tensors = tf.unstack(image_batch, num=batch_size, axis=0)
        #feature_shards = [[] for i in range(num_shards)]
        #for i in range(batch_size):
        #    idx = i % num_shards
        #    feature_shards[idx].append(tensors[i])
        #feature_shards = [tf.parallel_stack(x) for x in feature_shards]

        #return feature_shards

def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return

def get_experiment_fn(data_dir,
                      num_gpus,
                      variable_strategy):
    """
    Returns an experiment function
    Experiments perform training on several workers in parallel,
    in other words experiments know how to invoke train and eval in a sensible
    fashion for distributed training. Arguments passed directly to this
    function are not tunable, all other arguments should be passed within
    tf.HParams, passed to the enclosed function.
    Args:
        data_dir: str. Location of the data for input_fns.
        num_gpus: int. Number of GPUs on each worker.
        variable_strategy: String. CPU to use CPU as the parameter server
        and GPU to use the GPUs as the parameter server.
    Returns:
        A function (tf.estimator.RunConfig, tf.contrib.training.HParams) ->
        tf.contrib.learn.Experiment.
        Suitable for use by tf.contrib.learn.learn_runner, which will run various
        methods on Experiment (train, evaluate) based on information
        about the current runner in `run_config`.
    """
  
    def _experiment_fn(run_config, hparams):
        """Returns an Experiment."""
        # Create estimator.
        train_input_fn = functools.partial(
            input_fn,
            data_dir,
            subset='train',
            num_shards=num_gpus,
            batch_size=hparams.train_batch_size)

        eval_input_fn = functools.partial(
            input_fn,
            data_dir,
            subset='eval',
            batch_size=hparams.eval_batch_size,
            num_shards=num_gpus)

        num_eval_examples = num_examples_per_epoch('eval')
        if num_eval_examples % hparams.eval_batch_size != 0:
            print(num_eval_examples, hparams.eval_batch_size)
            raise ValueError(
                'validation set size must be multiple of eval_batch_size')

        train_steps = hparams.train_steps
        eval_steps = num_eval_examples // hparams.eval_batch_size
 
        model = tf.estimator.Estimator(
            model_fn=get_model_fn(num_gpus, variable_strategy,
                                  run_config.num_worker_replicas or 1),
            config=run_config,
            params=hparams)

        # Create experiment.
        return tf.contrib.learn.Experiment(
            model,
            train_input_fn=train_input_fn,
            eval_input_fn=eval_input_fn,
            train_steps=train_steps,
            eval_steps=eval_steps)

    return _experiment_fn

class RunConfig(tf.contrib.learn.RunConfig): 
    def uid(self, whitelist=None):
        """
        Generates a 'Unique Identifier' based on all internal fields.
        Caller should use the uid string to check `RunConfig` instance integrity
        in one session use, but should not rely on the implementation details, which
        is subject to change.
        Args:
          whitelist: A list of the string names of the properties uid should not
            include. If `None`, defaults to `_DEFAULT_UID_WHITE_LIST`, which
            includes most properties user allowes to change.
        Returns:
          A uid string.
        """
        if whitelist is None:
            whitelist = run_config._DEFAULT_UID_WHITE_LIST

        state = {k: v for k, v in self.__dict__.items() if not k.startswith('__')}
        # Pop out the keys in whitelist.
        for k in whitelist:
            state.pop('_' + k, None)

        ordered_state = collections.OrderedDict(
            sorted(state.items(), key=lambda t: t[0]))
        # For class instance without __repr__, some special cares are required.
        # Otherwise, the object address will be used.
        if '_cluster_spec' in ordered_state:
            ordered_state['_cluster_spec'] = collections.OrderedDict(
                sorted(ordered_state['_cluster_spec'].as_dict().items(), key=lambda t: t[0]))
        return ', '.join(
            '%s=%r' % (k, v) for (k, v) in six.iteritems(ordered_state))

def _train_op(variable_strategy, 
              update_ops, 
              learning_rate_ph, 
              **kwargs):

    with tf.variable_scope("generator"):
    
        tower_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="nn/GAN/Gen")
        tower_gradvars = []
        for tower_grad in kwargs['_tower_grads']:
            tower_gradvars.append(zip(tower_grad, tower_params))

        # Now compute global loss and gradients.
        gradvars = []
        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for grad, var in itertools.chain(*tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in six.iteritems(all_grads):
                # Average gradients on the same device as the variables
                # to which they apply.
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))

                gradvars.append((avg_grad, var))
                
        global_step = tf.train.get_global_step()

        # Device that runs the ops to apply global gradient updates.
        consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
        with tf.device(consolidation_device):

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_ph, beta1=0.5)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

        # Create single grouped train op
        train_op = [optimizer.apply_gradients(gradvars, global_step=global_step)]

        train_op.extend(update_ops)
        train_op = tf.group(*train_op)

    return train_op


def _discriminator_train_op(discriminator_tower_losses_ph, 
                            discriminator_variable_strategy, 
                            discriminator_update_ops, 
                            discriminator_learning_rate_ph, 
                            **kwargs):

    with tf.variable_scope("discriminator"):
        discriminator_tower_losses = tf.unstack(discriminator_tower_losses_ph, 2*effective_batch_size)

        discriminator_tower_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="nn/GAN/Discr")
        discriminator_tower_gradvars = []
        for tower_grad in kwargs['_tower_grads']:
            discriminator_tower_gradvars.append(zip(tower_grad, discriminator_tower_params))

        # Now compute global loss and gradients.
        discriminator_gradvars = []
        with tf.name_scope('gradient_averaging'):
            all_grads = {}
            for grad, var in itertools.chain(*discriminator_tower_gradvars):
                if grad is not None:
                    all_grads.setdefault(var, []).append(grad)
            for var, grads in six.iteritems(all_grads):
                # Average gradients on the same device as the variables
                # to which they apply.
                with tf.device(var.device):
                    if len(grads) == 1:
                        avg_grad = grads[0]
                    else:
                        avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))

                discriminator_gradvars.append((avg_grad, var))
                
        discriminator_global_step = tf.train.get_global_step()

        # Device that runs the ops to apply global gradient updates.
        consolidation_device = '/gpu:0' if discriminator_variable_strategy == 'GPU' else '/cpu:0'
        with tf.device(consolidation_device):

            discriminator_loss = tf.reduce_mean(discriminator_tower_losses, name='loss')
            discriminator_optimizer = tf.train.AdamOptimizer(discriminator_learning_rate_ph, 0.5)
            discriminator_optimizer = tf.contrib.estimator.clip_gradients_by_norm(discriminator_optimizer, 5.0)

        # Create single grouped train op
        discriminator_train_op = [
            discriminator_optimizer.apply_gradients(discriminator_gradvars, 
                                      global_step=discriminator_global_step)]

        discriminator_train_op.extend(discriminator_update_ops)
        discriminator_train_op = tf.group(*discriminator_train_op)

    return discriminator_train_op, discriminator_loss


def main(job_dir, data_dir, variable_strategy, num_gpus, log_device_placement, 
         num_intra_threads, **hparams):

    tf.reset_default_graph()

    temp = set(tf.all_variables())

    with open(discr_pred_file, 'a') as discr_pred_log:
        discr_pred_log.flush()
        with open(log_file, 'a') as log:
            log.flush()

            with open(val_log_file, 'a') as val_log:
                val_log.flush()

                # The env variable is on deprecation path, default is set to off.
                os.environ['TF_SYNC_ON_FINISH'] = '0'
                os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

                #with tf.device("/cpu:0"):
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
                with tf.control_dependencies(update_ops):

                    # Session configuration.
                    log_device_placement = False #Once placement is correct, this fills up too much of the cmd window...
                    sess_config = tf.ConfigProto(
                        allow_soft_placement=True,
                        log_device_placement=log_device_placement,
                        intra_op_parallelism_threads=num_intra_threads,
                        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

                    config = RunConfig(
                        session_config=sess_config, model_dir=job_dir)
                    hparams=tf.contrib.training.HParams(
                        is_chief=config.is_chief,
                        **hparams)

                    img = input_fn(data_dir, 'train', batch_size, num_gpus)
                    img_val = input_fn(data_dir, 'val', batch_size, num_gpus)

                    with tf.Session(config=sess_config) as sess: #Alternative is tf.train.MonitoredTrainingSession()

                        print("Session started")

                        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                        #sess.run( tf.global_variables_initializer())
                        temp = set(tf.all_variables())

                        ____img = sess.run(img)
                        img_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img') 
                                  for i in ____img]
                        img_truth_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img_truth') 
                                        for i in ____img]

                        is_training = True
                        generator_model_fn = get_model_fn(
                            num_gpus, variable_strategy, num_workers, component="generator")
                        discriminator_model_fn = get_model_fn(
                            num_gpus, variable_strategy, num_workers, component="discriminator")

                        print("Dataflow established")

                        #########################################################################################

                        results = generator_model_fn(img_ph, mode=is_training, 
                                                     params=hparams)
                        _tower_preds = results[0]
                        _discr_preds = results[1]
                        update_ops = results[2]
                        tower_grads = results[3:(3+batch_size)]

                        learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')

                        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                        temp = set(tf.all_variables())

                        mini_batch_dict = {}
                        for i in range(batch_size):
                            _img = sess.run(img[i])
                            mini_batch_dict.update({img_ph[i]: _img})
                            mini_batch_dict.update({img_truth_ph[i]: _img})

                        gradvars_pry, preds = sess.run([tower_grads, _tower_preds], 
                                                       feed_dict=mini_batch_dict)
                        del mini_batch_dict

                        tower_grads_ph = [[tf.placeholder(tf.float32, shape=t.shape, name='tower_grads') 
                                           for t in gradvars_pry[0]] 
                                           for _ in range(effective_batch_size)]
                        del gradvars_pry
            
                        train_op = _train_op(variable_strategy, update_ops, 
                                                       learning_rate_ph,
                                                       _tower_grads=tower_grads_ph)

                        print("Generator flow established")

                        #########################################################################################

                        pred_ph = [tf.placeholder(tf.float32, shape=i.shape, name='prediction')
                                   for i in preds]
                        label_ph = [tf.placeholder(tf.float32, shape=(1), name='label')
                                    for _ in range(len(preds))]

                        #The closer the label is to 1., the more confident the discriminator is that the image is real
                        discriminator_results = discriminator_model_fn(
                            pred_ph, label_ph, mode=is_training, params=hparams)
                        _discriminator_tower_losses = discriminator_results[0]
                        _discriminator_tower_preds = discriminator_results[1]
                        _discriminator_update_ops = discriminator_results[2]
                        _discriminator_tower_grads = discriminator_results[3:(3+batch_size)]

                        discriminator_tower_losses_ph = tf.placeholder(
                            tf.float32, shape=(2*effective_batch_size,), 
                            name='discriminator_tower_losses')
                        discriminator_learning_rate_ph = tf.placeholder(
                            tf.float32, name='discriminator_learning_rate')

                        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                        temp = set(tf.all_variables())

                        mini_batch_dict = {}
                        for i in range(batch_size):
                            mini_batch_dict.update({ph: np.random.rand(cropsize, cropsize, 1) for ph in pred_ph})
                            mini_batch_dict.update({ph: val for ph, val in zip(pred_ph, preds)})
                            mini_batch_dict.update({ph: np.array([0.5]) for ph in label_ph})
                        discriminator_gradvars_pry = sess.run(_discriminator_tower_grads, feed_dict=mini_batch_dict)
                        del preds
                        del mini_batch_dict

                        discriminator_tower_grads_ph = [[
                            tf.placeholder(tf.float32, shape=t.shape, name='discriminator_tower_grads') 
                            for t in discriminator_gradvars_pry[0]] for _ in range(2*effective_batch_size)]
                        del discriminator_gradvars_pry
            
                        discriminator_train_op, discriminator_get_loss = _discriminator_train_op(
                            discriminator_tower_losses_ph,
                            variable_strategy,
                            _discriminator_update_ops,
                            discriminator_learning_rate_ph,
                            _tower_grads=discriminator_tower_grads_ph)

                        print("Discriminator flows established")

                        #########################################################################################

                        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                        train_writer = tf.summary.FileWriter( logDir, sess.graph )

                        #print(tf.all_variables())
                        saver = tf.train.Saver()
                        #saver.restore(sess, tf.train.latest_checkpoint("X:/Jeffrey-Ede/models/gan-unsupervised-latent-3/model/"))

                        learning_rate = initial_learning_rate
                        discriminator_learning_rate = initial_discriminator_learning_rate

                        offset = 30
                        train_gen = False
                        avg_pred = 0. #To decide which to train
                        avg_pred_real = 0.
                        num_since_change = 0.
 
                        counter = 0
                        counter_init = counter+1
                        pred_avg = pred_avg_real = 0.5
                        while True:
                            #Train for a couple of hours
                            time0 = time.time()
                    
                            with open(model_dir+"learning_rate.txt") as lrf:
                                try:
                                    learning_rates = [float(line) for line in lrf]
                                    learning_rate = np.float32(learning_rates[0])
                                    discriminator_learning_rate = np.float32(learning_rate[1])
                                    print("Using learning rates: {}, {}".format(
                                        learning_rate, discriminator_learning_rate))
                                except:
                                    pass

                            generations = []
                            while time.time()-time0 < modelSavePeriod:
                                counter += 1

                                #Apply the generator
                                tower_preds_list = []
                                tower_ground_truths_list = []
                                if train_gen:
                                    discriminator_tower_preds_list = []
                                ph_dict = {}
                                for j in range(increase_batch_size_by_factor):

                                    mini_batch_dict = {}
                                    __img = sess.run(img)
                                    tower_ground_truths_list += __img

                                    for i in range(batch_size):
                                        mini_batch_dict.update({img_ph[i]: __img[i]})
                                        mini_batch_dict.update({img_truth_ph[i]: __img[i]})

                                        ph_dict.update({img_ph[i]: __img[i],
                                                        img_truth_ph[i]: __img[i]})

                                        if train_gen:
                                            mini_batch_results = sess.run([_tower_preds] +
                                                                           tower_grads + 
                                                                           [_discr_preds], 
                                                                           feed_dict=mini_batch_dict)
                                        else:
                                            mini_batch_results = sess.run([_tower_preds] +
                                                                           tower_grads, 
                                                                           feed_dict=mini_batch_dict)

                                    tower_preds_list += [x for x in mini_batch_results[0]]

                                    for i in range(1, 1+batch_size):
                                        ph_dict.update({ph: val for ph, val in 
                                                        zip(tower_grads_ph[j], 
                                                            mini_batch_results[i])})
                                    if train_gen:
                                        discriminator_tower_preds_list += [x for x in mini_batch_results[1+batch_size]]
                                del mini_batch_dict

                                #Save outputs occasionally
                                if counter <= 1 or not counter % save_result_every_n_batches or (counter < 10000 and not counter % 1000) or counter == counter_init:
                                    try:
                                        save_input_loc = model_dir+"input-"+str(counter)+".tif"
                                        save_output_loc = model_dir+"output-"+str(counter)+".tif"
                                        Image.fromarray(scale0to1(__img[0]).reshape(cropsize, cropsize).astype(np.float32)).save( save_input_loc )
                                        Image.fromarray(scale0to1(tower_preds_list[0]).reshape(cropsize, cropsize).astype(np.float32)).save( save_output_loc )
                                    except:
                                        print("Image save failed")

                                del mini_batch_results

                                ph_dict.update({learning_rate_ph: learning_rate})

                                #Train the generator
                                if train_gen:
                                    sess.run(train_op, feed_dict=ph_dict)
                                    del ph_dict
                                if not train_gen:
                                    discriminator_tower_losses_list = []
                                    discriminator_tower_preds_list = []
                                    discriminator_ph_dict = {}

                                    #Apply the discriminator to the generated images
                                    for j in range(increase_batch_size_by_factor):
                                        mini_batch_dict = {}

                                        for i in range(batch_size):
                                            mini_batch_dict.update({pred_ph[i]: tower_preds_list[batch_size*j+i]})
                                            label = 0. if np.random.rand() > 1.-1./(1.+np.exp(-flip_weight*pred_avg)) else 0.85+0.15*np.random.rand()
                                            mini_batch_dict.update({label_ph[i]: np.reshape(label, (1,))})
                                            discriminator_ph_dict.update({pred_ph[i]: tower_preds_list[batch_size*j+i]})

                                        mini_batch_results = sess.run([_discriminator_tower_losses, 
                                                                       _discriminator_tower_preds] +
                                                                       _discriminator_tower_grads,
                                                                       feed_dict=mini_batch_dict)

                                        discriminator_tower_losses_list += [x for x in mini_batch_results[0]]
                                        discriminator_tower_preds_list += [x for x in mini_batch_results[1]]

                                        for i in range(2, 2+batch_size):
                                            discriminator_ph_dict.update({ph: val for ph, val in 
                                                                          zip(discriminator_tower_grads_ph[batch_size*j+i-2], 
                                                                              mini_batch_results[i])})

                                    del tower_preds_list
                                    del mini_batch_dict

                                    #print(mini_batch_results[2][0])

                                    #Apply the discriminator to real images
                                    for j in range(effective_batch_size):
                                        mini_batch_dict = {}
                                        for i in range(batch_size):
                                            mini_batch_dict.update({pred_ph[i]: np.reshape(tower_ground_truths_list[batch_size*j+i],
                                                                                           (cropsize, cropsize, 1))})
                                            label = 0.85+0.15*np.random.rand() if np.random.rand() > 1.-1./(1.+np.exp(-flip_weight*pred_avg_real)) else 0.
                                            mini_batch_dict.update({label_ph[i]: np.reshape(label, (1,))})

                                        mini_batch_results = sess.run([_discriminator_tower_losses, 
                                                                       _discriminator_tower_preds] +
                                                                       _discriminator_tower_grads,
                                                                       feed_dict=mini_batch_dict)

                                        discriminator_tower_losses_list += [x for x in mini_batch_results[0]]
                                        discriminator_tower_preds_list += [x for x in mini_batch_results[1]]

                                        for i in range(2, 2+batch_size):
                                            discriminator_ph_dict.update({ph: val for ph, val in 
                                                                          zip(discriminator_tower_grads_ph[
                                                                              batch_size*j+i-2+effective_batch_size], 
                                                                              mini_batch_results[i])})
                                    del mini_batch_dict

                                    discriminator_ph_dict.update({discriminator_learning_rate_ph: discriminator_learning_rate})

                                    discrimination_losses = np.reshape(np.asarray(discriminator_tower_losses_list),
                                                                       (2*effective_batch_size,))
                                    discriminator_ph_dict.update({discriminator_tower_losses_ph: discrimination_losses})

                                    gen_losses = [np.max([np.reshape(x, (1,)), 0]) 
                                                  for x in discriminator_tower_losses_list[:effective_batch_size]]

                                    generation_losses = np.reshape(np.asarray(gen_losses), (effective_batch_size,))

                                    _, discr_loss = sess.run([discriminator_train_op, discriminator_get_loss], 
                                                             feed_dict=discriminator_ph_dict)

                                    del discriminator_ph_dict

                                    gen_loss = np.mean(generation_losses)
                                    discr_loss = np.mean(discrimination_losses)

                                    try:
                                        log.write("Iter: {}, Gen loss: {:.8f}, Discr loss: {:.8f}".format(
                                            counter, gen_loss, discr_loss))
                                    except:
                                        print("Failed to write to log")

                                    print("Iter: {}, Gen loss: {:.8f}, Discr loss: {:.8f}".format(
                                          counter, gen_loss, discr_loss))

                                avg_pred += np.sum(np.asarray(discriminator_tower_preds_list[:effective_batch_size]))
                                avg_pred_real += np.sum(np.asarray(discriminator_tower_preds_list[effective_batch_size:]))

                                print("Iter: {}, Preds: {}".format(counter, discriminator_tower_preds_list[:effective_batch_size]))
                                discr_pred_log.write("Iter: {}, ".format(counter))
                                for p in discriminator_tower_preds_list:
                                    discr_pred_log.write("{}, ".format(p))
                                discr_pred_log.write("\n")

                                if not counter % val_skip_n:

                                    tower_preds_list = []
                                    tower_ground_truths_list = []

                                    #Generate micrographs
                                    mini_batch_dict = {}
                                    for i in range(batch_size):
                                        __img = sess.run(img_val[i])
                                        tower_ground_truths_list.append(__img)
                                        mini_batch_dict.update({img_ph[i]: __img})
                                        mini_batch_dict.update({img_truth_ph[i]: __img})

                                    mini_batch_results = sess.run([_tower_preds] +
                                                                   tower_grads, 
                                                                   feed_dict=mini_batch_dict)

                                    tower_preds_list += [x for x in mini_batch_results[0]]

                                    del mini_batch_dict
                                    del mini_batch_results

                                    discriminator_tower_losses_list = []
                                    _discriminator_tower_preds_list = []

                                    #Apply discriminator to fake micrographs
                                    mini_batch_dict = {}
                                    j = 0
                                    for i in range(batch_size):
                                        mini_batch_dict.update({pred_ph[i]: np.reshape(tower_ground_truths_list[batch_size*j+i],
                                                                                       (cropsize, cropsize, 1))})
                                        mini_batch_dict.update({label_ph[i]: np.array([0.], dtype=np.float32)})

                                    mini_batch_results = sess.run([_discriminator_tower_losses, 
                                                                   _discriminator_tower_preds],
                                                                   feed_dict=mini_batch_dict)

                                    discriminator_tower_losses_list += [x for x in mini_batch_results[0]]
                                    _discriminator_tower_preds_list += [x for x in mini_batch_results[1]]

                                    #Apply discriminator to real micrographs
                                    mini_batch_dict = {}
                                    for i in range(batch_size):
                                        mini_batch_dict.update({pred_ph[i]: np.reshape(tower_ground_truths_list[batch_size*j+i],
                                                                                       (cropsize, cropsize, 1))})
                                        mini_batch_dict.update({label_ph[i]: np.array([1.], dtype=np.float32)})

                                    mini_batch_results = sess.run([_discriminator_tower_losses, 
                                                                   _discriminator_tower_preds],
                                                                   feed_dict=mini_batch_dict)

                                    del mini_batch_dict

                                    discriminator_tower_losses_list += [x for x in mini_batch_results[0]]
                                    _discriminator_tower_preds_list += [x for x in mini_batch_results[1]]

                                    discr_loss_val = np.mean(np.asarray(discriminator_tower_losses_list))
                                    gen_loss_val = np.mean(np.asarray(discriminator_tower_losses_list[:batch_size]))

                                    try:
                                        val_log.write("Iter: {}, Gen Val: {:.8f}, Disc Val {:.8f}".format(
                                            counter, gen_loss_val, discr_loss_val))
                                    except:
                                        print("Failed to write to val log")

                                    print("Iter: {}, Gen loss: {:.8f}, Discr loss: {:.8f}, Gen Val: {:.8f}, Discr Val: {:.8f}".format(
                                          counter, gen_loss, discr_loss, gen_loss_val, discr_loss_val))

                                if not counter % trainee_switch_skip_n:
                                    avg_pred /= trainee_switch_skip_n*effective_batch_size

                                    print("Iter: {}, Training {}. Avg pred: {}".format(counter, "gen" if train_gen else "discr", avg_pred))

                                    pred_avg = 0.9*pred_avg + 0.1*avg_pred

                                    real_flip_prob = 0.
                                    if avg_pred_real:
                                        avg_pred_real /= trainee_switch_skip_n*effective_batch_size
                                        avg_pred_real = 1. - avg_pred_real
                                        pred_avg_real = 0.9*pred_avg_real + 0.1*avg_pred_real
                                        real_flip_prob = 1.-1./(1.+np.exp(-flip_weight*pred_avg_real))

                                    print("Gen pred: {}, Real Pred: {}".format(avg_pred, 1.-avg_pred_real))
                                    print("Iter: {}, Training {}, G flip: {}, D flip:{}".format(
                                        counter, "gen" if train_gen else "discr", 
                                        1.-1./(1.+np.exp(-flip_weight*pred_avg)),
                                        real_flip_prob))

                                    if num_since_change >= max_num_since_change:
                                        num_since_change = 1
                                        train_gen = not train_gen
                                    else:
                                        if avg_pred < 0.3:
                                            if train_gen:
                                                num_since_change += 1
                                            else:
                                                num_since_change = 0
                                            train_gen = True
                                        elif avg_pred > 0.7:
                                            if not train_gen:
                                                num_since_change += 1
                                            else:
                                                num_since_change = 0
                                            train_gen = False
                                        else:
                                            num_since_change = 0
                                            train_gen = not train_gen

                                    avg_pred = 0.
                                    avg_pred_real = 0.

                                #train_writer.add_summary(summary, counter)

                            #Save the model
                            saver.save(sess, save_path=model_dir+"model/", global_step=counter)
    return 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data-dir',
        type=str,
        default=data_dir,
        help='The directory where the CIFAR-10 input data is stored.')
    parser.add_argument(
        '--job-dir',
        type=str,
        default=model_dir,
        help='The directory where the model will be stored.')
    parser.add_argument(
        '--variable-strategy',
        choices=['CPU', 'GPU'],
        type=str,
        default='GPU',
        help='Where to locate variable operations')
    parser.add_argument(
        '--num-gpus',
        type=int,
        default=num_gpus,
        help='The number of gpus used. Uses only CPU if set to 0.')
    parser.add_argument(
        '--log-device-placement',
        action='store_true',
        default=True,
        help='Whether to log device placement.')
    parser.add_argument(
        '--num-intra-threads',
        type=int,
        default=0,
        help="""\
        Number of threads to use for intra-op parallelism. When training on CPU
        set to 0 to have the system pick the appropriate number or alternatively
        set it to the number of physical CPU cores.\
        """)
    parser.add_argument(
        '--train-steps',
        type=int,
        default=80000,
        help='The number of steps to use for training.')
    parser.add_argument(
        '--train-batch-size',
        type=int,
        default=batch_size,
        help='Batch size for training.')
    parser.add_argument(
        '--eval-batch-size',
        type=int,
        default=batch_size,
        help='Batch size for validation.')
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='Momentum for MomentumOptimizer.')
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.1,
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.\
        """)
    parser.add_argument(
        '--sync',
        action='store_true',
        default=False,
        help="""\
        If present when running in a distributed environment will run on sync mode.\
        """)
    parser.add_argument(
        '--num-inter-threads',
        type=int,
        default=0,
        help="""\
        Number of threads to use for inter-op parallelism. If set to 0, the
        system will pick an appropriate number.\
        """)
    parser.add_argument(
        '--data-format',
        type=str,
        default="NHWC",
        help="""\
        If not set, the data format best for the training device is used. 
        Allowed values: channels_first (NCHW) channels_last (NHWC).\
        """)
    parser.add_argument(
        '--batch-norm-decay',
        type=float,
        default=0.997,
        help='Decay for batch norm.')
    parser.add_argument(
        '--batch-norm-epsilon',
        type=float,
        default=1e-5,
        help='Epsilon for batch norm.')
    args = parser.parse_args()

    if args.num_gpus > 0:
        assert tf.test.is_gpu_available(), "Requested GPUs but none found."
    if args.num_gpus < 0:
        raise ValueError(
            'Invalid GPU count: \"--num-gpus\" must be 0 or a positive integer.')
    if args.num_gpus == 0 and args.variable_strategy == 'GPU':
        raise ValueError('num-gpus=0, CPU must be used as parameter server. Set'
                         '--variable-strategy=CPU.')
    if args.num_gpus != 0 and args.train_batch_size % args.num_gpus != 0:
        raise ValueError('--train-batch-size must be multiple of --num-gpus.')
    if args.num_gpus != 0 and args.eval_batch_size % args.num_gpus != 0:
        raise ValueError('--eval-batch-size must be multiple of --num-gpus.')

    main(**vars(args))