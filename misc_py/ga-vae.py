from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

scale = 1 #Make scale large to spped up initial testing

inner_hidden_nodes = 4096
inner_enc_nodes = 2048

data_dir = "F:/ARM_scans-crops/"
#data_dir = "E:/stills_hq-mini/"

modelSavePeriod = 2. #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/ga-vae/"

shuffle_buffer_size = 5000
num_parallel_calls = 4
num_parallel_readers = 4
prefetch_buffer_size = 5
batch_size = 1
num_gpus = 1

#batch_size = 8 #Batch size to use during training
num_epochs = 1000000 #Dataset repeats indefinitely

logDir = "C:/dump/train/"
log_file = model_dir+"log.txt"
val_log_file = model_dir+"val_log.txt"
discr_pred_file = model_dir+"discr_pred.txt"
cumProbs = np.array([]) #Indices of the distribution plus 1 will be correspond to means

cropsize = 256 #Sidelength of images to feed the neural network
channels = 1 #Greyscale input image

save_result_every_n_batches = 25000

val_skip_n = 10
trainee_switch_skip_n = 1

disp_select = False #Display selelected pixels upon startup

def spectral_norm(w, iteration=1, count=0):
   w0 = w
   w_shape = w.shape.as_list()
   w = tf.reshape(w, [-1, w_shape[-1]])

   u = tf.get_variable("u"+str(count), 
                       [1, w_shape[-1]], 
                       initializer=tf.random_normal_initializer(mean=0.,stddev=0.05), 
                       trainable=False)

   u_hat = u
   v_hat = None
   for i in range(iteration):
       """
       power iteration
       Usually iteration = 1 will be enough
       """
       v_ = tf.matmul(u_hat, tf.transpose(w))
       v_hat = tf.nn.l2_normalize(v_)

       u_ = tf.matmul(v_hat, w)
       u_hat = tf.nn.l2_normalize(u_)

   u_hat = tf.stop_gradient(u_hat)
   v_hat = tf.stop_gradient(v_hat)

   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

   with tf.control_dependencies([u.assign(u_hat)]):
       w_norm = w / sigma
       w_norm = tf.reshape(w_norm, w_shape)

   return w_norm


def generator_architecture(inputs, batch_norm_decay, params=None,train_batch_norm=None):
    """VAE-GAN in VAE-GAN"""

    phase = True
    concat_axis = 3

    with tf.variable_scope("GAN/Gen"):

        def int_shape(x):
            return list(map(int, x.get_shape()))

        def _instance_norm(net, train=phase):
            batch, rows, cols, channels = [i.value for i in net.get_shape()]
            var_shape = [channels]
            mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
            shift = tf.Variable(tf.zeros(var_shape), trainable=False)
            scale = tf.Variable(tf.ones(var_shape), trainable=False)
            epsilon = 1.e-3
            normalized = (net-mu)/(sigma_sq + epsilon)**(.5)
            return scale * normalized + shift

        def instance_then_activ(input):
            batch_then_activ = _instance_norm(input)
            batch_then_activ = tf.nn.leaky_relu(batch_then_activ)
            return batch_then_activ

        def pad(tensor, size):
            d1_pad = size[0]
            d2_pad = size[1]

            paddings = tf.constant([[0, 0], [d1_pad, d1_pad], [d2_pad, d2_pad], [0, 0]], dtype=tf.int32)
            padded = tf.pad(tensor, paddings, mode="REFLECT")
            return padded

        def periodic_pad(tensor, size):
            """
            Pads tensor cyclically along theta axis so it is a periodic function.
            Zero padded along r-axis
            TODO: implement this
            """

            padded = tf.concat([tensor, tf.zeros([4*batch_size,cropsize,size,channels])], axis=2)
            padded = tf.concat([tf.zeros([4*batch_size,cropsize,size,channels]), tensor], axis=2)

            padded = tf.concat([tensor, tensor[:, :, 0:size, :]], axis=1)
            padded = tf.concat([tensor[:, :, (cropsize-size-1):cropsize, :], tensor], axis=1)

            return padded

        ##Reusable blocks
        def _batch_norm_fn(input):
            batch_norm = tf.contrib.layers.batch_norm(
                input,
                epsilon=0.001,
                decay=batch_norm_decay,
                center=True, scale=True,
                is_training=True,
                fused=True,
                zero_debias_moving_mean=False,
                renorm=False)
            return batch_norm

        def prelu(_x, scope=None):
            """parametric ReLU activation"""
            with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
                _alpha = tf.get_variable("prelu", shape=_x.get_shape()[-1],
                                         dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
                return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)

        def batch_then_activ(input):
            _batch_then_activ = _batch_norm_fn(input)
            _batch_then_activ = tf.nn.leaky_relu(_batch_then_activ, alpha=0.01)#_batch_then_activ)
            return _batch_then_activ

        def conv_block_not_sep(input, filters, kernel_size=3, phase=phase, pad_size=None,
                               batch_plus_activ=True):

            if pad_size:
                input = pad(input, pad_size)

            conv_block = slim.conv2d(
                inputs=input,
                num_outputs=filters,
                kernel_size=kernel_size,
                activation_fn=None,
                padding='SAME' if not pad_size else 'VALID')

            if batch_plus_activ:
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
                               extra_batch_norm=True, kernel_size=3, pad_size=None,
                               batch_plus_activ=True, shape=None):
            if pad_size:
                strided_conv = pad(input, pad_size)
            else:
                strided_conv = input

            if batch_plus_activ:
                strided_conv = _batch_norm_fn(strided_conv)

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

            if batch_plus_activ:
                strided_conv = tf.nn.leaky_relu(strided_conv, 0.01)

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

        def deconv_block(input, filters, kernel_size=3, stride=2):
            '''Transpositionally convolute a feature space to upsample it'''

            deconv_block = slim.conv2d_transpose(
                inputs=input,
                num_outputs=filters,
                kernel_size=kernel_size,
                stride=stride,
                padding="same",
                activation_fn=None)
            deconv_block = batch_then_activ(deconv_block)

            return deconv_block

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

        def aspp(input, aspp_filters, aspp_output, phase=True):
            """
            Atrous spatial pyramid pooling
            phase defaults to true, meaning that the network is being trained
            """

            aspp_rateSmall = 3
            aspp_rateMedium = 6
            aspp_rateLarge = 9

            ##Convolutions at multiple rates
            conv1x1 = _batch_norm_fn(input)
            conv1x1 = slim.conv2d(inputs=conv1x1,
                num_outputs=aspp_filters,
                kernel_size=1,
                activation_fn=None,
                padding="same")
            conv1x1 = tf.nn.leaky_relu(conv1x1, 0.01)

            conv3x3_rateSmall = strided_conv_block(input=input,
                                         filters=aspp_filters,
                                         stride=1,
                                         rate=aspp_rateSmall)
            conv3x3_rateSmall = _batch_norm_fn(conv3x3_rateSmall)

            conv3x3_rateMedium = strided_conv_block(input=input,
                                         filters=aspp_filters,
                                         stride=1,
                                         rate=aspp_rateMedium)
            conv3x3_rateMedium = _batch_norm_fn(conv3x3_rateMedium)

            conv3x3_rateLarge = strided_conv_block(input=input,
                                         filters=aspp_filters,
                                         stride=1,
                                         rate=aspp_rateLarge)
            conv3x3_rateLarge = _batch_norm_fn(conv3x3_rateLarge)

            #Image-level features
            shape = int_shape(input)

            pooling = _batch_norm_fn(input)
            pooling = tf.image.resize_images(pooling, (1,1))
            pooling = slim.conv2d(inputs=pooling,
                num_outputs=aspp_filters,
                kernel_size=1,
                activation_fn=None,
                padding="same")
            pooling = _batch_norm_fn(pooling)
            pooling = tf.image.resize_images(pooling, shape[1:3])

            #Concatenate the atrous and image-level pooling features
            concatenation = tf.concat(
                values=[conv1x1, conv3x3_rateSmall, conv3x3_rateMedium, conv3x3_rateLarge, pooling],
                axis=concat_axis)

            #Reduce the number of channels
            concatenation = _batch_norm_fn(concatenation)
            reduced = slim.conv2d(
                inputs=concatenation,
                num_outputs=aspp_output,
                kernel_size=1,
                activation_fn=None,
                padding="SAME")
            reduced = tf.nn.leaky_relu(reduced, 0.01)

            return reduced

        def xception_block(x, features, features_end=None):

            x1 = strided_conv_block(x, features_end if features_end else features, stride=2, kernel_size=1)

            x = strided_conv_block(x, features, stride=1, kernel_size=3)
            x = strided_conv_block(x, features, stride=1, kernel_size=3)
            x = strided_conv_block(x, features_end if features_end else features, stride=2, kernel_size=3)

            return x + x1

        latent_depth = 16
        output_stride = 16
        num_layers = 7-1

        def vaegan_in_vaegan(img):
            """VAE-GAN in VAE-GAN architecture"""

            with tf.variable_scope("vaegan_in_vaegan"):

                def outer_enc(img1, reuse=False):
                    """Encode to sematic latent space with Xception+ASPP"""
                    with tf.variable_scope("Enc", reuse=reuse):

                        print("outer enc", tf.get_variable_scope().name)

                        features = [32*2**i for i in range(6)]
                        x = strided_conv_block(img1, features[0], stride=2, kernel_size=3)
                        x = strided_conv_block(x, features[1], stride=1, kernel_size=3)

                        x = xception_block(x, features[2])
                        x = xception_block(x, features[3])
                        x = xception_block(x, features[4])

                        for _ in range(8):
                            x = xception_middle_block(x, features[4])

                        x = aspp(x, features[4], features[4])
                        aspp_enc = x

                        #Learn means and log(sigma**2) of multinormal priors
                        mu_enc = strided_conv_block(x, latent_depth, stride=1, kernel_size=3, 
                                                    extra_batch_norm=False, batch_plus_activ=False)
                        log_sigma2_enc = strided_conv_block(x, latent_depth, stride=1, kernel_size=3,
                                                            extra_batch_norm=False, batch_plus_activ=False)
                        sigma = tf.exp(log_sigma2_enc/2)

                        #Standard multinormal distribution for reference when calculating Kullback-Leibler divergence
                        multinormal = tf.distributions.Normal(loc=mu_enc, scale=sigma)

                        return aspp_enc, multinormal


                def inner_enc(x, reuse=False):
                    with tf.variable_scope("Inner/Enc", reuse=reuse):

                        print("inner enc", tf.get_variable_scope().name)

                        x = tf.contrib.layers.flatten(x)
                        x = tf.contrib.layers.fully_connected(x, num_outputs=inner_hidden_nodes)

                        #Learn means and log(sigma**2) of multinormal priors
                        inner_mu_enc = tf.contrib.layers.fully_connected(x, num_outputs=inner_enc_nodes)
                        inner_log_sigma2_enc = tf.contrib.layers.fully_connected(x, num_outputs=inner_enc_nodes)
                        sigma = tf.exp(inner_log_sigma2_enc/2)

                        multinormal = tf.distributions.Normal(loc=inner_mu_enc, scale=sigma)

                        return multinormal

                def inner_dec(x, reuse=False):
                    """Decode classifying nodes"""

                    with tf.variable_scope("Inner/Dec", reuse=reuse):

                        print("inner dec", tf.get_variable_scope().name)

                        #Spatial semantic volume
                        side = cropsize//output_stride
                        latent_volume = int( side**2 * latent_depth )

                        x = tf.contrib.layers.fully_connected(x, num_outputs=latent_volume)
                        x = tf.reshape(x, [-1, side, side, latent_depth])

                        x0 = x
                        x = strided_conv_block(x, latent_depth, stride=1, kernel_size=3)
                        x = strided_conv_block(x, latent_depth, stride=1, kernel_size=3)
                        x = strided_conv_block(x, latent_depth, stride=1, kernel_size=3)
                        x += x0

                        return x


                def dec(x, reuse=False):

                    with tf.variable_scope("Dec", reuse=reuse):

                        print("outer dec", tf.get_variable_scope().name)

                        x = deconv_block(x, 256, kernel_size=3, stride=2)
                        x = deconv_block(x, 128, kernel_size=3, stride=2)
                        x = deconv_block(x, 64, kernel_size=3, stride=2)

                        for _ in range(3):
                            x = xception_middle_block(x, 64)

                        x = deconv_block(x, 32, kernel_size=3, stride=2)

                        x = slim.conv2d(
                                inputs=x,
                                num_outputs=1,
                                kernel_size=3,
                                padding="SAME",
                                activation_fn=None,
                                weights_initializer=None,
                                biases_initializer=None)

                    return x
                    
                def siamese_enc(input, reuse=True):
                    """For encoding decodings"""

                    _, x = outer_enc(input, reuse=reuse)
                    x = inner_enc(x.mean(), reuse=reuse)

                    return x

                def svm(x):
                    """Support vector machine"""
                    return x

                #Outer encode
                _, multinormal = outer_enc(img)

                #Straighthrough outer
                direct_output = dec(multinormal.mean(), reuse=False)
                #Prior outer
                prior_output = dec(multinormal.sample(), reuse=True)

                #Prepare inner encoding
                inner_multinormal = inner_enc(multinormal.mean(), reuse=False)

                #Straighthrough inner
                inner_direct_output = inner_dec(inner_multinormal.mean(), reuse=False)
                inner_direct_output = dec(inner_direct_output, reuse=True)

                #Prior inner
                inner_prior_output = inner_dec(inner_multinormal.sample(), reuse=True)
                inner_prior_output = dec(inner_prior_output, reuse=True)

                outputs_dict = { 'direct_output': direct_output,
                                    'multinormal': multinormal,
                                    'prior_output': prior_output,
                                    'inner_direct_output': inner_direct_output,
                                    'inner_multinormal': inner_multinormal,
                                    'inner_prior_output': inner_prior_output,
                                    'siamese_enc': siamese_enc,
                                    'svm': svm }

                return outputs_dict

        img = tf.reshape(inputs, [4*batch_size, cropsize, cropsize, channels])
        outputs = vaegan_in_vaegan(img)

        return outputs


def discriminator_architecture(inputs, second_input=None,
                               phase=False, params=None, gen_loss=0., reuse=False):
    """Three discriminators to discriminate between two data discributions"""

    with tf.variable_scope("GAN/Discr", reuse=reuse):

        def int_shape(x):
            return list(map(int, x.get_shape()))

        #phase = mode == tf.estimator.ModeKeys.TRAIN #phase is true during training
        concat_axis = 3

        def _instance_norm(net, train=phase):
            batch, rows, cols, channels = [i.value for i in net.get_shape()]
            var_shape = [channels]
            mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
            shift = tf.Variable(tf.zeros(var_shape), trainable=False)
            scale = tf.Variable(tf.ones(var_shape), trainable=False)
            epsilon = 1.e-3
            normalized = (net - mu) / (sigma_sq + epsilon)**(.5)
            return scale*normalized + shift

        def instance_then_activ(input):
            batch_then_activ = _instance_norm(input)
            batch_then_activ = tf.nn.relu(batch_then_activ)
            return batch_then_activ

        ##Reusable blocks
        def _batch_norm_fn(input):
            batch_norm = tf.contrib.layers.batch_norm(
                input,
                epsilon=0.001,
                decay=0.999,
                center=True, 
                scale=True,
                is_training=phase,
                fused=True,
                zero_debias_moving_mean=False,
                renorm=False)
            return batch_norm

        def batch_then_activ(input): #Changed to instance norm for stability
            batch_then_activ = input#_instance_norm(input)
            batch_then_activ = tf.nn.leaky_relu(batch_then_activ, alpha=0.2)
            return batch_then_activ

        def conv_block_not_sep(input, filters, kernel_size=3, phase=phase, batch_and_activ=True):
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

            if batch_and_activ:
                conv_block = batch_then_activ(conv_block)

            return conv_block

        def conv_block(input, filters, phase=phase):
            """
            Convolution -> batch normalisation -> leaky relu
            phase defaults to true, meaning that the network is being trained
            """

            conv_block = strided_conv_block(input, filters, 1, 1)

            return conv_block

        count = 0
        def strided_conv_block(input, filters, stride, rate=1, phase=phase, 
                               extra_batch_norm=False, kernel_size=3):
        
            nonlocal count 
            count += 1
            w = tf.get_variable("kernel"+str(count), shape=[kernel_size, kernel_size, input.get_shape()[-1], filters])
            b = tf.get_variable("bias"+str(count), [channels], initializer=tf.constant_initializer(0.0))

            x = tf.nn.conv2d(input=input, filter=spectral_norm(w, count=count), 
                             strides=[1, stride, stride, 1], padding='SAME') + b

            x = batch_then_activ(x)

            return x

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

        def max_pool(input, size=2, stride=2):

            pool = tf.contrib.layers.max_pool2d(inputs=input,
                                                kernel_size=size,
                                                stride=stride,
                                                padding='SAME')
            return pool

        features1 = 64
        features2 = 128
        features3 = 256
        features4 = 512

        '''Model building'''        
        with tf.variable_scope("small", reuse=reuse) as small_scope:
            small = inputs[0]
            small = strided_conv_block(small, features1, 2, 1, kernel_size=4)
            small = strided_conv_block(small, features2, 2, 1, kernel_size=4)
            small = strided_conv_block(small, features3, 2, 1, kernel_size=3)
            small = strided_conv_block(small, features3, 1, 1, kernel_size=3)
            small = strided_conv_block(small, features4, 2, 1, kernel_size=3)
            small = strided_conv_block(small, features4, 1, 1, kernel_size=3)
            shape = int_shape(small)
            small = tf.reshape(small, (-1, shape[1]*shape[2]*shape[3]))
            small = tf.contrib.layers.fully_connected(inputs=small,
                                                      num_outputs=1,
                                                      activation_fn=None,
                                                      biases_initializer=None)

        with tf.variable_scope("medium", reuse=reuse) as medium_scope:
            medium = inputs[1]
            medium = strided_conv_block(medium, features1, 2, 1, kernel_size=4)
            medium = strided_conv_block(medium, features2, 2, 1, kernel_size=4)
            medium = strided_conv_block(medium, features3, 2, 1, kernel_size=3)
            medium = strided_conv_block(medium, features3, 1, 1, kernel_size=3)
            medium = strided_conv_block(medium, features4, 2, 1, kernel_size=3)
            medium = strided_conv_block(medium, features4, 1, 1, kernel_size=3)
            shape = int_shape(medium)
            medium = tf.reshape(medium, (-1, shape[1]*shape[2]*shape[3]))
            medium = tf.contrib.layers.fully_connected(inputs=medium,
                                                       num_outputs=1,
                                                       activation_fn=None,
                                                       biases_initializer=None)

        with tf.variable_scope("large", reuse=reuse) as large_scope:
            large = inputs[2]
            large = strided_conv_block(large, features1, 2, 1, kernel_size=4)
            large = strided_conv_block(large, features2, 2, 1, kernel_size=4)
            large = strided_conv_block(large, features3, 2, 1, kernel_size=3)
            large = strided_conv_block(large, features3, 1, 1, kernel_size=3)
            large = strided_conv_block(large, features4, 2, 1, kernel_size=3)
            large = strided_conv_block(large, features4, 1, 1, kernel_size=3)
            shape = int_shape(large)
            large = tf.reshape(large, (-1, shape[1]*shape[2]*shape[3]))
            large = tf.contrib.layers.fully_connected(inputs=large,
                                                      num_outputs=1,
                                                      activation_fn=None,
                                                      biases_initializer=None)

        small = tf.exp(small)
        medium = tf.exp(medium)
        large = tf.exp(large)

    return [small, medium, large]


def get_multiscale_crops(input, multiscale_channels=1):

    def pad(tensor, size):
        d1_pad = size[0]
        d2_pad = size[1]

        paddings = tf.constant([[0, 0], [d1_pad, d1_pad], [d2_pad, d2_pad], [0, 0]], dtype=tf.int32)
        padded = tf.pad(tensor, paddings, mode="REFLECT")
        return padded

    input = pad(input, (35,35))#TODO

    small = tf.random_crop(
                input,
                size=(4*batch_size, 70, 70, multiscale_channels))
    small = tf.image.resize_images(small, (70, 70))
    medium = tf.random_crop(
                input,
                size=(4*batch_size, 225, 225, multiscale_channels))
    medium = tf.image.resize_images(medium, (70, 70))
    large = tf.random_crop(
                input,
                size=(4*batch_size, 256, 256, multiscale_channels))
    large = tf.image.resize_images(large, (70, 70))

    return small, medium, large


def experiment(input1, input2, trans1, trans2, outer_enc_lr, inner_enc_lr, outer_dec_lr, 
               inner_dec_lr, discr_lr, siamese_lr, outer_enc_beta1, inner_enc_beta1, 
               outer_dec_beta1, inner_dec_beta1, discr_beta1, siamese_beta1, batch_norm_decay, 
               train_batch_norm):

    #Weight loss contributions
    enc_l2_weight = 5.e-5
    dec_l2_weight = 5.e-5
    inner_enc_l2_weight = 5.e-5
    inner_dec_l2_weight = 5.e-5
    discr_l2_weight = 1.e-3
    wass_weight = 1.
    gradient_penalty_weight = 10.
    rot_invar_weight = 1.
    siamese_weight = 1.
    mse_weight = 1.

    minibatch = [input1, input2, trans1, trans2]
    inputs = tf.concat([tf.reshape(img, [-1, cropsize, cropsize, channels]) 
                        for img in minibatch], axis=0)
    
    ##Utility
    def gaussian_kernel(size: int,
                    mean: float,
                    std: float,
                   ):
        """Makes 2D gaussian Kernel for convolution."""

        d = tf.distributions.Normal(mean, std)

        vals = d.prob(tf.range(start = -size, limit = size + 1, dtype = tf.float32))

        gauss_kernel = tf.einsum('i,j->ij', vals, vals)

        return gauss_kernel / tf.reduce_sum(gauss_kernel)

    def blur(image):
        gauss_kernel = gaussian_kernel( 2, 0., 2.5 )

        # Expand dimensions of `gauss_kernel` for `tf.nn.conv2d` signature.
        gauss_kernel = gauss_kernel[:, :, tf.newaxis, tf.newaxis]

        # Convolve.
        return tf.nn.conv2d(image, gauss_kernel, strides=[1, 1, 1, 1], padding="VALID")
   
    #Variational autoencoding and decoding using VAE-GAN in VAE-GAN
    generation = generator_architecture(inputs, batch_norm_decay, train_batch_norm=train_batch_norm)
    direct_outputs = generation['direct_output']
    multinormal = generation['multinormal']
    prior_outputs = generation['prior_output']
    inner_direct_outputs = generation['inner_direct_output']
    inner_multinormal = generation['inner_multinormal']
    inner_prior_outputs = generation['inner_prior_output']
    siamese_architecture = generation['siamese_enc']

    #Convenience aliases
    mu_encs = multinormal.mean()
    log_sigma2 =  2*tf.log(multinormal.stddev())
    
    inner_mu_encs = inner_multinormal.mean()
    inner_log_sigma2 = 2*tf.log(inner_multinormal.stddev())

    #Discrimination
    def apply_discriminators(output, truth, prior_output, inner_output, inner_prior_output, reuse=False):

        #Create multiscale crops to feed to discriminators
        epsilon = tf.random_uniform(
                    shape=[1, 1, 1, 1],
				    minval=0.,
				    maxval=1.)
        X_hat = (1-epsilon)*truth + epsilon*output

        #Concatenate images so they will all be cropped at the same place
        outputs = [output, truth, X_hat, prior_output, inner_output, inner_prior_output]
        concat = tf.concat(outputs, axis=3)

        #Crop
        shape = (4*batch_size, 70, 70, channels)
        num_channels = len(outputs)
        multiscale_crops = get_multiscale_crops(concat, multiscale_channels=num_channels)
        multiscale_crops = [tf.unstack(crop, axis=3) for crop in multiscale_crops]

        #Sort crops into categories
        crops_set = []
        for crops in multiscale_crops:
            crops_set.append( [tf.reshape(unstacked, shape) for unstacked in crops] )

        #Concatenate so the crops can be processed as a single batch
        multiscale = []
        for crops in crops_set:
            multiscale.append( tf.concat(crops, axis=0) )

        #Apply discriminators
        _discrimination = discriminator_architecture( multiscale )

        step = 4*batch_size
        return {'discrimination': [d[0:step] for d in _discrimination],
                'discrimination_natural': [d[step:2*step] for d in _discrimination],
                'discrimination_xhat': [d[2*step:3*step] for d in _discrimination],
                'discrimination_prior': [d[3*step:4*step] for d in _discrimination],
                'discrimination_inner': [d[4*step:5*step] for d in _discrimination],
                'discrimination_inner_prior': [d[5*step:6*step] for d in _discrimination],
                'multiscale_xhat': [m[2] for m in crops_set]} #X_hat is at index 2 in the cropping list

    def siamese_assessor(enc1, enc2, same=False, reuse=False):
        """Use support vector machine to calculate siamese loss"""
        with tf.variable_scope("siamese_assessor", reuse=reuse):
            
            start1 = tf.distributions.Normal(loc=enc1.mean()[:2], scale=enc1.stddev()[:2])
            end1 = tf.distributions.Normal(loc=enc1.mean()[2:4], scale=enc1.stddev()[2:4])

            start2 = tf.distributions.Normal(loc=enc2.mean()[:2], scale=enc2.stddev()[:2])
            end2 = tf.distributions.Normal(loc=enc2.mean()[2:4], scale=enc2.stddev()[2:4])

            dist = ( tf.distributions.kl_divergence(start1, start2) + 
                     tf.distributions.kl_divergence(start1, end2) +
                     tf.distributions.kl_divergence(end1, start2) +
                     tf.distributions.kl_divergence(end1, end2) )

            if not same: 
                siamese_loss = tf.maximum(1.-dist, 0.)
            else:
                siamese_loss = dist

        return siamese_loss

    def apply_siamese(inputs_enc, direct_outputs, inner_outputs):

        #Get siamese encodings
        with tf.variable_scope("GAN/Gen/vaegan_in_vaegan"): #Same variable scope as rest of gen for variable reuse
            direct_outputs_enc = siamese_architecture(direct_outputs)
            inner_outputs_enc = siamese_architecture(inner_outputs)

        #Same combinations
        same_preds = siamese_assessor( inputs_enc, direct_outputs_enc, same=True )

        #Different combinations
        shift1_direct_outputs_enc = tf.distributions.Normal(
            loc=tf.concat( [direct_outputs_enc.mean()[1:], direct_outputs_enc.mean()[0:]], axis=0 ), 
            scale=tf.concat( [direct_outputs_enc.stddev()[1:], direct_outputs_enc.stddev()[0:]], axis=0 )
            )
        different_preds = siamese_assessor( inputs_enc, shift1_direct_outputs_enc, same=False, reuse=True )

        #Generated same combinations
        gen_same_preds = siamese_assessor( inputs_enc, inner_outputs_enc, same=True, reuse=True )

        #Generated from prior same combinationts
        shift1_inner_outputs_enc = tf.distributions.Normal(
            loc=tf.concat( [inner_outputs_enc.mean()[1:], inner_outputs_enc.mean()[0:]], axis=0 ), 
            scale=tf.concat( [inner_outputs_enc.stddev()[1:], inner_outputs_enc.stddev()[0:]], axis=0 )
            )
        gen_different_preds = siamese_assessor( inputs_enc, shift1_inner_outputs_enc, same=False, reuse=True )

        avg_same_pred = tf.reduce_mean( same_preds )
        avg_different_pred = tf.reduce_mean( different_preds )
        avg_gen_same_pred = tf.reduce_mean( gen_same_preds )
        avg_gen_different_pred = tf.reduce_mean( gen_different_preds )

        #Calculate losses
        inner_loss = tf.reduce_mean( gen_different_preds - gen_same_preds )

        siamese_loss = tf.reduce_mean( same_preds + tf.maximum(1. - different_preds, 0) )

        return {'avg_same_pred': avg_same_pred,
                'avg_different_pred': avg_different_pred,
                'avg_gen_same_pred': avg_gen_same_pred,
                'avg_gen_different_pred': avg_gen_different_pred,
                'siamese_loss': siamese_loss,
                'inner_loss': inner_loss,
                'direct_outputs_enc': direct_outputs_enc,
                'inner_outputs_enc': inner_outputs_enc}

    #Apply discriminators to all generations: direct and from prior for inner and outer VAE-GAN
    multiscale_xhat = []
    discrimination = []
    discrimination_natural = []
    discrimination_prior = []
    discrimination_xhat = []
    discrimination_inner = []
    discrimination_inner_prior = []

    discr_dict = apply_discriminators(direct_outputs, inputs, prior_outputs, 
                                      inner_direct_outputs, inner_prior_outputs)

    discrimination.append( discr_dict['discrimination'] )
    discrimination_natural.append( discr_dict['discrimination_natural'] )
    discrimination_prior.append( discr_dict['discrimination_prior'] )
    discrimination_xhat.append( discr_dict['discrimination_xhat'] )
    discrimination_inner.append( discr_dict['discrimination_inner'] )
    discrimination_inner_prior.append( discr_dict['discrimination_inner_prior'] )
    multiscale_xhat.append( discr_dict['multiscale_xhat'] )
        
    ##Discriminator losses
    def perform_discrimination(discr_model_params, idx, discriminations, discrimination_naturals, discrimination_priors,
                       discrimination_inners, discrimination_inner_priors, discrimination_xhats, 
                       multiscale_xhats, gradient_penalty_weight, discr_l2_weight):

        pred_real_t = 0.
        pred_fake_t = 0.
        pred_prior_t = 0.
        pred_inner_t = 0.
        pred_inner_prior_t = 0.
        avg_d_grads_t = 0.
        tower_loss = 0.
        loop_count = 0
        for d, d_natural, d_prior, d_inner, d_inner_prior, d_xhat, multiscale_xhat \
            in zip(discriminations, discrimination_naturals, discrimination_priors, discrimination_inners, 
                   discrimination_inner_priors, discrimination_xhats, multiscale_xhats):

            loop_count += 1

            #Wasserstein
            pred_fake = d[idx]
            pred_real = d_natural[idx]
            pred_prior = d_prior[idx]
            pred_inner = d_inner[idx]
            pred_inner_prior = d_inner_prior[idx]
            wasserstein_loss = tf.reduce_mean(pred_real +
                                              tf.maximum(1.-pred_fake, 0) +
                                              tf.maximum(1.-pred_prior, 0) + 
                                              tf.maximum(1.-pred_inner, 0) +
                                              tf.maximum(1.-pred_inner_prior, 0))

            #Running totals for predictions
            pred_real_t += pred_real
            pred_fake_t += pred_fake
            pred_prior_t += pred_prior
            pred_inner_t += pred_inner
            pred_inner_t += pred_inner_prior

            #Gradient penalty
            grad_D_X_hat = tf.gradients(d_xhat[idx], [multiscale_xhat[idx]])[0]
            red_idx = [i for i in range(2, multiscale_xhat[idx].shape.ndims)]
            slopes = tf.sqrt(1.e-8+tf.reduce_sum(tf.square(grad_D_X_hat), axis=red_idx))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            gradient_d_loss = gradient_penalty_weight * gradient_penalty

            avg_d_grads_t += gradient_penalty

            #L2
            if discr_l2_weight:
                discr_l2_loss = discr_l2_weight * tf.add_n([tf.nn.l2_loss(v) for v in discr_model_params])
            else:
                discr_l2_loss = 0.

            tower_loss += wass_weight*wasserstein_loss + gradient_d_loss

        discr_train_op = tf.train.AdamOptimizer(discr_lr, discr_beta1).minimize(
            tower_loss, var_list=discr_model_params)

        #Avg predictions
        avg_pred_fake = pred_fake_t/loop_count
        avg_pred_real = pred_real_t/loop_count
        avg_pred_prior = pred_prior_t/loop_count
        avg_pred_inner = pred_inner_t/loop_count
        avg_pred_inner_prior = pred_inner_prior_t/loop_count
        avg_d_grads = avg_d_grads_t/loop_count

        return {'discr_train_op': discr_train_op,
                'pred_fake': avg_pred_fake,
                'pred_real': avg_pred_real,
                'pred_prior': avg_pred_prior,
                'pred_inner': avg_pred_inner,
                'pred_inner_prior': avg_pred_inner_prior,
                'avg_d_grads': avg_d_grads,
                'total_loss': tower_loss}

    #Discriminator model parameters
    small_discr_model_params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Discr/small")
    medium_discr_model_params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Discr/medium")
    large_discr_model_params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Discr/large")

    small_discrimination = perform_discrimination(small_discr_model_params, 0, discrimination, discrimination_natural, 
                                          discrimination_prior, discrimination_inner, discrimination_inner_prior, 
                                          discrimination_xhat, multiscale_xhat, gradient_penalty_weight, discr_l2_weight)
    small_discr_train_op = small_discrimination['discr_train_op']

    medium_discrimination = perform_discrimination(medium_discr_model_params, 1, discrimination, discrimination_natural, 
                                           discrimination_prior, discrimination_inner, discrimination_inner_prior,
                                           discrimination_xhat, multiscale_xhat, gradient_penalty_weight, discr_l2_weight)
    medium_discr_train_op = medium_discrimination['discr_train_op']

    large_discrimination = perform_discrimination(large_discr_model_params, 2, discrimination, discrimination_natural, 
                                          discrimination_prior, discrimination_inner, discrimination_inner_prior,
                                          discrimination_xhat, multiscale_xhat, gradient_penalty_weight, discr_l2_weight)
    large_discr_train_op = large_discrimination['discr_train_op']

    multiscale_discrs = [small_discrimination, medium_discrimination, large_discrimination]

    def multiscale_discr_mean_stat(multiscale_discrs, stat_key):
        """Average statistics calulated by the multiscal discriminators"""

        mean_stat = multiscale_discrs[0][stat_key]
        for discr in multiscale_discrs[1:]:
            mean_stat += discr[stat_key]
        mean_stat /= len(multiscale_discrs)

        return mean_stat

    #Convenience aliases
    mean_discr = multiscale_discr_mean_stat(multiscale_discrs, 'pred_fake')
    mean_discr_natural = multiscale_discr_mean_stat(multiscale_discrs, 'pred_real')
    mean_discr_prior = multiscale_discr_mean_stat(multiscale_discrs, 'pred_prior')
    mean_discr_inner = multiscale_discr_mean_stat(multiscale_discrs, 'pred_inner')
    mean_discr_inner_prior = multiscale_discr_mean_stat(multiscale_discrs, 'pred_inner_prior')

    #Apply siamese architecture to outputs and encodings
    siamese_dict = apply_siamese( inner_multinormal, direct_outputs, inner_direct_outputs )

    avg_same_pred = siamese_dict['avg_same_pred']
    avg_different_pred = siamese_dict['avg_different_pred']
    avg_gen_same_pred = siamese_dict['avg_gen_same_pred']
    avg_gen_different_pred = siamese_dict['avg_gen_different_pred']
    siamese_loss = siamese_dict['siamese_loss']
    siamese_loss = siamese_dict['inner_loss']
    siamese_direct_outputs_enc = siamese_dict['direct_outputs_enc']
    siamese_inner_outputs_enc = siamese_dict['inner_outputs_enc']

    #Generator model parameters

    outer_enc_model_params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Gen/vaegan_in_vaegan/Enc")
    inner_enc_model_params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Gen/vaegan_in_vaegan/Inner/Enc")
    outer_dec_model_params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Gen/vaegan_in_vaegan/Dec")
    inner_dec_model_params = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Gen/vaegan_in_vaegan/Inner/Dec")

    #MSE guidance loss
    mse = 100*tf.losses.mean_squared_error( blur(inputs), blur(direct_outputs) )
    mse = 3.5*tf.cond( mse < 1, lambda: mse, lambda: tf.sqrt(mse+1.e-8) )
    mse = tf.reduce_mean(mse)

    #Measure similarity of transformed image to encoding
    partition = 4*batch_size//2
    rot_invar_kl_loss = 0.
    for i in range(partition):
        distribution1 = tf.distributions.Normal(
            loc=inner_multinormal.mean()[i], scale=inner_multinormal.stddev()[i])
        distribution2 = tf.distributions.Normal(
            loc=inner_multinormal.mean()[i+partition], 
            scale=inner_multinormal.stddev()[i+partition])
        rot_invar_kl_loss += tf.distributions.kl_divergence( distribution1, distribution2 )
    rot_invar_kl_loss /= partition #Mean KL divergence

    #L2 regularization
    l2_loss_dec = tf.add_n( [tf.nn.l2_loss(v) for v in outer_dec_model_params] )
    l2_loss_inner_dec = tf.add_n( [tf.nn.l2_loss(v) for v in inner_dec_model_params] )
    l2_loss_enc = tf.add_n( [tf.nn.l2_loss(v) for v in outer_enc_model_params] )
    l2_loss_inner_enc = tf.add_n( [tf.nn.l2_loss(v) for v in inner_enc_model_params] )

    l2_loss_small_discr = tf.add_n( [tf.nn.l2_loss(v) for v in small_discr_model_params] )
    l2_loss_medium_discr = tf.add_n( [tf.nn.l2_loss(v) for v in medium_discr_model_params] )
    l2_loss_large_discr = tf.add_n( [tf.nn.l2_loss(v) for v in large_discr_model_params] )

    ##Outer decoder loss and train op    
    loss = 0.

    #Adversarial losses
    loss += mean_discr
    loss -= mean_discr_natural
    loss += mean_discr_prior
    loss += mean_discr_inner
    loss += mean_discr_inner_prior

    #MSE guidance
    if mse_weight:
        loss += mse_weight*mse

    #L2 regularization
    if dec_l2_weight:
        loss += dec_l2_weight*l2_loss_dec

    #Siamese guidance
    if siamese_weight:
        loss += siamese_weight*siamese_loss

    #Train op
    outer_dec_train_op = tf.train.AdamOptimizer(outer_dec_lr, outer_dec_beta1).minimize(
        loss, var_list=outer_dec_model_params)

    ##Inner decoder loss
    loss = 0.

    #Adversarial losses
    loss -= mean_discr_natural
    loss += mean_discr_inner
    loss += mean_discr_inner_prior

    #Siamese guidance
    if siamese_weight:
        loss += siamese_weight*siamese_loss

    #L2 regularization
    if inner_dec_l2_weight:
        loss += inner_dec_l2_weight*l2_loss_inner_dec

    #Train op
    inner_dec_train_op = tf.train.AdamOptimizer(inner_dec_lr, inner_dec_beta1).minimize(
            loss, var_list=inner_dec_model_params)

    #Outer encoder loss
    loss = 0.

    #MSE guidance
    if mse_weight:
        loss += mse_weight*mse

    #Outer Kullback-Leibler divergence
    #Mean; rather than sum, to go to probability space
    outer_kl_loss = 1 * -0.5 * tf.reduce_mean( 1 + log_sigma2 - 
                                       mu_encs**2 - tf.exp(log_sigma2) )
    loss += outer_kl_loss

    if enc_l2_weight:
        loss += enc_l2_weight*l2_loss_dec

    #Siamese guidance
    if siamese_weight:
        loss += siamese_weight*siamese_loss

    #Train op
    outer_enc_train_op = tf.train.AdamOptimizer(outer_enc_lr, outer_enc_beta1).minimize(
            loss, var_list=outer_enc_model_params)

    ##Inner encoder loss
    loss = 0.

    #Inner Kullback-Leibler divergence
    #Mean; rather than sum, to go to probability space
    inner_kl_loss = 1 * -0.5 * tf.reduce_mean( 1 + inner_log_sigma2 - 
                                       inner_mu_encs**2 - tf.exp(inner_log_sigma2) )
    loss += inner_kl_loss

    if inner_enc_l2_weight:
        loss += inner_enc_l2_weight*l2_loss_inner_enc

    #Siamese guidance
    if siamese_weight:
        loss += siamese_weight*siamese_loss

    #Rotational invariance loss
    loss += rot_invar_weight + rot_invar_kl_loss

    #Train op
    inner_enc_train_op = tf.train.AdamOptimizer(inner_enc_lr, inner_enc_beta1).minimize(
            loss, var_list=inner_enc_model_params)


    ##Group ops together for convenience
    #Train ops
    train_ops = [outer_enc_train_op,
                 inner_enc_train_op,
                 outer_dec_train_op,
                 inner_dec_train_op,
                 small_discr_train_op,
                 medium_discr_train_op,
                 large_discr_train_op]

    #Example images
    full_output_ops = [direct_outputs, prior_outputs, inner_direct_outputs, inner_prior_outputs]
    output_ops = [x[0] for x in full_output_ops]
    output_ops_names =  ['direct', 'prior', 'inner_direct', 'inner_prior']

    #Statistics
    output_stats = [mean_discr_natural, 
                    mean_discr, 
                    mean_discr_prior, 
                    mean_discr_inner,
                    mean_discr_inner_prior,
                    outer_kl_loss,
                    inner_kl_loss,
                    rot_invar_kl_loss,
                    mse]
    output_stats_names = ['mean_discr_natural',
                          'mean_discr',
                          'mean_discr_prior',
                          'mean_discr_inner',
                          'mean_discr_inner_prior',
                          'outer_kl_loss',
                          'inner_kl_loss',
                          'rot_invar_kl_loss',
                          'mse']

    return { 'train_ops': train_ops, 
             'output_ops': output_ops,
             'output_ops_names': output_ops_names,
             'output_stats': output_stats,
             'output_stats_names': output_stats_names }


def random_polar_transform(img):

    #Rotate
    shift = np.random.randint(img.shape[0])
    if shift:
        img = np.concatenate( (img[:, shift:img.shape[0]],img[:, :shift] ), axis=1 )

    #Reverse phase
    if random.randint(0,1):
        img = np.flip(img, axis=1)
        
    return img

def polar2cart(r, theta, center):
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return x, y

def img2polar(img, center=None, initial_radius=None, final_radius=None, phase_width=None):

    if center is None:
        center = (img.shape[0]/2, img.shape[1]/2)

    if initial_radius is None:
        initial_radius = 0

    if final_radius is None:
        final_radius = int((min(img.shape)-1)/2)

    if phase_width is None:
        phase_width = min(img.shape)

    theta , R = np.meshgrid(np.linspace(0, 2*np.pi, phase_width*10), 
                            np.sqrt(np.linspace(initial_radius**2, final_radius**2, phase_width)))

    Xcart, Ycart = polar2cart(R, theta, center)

    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)

    polar_img = img[Ycart,Xcart]

    polar_img = cv2.resize(polar_img, (cropsize, cropsize), interpolation=cv2.INTER_AREA)

    return polar_img

def cart_to_polar_random_transform(img):
    return random_polar_transform(img2polar(img)).astype(np.float32)


def flip_rotate(img):
    """Applies a random flip || rotation to the image, possibly leaving it unchanged"""

    choice = np.random.randint(0, 8)
    
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


def load_image(addr, resize_size=cropsize, img_type=np.float32):
    """Read an image and make sure it is of the correct type. Optionally resize it"""
    
    try:
        img = imread(addr, mode='F')
    except:
        img = np.zeros((cropsize,cropsize))
        print("Image read failed")

    if resize_size:
        img = cv2.resize(img, (resize_size, resize_size), interpolation=cv2.INTER_AREA)

    return img.astype(img_type)

def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def norm_img(img):
    
    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.)
    else:
        a = 0.5*(min+max)
        b = 0.5*(max-min)

        img = (img-a) / b

    return img.astype(np.float32)

def preprocess(img):

    img[np.isnan(img)] = 0.
    img[np.isinf(img)] = 0.

    cart_to_polar_random_transform(img)

    img = norm_img(img)

    return img

def cutout_regularizer(img, side_prop=0.3, regularization_prob=0.5):
    """Apply cutout regularization to a proportion of images"""

    if np.random.rand() < regularization_prob: #Apply cutout regularization
        side = int(side_prop*min(img.shape))

        x, y = np.random.randint(high=img.shape[0]-side), np.random.randint(high=img.shape[1]-side)

        cutout_img = np.copy(img)
        cutout_img[x:(x+side), y:(y+side)] = 0

        return cutout_img
    else: #No cutout regularization
        return img

def record_parser(record):
    """Parse files and generate lower quality images from them"""

    img1 = preprocess(load_image(record))
    img2 = random_polar_transform(img1)
    if np.sum(np.isfinite(img1)) != cropsize**2 or np.sum(np.isfinite(img2)) != cropsize**2:
        img1 = img2 = np.zeros((cropsize,cropsize))

    return img1, img2

def reshaper(img1, img2):
    img1 = tf.reshape(img1, [cropsize, cropsize, channels])
    img2 = tf.reshape(img2, [cropsize, cropsize, channels])
    return img1, img2

def input_fn(dir, subset, batch_size, num_shards):
    """Create a dataset from a list of filenames and shard batches from it"""

    with tf.device('/cpu:0'):

        data_dir1 = "G:/stills_hq-mini/"
        data_dir2 = "F:/ARM_scans-crops/"
        dataset1 = tf.data.Dataset.list_files(data_dir1+subset+"/"+"*.tif")
        dataset2 = tf.data.Dataset.list_files(data_dir2+subset+"/"+"*.tif")
        dataset = dataset1.concatenate(dataset2)

        #dataset = tf.data.Dataset.list_files(dir+subset+"/"+"*.tif")
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(
            lambda file: tf.py_func(record_parser, [file], [tf.float32, tf.float32]),
            num_parallel_calls=num_parallel_calls)
        #print(dataset.output_shapes, dataset.output_types)
        dataset = dataset.map(reshaper, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

        iter = dataset.make_one_shot_iterator()
        img_batch = iter.get_next()

        if num_shards <= 1:
            # No GPU available or only 1 GPU.
            return [img_batch[0]], [img_batch[1]]
        else:
            image_batch = tf.unstack(img_batch, num=batch_size, axis=1)
            feature_shards = [[] for i in range(num_shards)]
            feature_shards_truth = [[] for i in range(num_shards)]
            for i in range(batch_size):
                idx = i % num_shards
                tensors = tf.unstack(image_batch[i], num=2, axis=0)
                feature_shards[idx].append(tensors[0])
                feature_shards_truth[idx].append(tensors[1])
            feature_shards = [tf.parallel_stack(x) for x in feature_shards]
            feature_shards_truth = [tf.parallel_stack(x) for x in feature_shards_truth]

            return feature_shards, feature_shards_truth

def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return

if disp_select:
    disp(select)

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

def sigmoid(x,shift=0,mult=1):
    return 1. / (1. + np.exp(-(x+shift)*mult))

def main():

    tf.reset_default_graph()

    temp = set(tf.all_variables())

    with open(log_file, 'a') as log:
        log.flush()

        # The env variable is on deprecation path, default is set to off.
        os.environ['TF_SYNC_ON_FINISH'] = '0'
        os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

        #with tf.device("/cpu:0"):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
        with tf.control_dependencies(update_ops):

            # Session configuration.
            log_device_placement = False #Once placement is correct, this fills up too much of the cmd window...
            num_intra_threads = 0
            sess_config = tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=log_device_placement,
                intra_op_parallelism_threads=num_intra_threads,
                gpu_options=tf.GPUOptions(force_gpu_compatible=True))

            config = RunConfig(
                session_config=sess_config, model_dir=model_dir)
            hparams=tf.contrib.training.HParams(
                is_chief=config.is_chief)

            #Data pipeline
            img, img_distorted = input_fn(data_dir, 'train', batch_size, num_gpus)
            img_val, img_distorted_val = input_fn(data_dir, 'val', batch_size, num_gpus)

            with tf.Session(config=sess_config) as sess:

                print("Session started")

                sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                temp = set(tf.all_variables())

                #Placeholders to feed data into network
                ____img, ____img_distorted = sess.run([img, img_distorted])
                img1_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img1') for i in ____img]
                img1_distorted_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img1_distorted')
                                     for i in ____img_distorted]
                img2_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img2') for i in ____img]
                img2_distorted_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img2_distorted')
                                     for i in ____img_distorted]

                print("Dataflow established")

                #Hyperparameter placeholders
                outer_enc_lr = tf.placeholder(tf.float32, name='outer_enc_lr')
                inner_enc_lr = tf.placeholder(tf.float32, name='inner_enc_lr')
                outer_dec_lr = tf.placeholder(tf.float32, name='outer_dec_lr')
                inner_dec_lr = tf.placeholder(tf.float32, name='inner_dec_lr')
                discr_lr = tf.placeholder(tf.float32, name='discr_lr')
                siamese_lr = tf.placeholder(tf.float32, name='siamese_lr')

                outer_enc_beta1 = tf.placeholder(tf.float32, shape=(), name='outer_enc_lr')
                inner_enc_beta1 = tf.placeholder(tf.float32, shape=(), name='inner_enc_lr')
                outer_dec_beta1 = tf.placeholder(tf.float32, shape=(), name='outer_dec_lr')
                inner_dec_beta1 = tf.placeholder(tf.float32, shape=(), name='inner_dec_lr')
                discr_beta1 = tf.placeholder(tf.float32, shape=(), name='discr_lr')
                siamese_beta1 = tf.placeholder(tf.float32, shape=(), name='siamese_lr')

                norm_decay_ph = tf.placeholder(tf.float32, shape=(), name='norm_decay')
                train_batch_norm = tf.placeholder(tf.bool, name='norm_decay')

                #########################################################################################

                #Set up experiment
                experiment(img1_ph[0], img2_ph[0], img1_distorted_ph[0], img2_distorted_ph[0], 
                           outer_enc_lr, inner_enc_lr, outer_dec_lr, inner_dec_lr, discr_lr, siamese_lr, 
                           outer_enc_beta1, inner_enc_beta1, outer_dec_beta1, inner_dec_beta1, 
                           discr_beta1, siamese_beta1, norm_decay_ph, train_batch_norm)

                train_ops = exp_dict['train_ops']
                other_ops = exp_dict['other_ops']
                output_ops = exp_dict['output_ops']

                print("Created experiment")

                sess.run(tf.initialize_variables( set(tf.all_variables())-temp), feed_dict={beta1_ph: np.float32(0.9)} )
                train_writer = tf.summary.FileWriter( logDir, sess.graph )

                #Restore session
                saver = tf.train.Saver(max_to_keep=2)
                if not initialize:
                    saver.restore(sess, tf.train.latest_checkpoint(model_dir+"model/"))

                #Track iterations
                counter = 0
                val_counter = 0
                save_counter = counter
                counter_init = counter+1

                base_rate = 0.0002

                #Buffer for hard examples that will be reshown to network
                bad_buffer_size = 25
                bad_buffer_truth = []
                bad_buffer = []
                bad_buffer_mask = []
                for _ in range(bad_buffer_size):
                    buffer_img, buffer_img_distorted = sess.run([img, img_distorted])
                    bad_buffer.append(buffer_img)
                    bad_buffer_distorted.append(buffer_img_distorted)

                bad_buffer_prob = 0.2
                bad_buffer_beta = 0.99
                bad_buffer_thresh = 0.
                bad_buffer_tracker = bad_buffer_prob
                bad_buffer_tracker_beta = 0.99
                bad_buffer_num_uses = 1

                #Here our 'natural' statistics are the total losses of the networks
                nat_stat_mean_beta = 0.99
                nat_stat_std_dev_beta = 0.99
                nat_stat_mean = 1.5
                nat_stat2_mean = 4.

                total_iters = int(1e6)

                print("Starting training")

                def get_example(forbidden=None):
                    """Get example directly from dataset or from hard examples buffer at random"""

                    use_buffer = np.random.rand() < bad_buffer_num_uses*bad_buffer_prob
                    if use_buffer:
                        idx = np.random.randint(0, bad_buffer_size)
                        while idx is forbidden:
                            idx = np.random.randint(0, bad_buffer_size)

                        _img = bad_buffer[idx]
                        _img_distorted = bad_buffer_distorted[idx]
                    else:
                        _img, _img_distorted = sess.run([img, img_distorted])

                    return _img, _img_distorted, idx if use_buffer else None

                while True:
                    #Train for a couple of hours
                    time0 = time.time()

                    while time.time()-time0 < modelSavePeriod:

                        if not val_counter % val_skip_n:
                            val_counter = 0
                        val_counter += 1
                        if val_counter % val_skip_n: #Only increment on non-validation iterations
                            counter += 1

                        if counter <= inner_iters+outer_iters:
                            rate = base_rate
                        elif counter > inner_iters+outer_iters+full_iters:
                            saver.save(sess, save_path=model_dir+"model/", global_step=counter)
                            quit()
                        else:
                            num_steps = 5
                            stepsize = full_iters/num_steps
                            rel_counter = counter-inner_iters-outer_iters
                            rate = np.float32( base_rate * 0.5 ** np.ceil(rel_counter/stepsize) )

                        learning_rate = np.float32(rate)

                        if counter < inner_iters+outer_iters:
                            beta1 = 0.9
                        else:
                            beta1 = 0.9 - 0.4*(counter-inner_iters-outer_iters) / full_iters

                        if counter < inner_iters:
                            norm_decay = 0.9*np.sqrt(counter/inner_iters)
                        elif counter < inner_iters+outer_iters:
                            norm_decay = 0.9*np.sqrt((counter-inner_iters)/outer_iters)
                        elif counter < total_iters - 2*full_iters/3:
                            norm_decay = 0.9
                        else:
                            norm_decay = 1.

                        base_dict = { learning_rate_ph: learning_rate, 
                                      beta1_ph: np.float32(beta1), 
                                      norm_decay_ph: np.float32(norm_decay) }

                        dict = base_dict.copy()

                        #Get examples
                        _img, _img_distorted, buffer_idx = get_example()
                        dict.update( { img1_ph[0]: _img[0], img1_distorted_ph[0]: _img_distorted[0] } )

                        _img, _img_distorted, _ = get_example(buffer_idx)
                        dict.update( { img2_ph[0]: _img[0], img2_distorted_ph[0]: _img_distorted[0] } )

                        #Save outputs occasionally
                        if 0 <= counter <= 1 or not counter % save_result_every_n_batches or (0 <= counter < 10000 and not counter % 1000) or counter == counter_init:

                            #Don't train on validation examples
                            if not val_counter % val_skip_n:
                                results = sess.run( other_ops + output_ops, feed_dict=dict )
                            else:
                                results = sess.run( other_ops + output_ops + train_ops, feed_dict=dict )

                            mse = results[0]
                            output = results[len(other_ops)]

                            #Save images output by networks
                            try:
                                for output_img, output_img_label, output_img_size in \
                                    zip(output_imgs, output_img_label, output_img_sizes):
                                    
                                    save_loc = model_dir + output_img_label + str(counter) + ".tif"
                                    Image.fromarray(output_img).reshape((output_img_size, 
                                        output_img_size)).astype(np.float32).save( save_loc )
                            except:
                                print("Image save failed")
                        else:
                            #Don't train on validation examples
                            if not val_counter % val_skip_n:
                                results = sess.run( other_ops, feed_dict=dict )
                            else:
                                results = sess.run( other_ops + train_ops, feed_dict=dict )

                            mse = results[0]

                        nat_stat_mean = ( nat_stat_mean_beta*nat_stat_mean + 
                                          (1.-nat_stat_mean_beta)*mse[0] )
                        nat_stat2_mean = ( nat_stat_std_dev_beta*nat_stat2_mean + 
                                           (1.-nat_stat_std_dev_beta)*mse[0]**2 )

                        nat_stat_std_dev = np.sqrt(nat_stat2_mean - nat_stat_mean**2)

                        #Decide whether or not to add to buffer using natural statistics
                        if not use_buffer and mse[0] > bad_buffer_thresh:
                            idx = np.random.randint(0, bad_buffer_size)
                            bad_buffer[idx] = _img
                            bad_buffer_distorted[idx] = _img_distorted
                            
                            bad_buffer_tracker = ( bad_buffer_tracker_beta*bad_buffer_tracker + 
                                                   (1.-bad_buffer_tracker_beta) )
                            print("To buffer")#, bad_buffer_thresh, bad_buffer_prob, bad_buffer_tracker)
                        else:
                            bad_buffer_tracker = bad_buffer_tracker_beta*bad_buffer_tracker

                        if bad_buffer_tracker < bad_buffer_prob:
                            step = nat_stat_mean-5*nat_stat_std_dev
                            bad_buffer_thresh = bad_buffer_beta*bad_buffer_thresh + (1.-bad_buffer_beta)*step

                        if bad_buffer_tracker >= bad_buffer_prob:
                            step = nat_stat_mean+5*nat_stat_std_dev
                            bad_buffer_thresh = bad_buffer_beta*bad_buffer_thresh + (1.-bad_buffer_beta)*step

                        message = "NiN w/o Residuals, Iter: {}, MSE: {}, Val: {}".format(
                            counter, mse, 1 if not val_counter % val_skip_n else 0)
                        print(message)
                        try:
                            log.write(message)
                        except:
                            print("Write to log failed")

                    #Save the model
                    saver.save(sess, save_path=model_dir+"model/model.ckpt", global_step=counter)
                    save_counter = counter
    return


if __name__ == '__main__':
    main()