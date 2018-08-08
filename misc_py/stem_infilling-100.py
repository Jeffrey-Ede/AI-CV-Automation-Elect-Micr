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

gen_features0 = 32
gen_features1 = 64
gen_features2 = 64
gen_features3 = 32

nin_features1 = 128
nin_features2 = 256
nin_features3 = 768

nin_features_out1 = 256
nin_features_out2 = 128
nin_features_out3 = 64

#features1 = 32
#features2 = 64
#features3 = 128
#features4 = 256
#features5 = 512

features1 = 64
features2 = 128
features3 = 256
features4 = 512
features5 = features4

num_global_enhancer_blocks = 8
num_local_enhancer_blocks = 4

data_dir = "F:/ARM_scans-crops/"
#data_dir = "E:/stills_hq-mini/"

modelSavePeriod = 0.5 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/gan-infilling-100-STEM/"

shuffle_buffer_size = 5000
num_parallel_calls = 4
num_parallel_readers = 4
prefetch_buffer_size = 5
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
cropsize = 512
generator_input_size = cropsize
height_crop = width_crop = cropsize

#hparams = experiment_hparams(train_batch_size=batch_size, eval_batch_size=16)

weight_decay = 0.0
batch_decay_gen = 0.999
batch_decay_discr = 0.999
initial_learning_rate = 0.001
initial_discriminator_learning_rate = 0.001
num_workers = 1

increase_batch_size_by_factor = 1
effective_batch_size = increase_batch_size_by_factor*batch_size

save_result_every_n_batches = 25000

val_skip_n = 10
trainee_switch_skip_n = 1

max_num_since_training_change = 0

flip_weight = 25

disp_select = False #Display selelected pixels upon startup

#def spectral_norm(w, iteration=1):
#   w_shape = w.shape.as_list()
#   w = tf.reshape(w, [-1, w_shape[-1]])

#   u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.random_normal_initializer(), trainable=False)

#   u_hat = u
#   v_hat = None
#   for i in range(iteration):
#       """
#       power iteration
#       Usually iteration = 1 will be enough
#       """
#       v_ = tf.matmul(u_hat, tf.transpose(w))
#       v_hat = tf.nn.l2_normalize(v_)

#       u_ = tf.matmul(v_hat, w)
#       u_hat = tf.nn.l2_normalize(u_)

#   u_hat = tf.stop_gradient(u_hat)
#   v_hat = tf.stop_gradient(v_hat)

#   sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

#   with tf.control_dependencies([u.assign(u_hat)]):
#       w_norm = w / sigma
#       w_norm = tf.reshape(w_norm, w_shape)

#   return w_norm


def generator_architecture(inputs, phase=False, params=None,train_batch_norm=None):
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
                epsilon=0.01,
                decay=batch_decay_gen,
                center=True, scale=True,
                is_training=train_batch_norm,
                fused=True,
                zero_debias_moving_mean=False,
                renorm=False)
            return batch_norm

        def batch_then_activ(input):
            #_batch_then_activ = _batch_norm_fn(input)
            _batch_then_activ = tf.nn.relu(input)#_batch_then_activ)
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
                               batch_plus_activ=True):
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

            if batch_plus_activ:
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

        def network_in_network(input):

            nin = strided_conv_block(input, nin_features1, 2, 1, pad_size=(1,1))
            nin = strided_conv_block(nin, nin_features2, 2, 1, pad_size=(1,1))
            nin = strided_conv_block(nin, nin_features3, 2, 1, pad_size=(1,1))

            for _ in range(num_global_enhancer_blocks):
                nin = xception_middle_block(nin, nin_features3, pad_size=(1,1))

            nin = deconv_block(nin, nin_features_out1, new_size=(64, 64), pad_size=(1,1))
            nin = deconv_block(nin, nin_features_out2, new_size=(128, 128), pad_size=(1,1))
            nin = deconv_block(nin, nin_features_out3, new_size=(256, 256), pad_size=(1,1))

            return nin

        ##Model building
        input_layer = tf.reshape(inputs, 
                                 [-1, generator_input_size, generator_input_size, channels])

        enc = strided_conv_block(input=input_layer,
                                 filters=gen_features0,
                                 stride=1,
                                 kernel_size=11,
                                 pad_size=(5,5))

        enc = strided_conv_block(enc, gen_features1, 2, 1, pad_size=(1,1))

        with tf.variable_scope("reg"):

            enc += network_in_network(enc)

            for _ in range(num_local_enhancer_blocks):
                enc = xception_middle_block(enc, gen_features2, pad_size=(1,1))

            enc = deconv_block(enc, gen_features3, new_size=(512, 512), pad_size=(1,1))
            enc = strided_conv_block(enc, gen_features3, 1, 1, pad_size=(1,1))

            #enc = conv_block_not_sep(enc, 1, pad_size=(1,1), batch_plus_activ=False)


        enc = pad(enc, (1,1))
        enc = slim.conv2d(
                inputs=enc,
                num_outputs=1,
                kernel_size=3,
                padding="VALID",
                activation_fn=None,
                weights_initializer=None,
                biases_initializer=None)

        enc = tf.tanh(enc)

        return enc

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
            epsilon = 1.e-3
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
                epsilon=0.001,
                decay=batch_decay_discr,
                center=True, 
                scale=True,
                is_training=phase,
                fused=True,
                zero_debias_moving_mean=False,
                renorm=False)
            return batch_norm

        def batch_then_activ(input): #Changed to instance norm for stability
            batch_then_activ = _instance_norm(input)
            batch_then_activ = tf.nn.leaky_relu(batch_then_activ)
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

        def strided_conv_block(input, filters, stride, rate=1, phase=phase, 
                               extra_batch_norm=False, kernel_size=3):
        
            #w = tf.get_variable("kernel", shape=[kernel_size, kernel_size, input.get_shape()[-1], filters])
            #b = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))

            #x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[1, stride, stride, 1]) + b

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
                normalizer_fn=_batch_norm_fn if extra_batch_norm else None,
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

        with tf.variable_scope("small", reuse=reuse) as small_scope:
            small = inputs[0]
            small = strided_conv_block(small, features1, 2, 1)
            layers.append(small)
            small = strided_conv_block(small, features2, 2, 1)
            layers.append(small)
            small = strided_conv_block(small, features3, 2, 1)
            layers.append(small)
            small = strided_conv_block(small, features4, 2, 1)
            layers.append(small)
            small = strided_conv_block(small, features5, 2, 1)
            layers.append(small)
            #small = tf.reduce_mean(small, [1,2])
            small = tf.reshape(small, (-1, features5*4))
            small = tf.contrib.layers.fully_connected(inputs=small,
                                                   num_outputs=1,
                                                   activation_fn=None)

        with tf.variable_scope("medium", reuse=reuse) as medium_scope:
            medium = inputs[1]
            #medium = tf.nn.avg_pool(medium,
            #                        [1, 2, 2, 1],
            #                        strides=[1, 2, 2, 1],
            #                        padding='SAME')
            medium = strided_conv_block(medium, features1, 2, 1)
            layers.append(medium)
            medium = strided_conv_block(medium, features2, 2, 1)
            layers.append(medium)
            medium = strided_conv_block(medium, features3, 2, 1)
            layers.append(medium)
            medium = strided_conv_block(medium, features4, 2, 1)
            layers.append(medium)
            medium = strided_conv_block(medium, features5, 2, 1)
            layers.append(medium)
            #medium = tf.reduce_mean(medium, [1,2])
            medium = tf.reshape(medium, (-1, features5*4))
            medium = tf.contrib.layers.fully_connected(inputs=medium,
                                                   num_outputs=1,
                                                   activation_fn=None)

        with tf.variable_scope("large", reuse=reuse) as large_scope:
            large = inputs[2]
            large = strided_conv_block(large, features1, 2, 1)
            layers.append(large)
            large = strided_conv_block(large, features2, 2, 1)
            layers.append(large)
            large = strided_conv_block(large, features3, 2, 1)
            layers.append(large)
            large = strided_conv_block(large, features4, 2, 1)
            layers.append(large)
            large = strided_conv_block(large, features5, 2, 1)
            layers.append(large)
            #large = tf.reduce_mean(large, [1,2])
            large = tf.reshape(large, (-1, features5*4))
            large = tf.contrib.layers.fully_connected(inputs=large,
                                                   num_outputs=1,
                                                   activation_fn=None)

        #with tf.variable_scope("small-start", reuse=reuse) as small_start_scope:
        #    small = inputs[0]
        #    small = strided_conv_block(small, features1, 1, 1)
        #    layers.append(small)
        #    small = strided_conv_block(small, features2, 1, 1)
        #    layers.append(small)

        #with tf.variable_scope("medium-start", reuse=reuse):
        #    medium = inputs[1]
        #    medium = tf.nn.avg_pool(medium,
        #                            [1, 2, 2, 1],
        #                            strides=[1, 2, 2, 1],
        #                            padding='SAME')
        #    medium = strided_conv_block(medium, features1, 1, 1)
        #    layers.append(medium)
        #    medium = strided_conv_block(medium, features2, 1, 1)
        #    layers.append(medium)

        #with tf.variable_scope("large-start", reuse=reuse):
        #    large = inputs[2]
        #    #large = tf.nn.avg_pool(large,
        #    #                        [1, 4, 4, 1],
        #    #                        strides=[1, 4, 4, 1],
        #    #                        padding='SAME')
        #    large = strided_conv_block(large, features1, 1, 1)
        #    layers.append(large)
        #    large = strided_conv_block(large, features2, 1, 1)
        #    layers.append(large)

        #with tf.variable_scope("shared", reuse=reuse) as shared_scope:
        #    small, layers = shared_flow(small, layers)
        #with tf.variable_scope(shared_scope, reuse=True):
        #    medium, layers = shared_flow(medium, layers)
        #with tf.variable_scope(shared_scope, reuse=True):
        #    large, layers = shared_flow(large, layers)

        #with tf.variable_scope("small-end", reuse=reuse) as small_end_scope:
        #    small = xception_middle_block(small, features5)
        #    layers.append(small)
        #    small = xception_middle_block(small, features5)
        #    layers.append(small)
        #    small = slim.conv2d(
        #        inputs=large,
        #        num_outputs=features5,
        #        kernel_size=3,
        #        padding="SAME",
        #        activation_fn=tf.nn.leaky_relu)
        #    small = instance_then_activ(small)
        #    small = terminating_fc(small)

        #with tf.variable_scope("medium-end", reuse=reuse):
        #    medium = xception_middle_block(medium, features5)
        #    layers.append(medium)
        #    medium = xception_middle_block(medium, features5)
        #    layers.append(medium)
        #    medium = slim.conv2d(
        #        inputs=medium,
        #        num_outputs=features5,
        #        kernel_size=3,
        #        padding="SAME",
        #        activation_fn=tf.nn.leaky_relu)
        #    medium = instance_then_activ(medium)
        #    medium = terminating_fc(medium)

        #with tf.variable_scope("large-end", reuse=reuse):
        #    large = xception_middle_block(large, features5)
        #    layers.append(large)
        #    large = xception_middle_block(large, features5)
        #    layers.append(large)
        #    large = slim.conv2d(
        #        inputs=large,
        #        num_outputs=features5,
        #        kernel_size=3,
        #        padding="SAME",
        #        activation_fn=tf.nn.leaky_relu)
        #    large = instance_then_activ(large)
        #    large = terminating_fc(large)

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
        def _model_fn(features, ground_truths, mode=None, params=None, train_batch_norm=None):
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
            tower_truths = ground_truths
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
                                is_training, tower_features[i], tower_truths[i], train_batch_norm)

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
        def _model_fn(features, labels=None, mode=None, params=None, adapt_rates=None):
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
                                is_training, tower_features[i], tower_labels[i], adapt_rates[i])

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

    def pad(tensor, size):
        d1_pad = size[0]
        d2_pad = size[1]

        paddings = tf.constant([[0, 0], [d1_pad, d1_pad], [d2_pad, d2_pad], [0, 0]], dtype=tf.int32)
        padded = tf.pad(tensor, paddings, mode="REFLECT")
        return padded

    input = pad(input, ((3*cropsize)//4, (3*cropsize)//4))

    small = tf.random_crop(
                input,
                size=(batch_size, cropsize//4, cropsize//4, multiscale_channels))
    small = tf.image.resize_images(small, (cropsize//8, cropsize//8))
    medium = tf.random_crop(
                input,
                size=(batch_size, cropsize//2, cropsize//2, multiscale_channels))
    medium = tf.image.resize_images(medium, (cropsize//8, cropsize//8))
    large = tf.random_crop(
                input,
                size=(batch_size, (3*cropsize)//4, (3*cropsize)//4, multiscale_channels))
    large = tf.image.resize_images(large, (cropsize//8, cropsize//8))

    return small, medium, large

def _generator_tower_fn(is_training, feature, ground_truth, train_batch_norm):

    #phase = tf.estimator.ModeKeys.TRAIN if is_training else tf.estimator.ModeKeys.EVAL
    feature = tf.reshape(feature, [-1, cropsize, cropsize, channels])
    truth = tf.reshape(ground_truth, [-1, cropsize, cropsize, channels])

    output = generator_architecture(feature, is_training, train_batch_norm=train_batch_norm)

    model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                     scope="nn/GAN/Gen")

    l2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                     scope="nn/GAN/Gen/reg")

    concat = tf.concat([output, truth], axis=3)
    shapes = [(batch_size, cropsize//8, cropsize//8, channels),
              (batch_size, cropsize//8, cropsize//8, channels),
              (batch_size, cropsize//8, cropsize//8, channels)]
    multiscale_crops = get_multiscale_crops(concat, multiscale_channels=2)
    multiscale_crops = [tf.unstack(crop, axis=3) for crop in multiscale_crops]
    multiscale = [tf.reshape(unstacked[0], shape) 
                  for unstacked, shape in zip(multiscale_crops, shapes)]
    multiscale_natural = [tf.reshape(unstacked[1], shape) 
                          for unstacked, shape in zip(multiscale_crops, shapes)]

    discrimination = discriminator_architecture(multiscale, is_training, reuse=False)

    #Compare discrimination features for generation against those for a real image
    discrimination_natural = discriminator_architecture(multiscale_natural, 
                                                        is_training, 
                                                        reuse=True)

    output_d = discrimination_natural[0] - discrimination[0]
    output_d = tf.sigmoid(output_d)

    natural_stat_losses = []
    for i in range(1, len(discrimination)):
        natural_stat_losses.append(
            tf.reduce_mean(
                tf.losses.absolute_difference(
                    discrimination[i], discrimination_natural[i])))
    natural_stat_loss = tf.add_n(natural_stat_losses)

    weight_natural_stats = 10.

    loss = -tf.log(tf.clip_by_value(output_d, 1.e-8, 1.))
    loss += weight_natural_stats*natural_stat_loss

    #loss += 1.e-6 * tf.add_n(
    #    [tf.nn.l2_loss(v) for v in l2_params])
    
    tower_grad = tf.gradients(loss, model_params)

    return tower_grad, output, output_d


def _discriminator_tower_fn(is_training, feature, ground_truth, adapt_rates):
    """Build computation tower.
        Args:
        is_training: true if is training graph.
        feature: a Tensor.
        Returns:
        A tuple with the loss for the tower, the gradients and parameters, and
        predictions.
    """

    feature = tf.reshape(feature, [-1, cropsize, cropsize, channels])
    shapes = [(batch_size, cropsize//8, cropsize//8, channels),
              (batch_size, cropsize//8, cropsize//8, channels),
              (batch_size, cropsize//8, cropsize//8, channels)]
    multiscale_crops = get_multiscale_crops(feature)
    multiscale_crops = [tf.reshape(crop, shape) 
                         for crop, shape in zip(multiscale_crops, shapes)]

    output = discriminator_architecture(multiscale_crops, is_training, reuse=True)[0]
    output = tf.sigmoid(output)

    model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="nn/GAN/Discr")

    #tower_loss_usual = -tf.cond(ground_truth[0] > 0.5, #Only works for batch size 1
    #                      lambda: tf.log(tf.clip_by_value(ground_truth[0]+output-1.e-8, 1.e-8, 1.)), 
    #                      lambda: tf.log(tf.clip_by_value(tf.abs(ground_truth[0]-output), 1.e-8, 1.)))
    #tower_loss_flipped = -tf.cond(-ground_truth[0] < 0.5, #Only works for batch size 1
    #                      lambda: tf.log(tf.clip_by_value(-ground_truth[0]+output-1.e-8, 1.e-8, 1.)), 
    #                      lambda: tf.log(tf.clip_by_value(tf.abs(-ground_truth[0]-output), 1.e-8, 1.)))

    #tower_loss = tf.cond(ground_truth[0] >= 0, lambda: tower_loss_usual, lambda: tower_loss_flipped)

    tower_loss = -tf.log(tf.clip_by_value(1.-tf.abs(ground_truth[0]-output), 1.e-8, 1.-1.e-8))

    #tower_loss += 5.e-6 * tf.add_n([tf.nn.l2_loss(v) for v in model_params])

    adjusted_tower_loss = adapt_rates[0] * tower_loss

    tower_grad = tf.gradients(adjusted_tower_loss, model_params)

    return tower_loss, tower_grad, output


def adam_updates(params, cost_or_grads, lr=0.001, mom1=np.array([0.5], dtype=np.float32), 
                 mom2=np.array([0.999], dtype=np.float32), clip_norm=40):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads
    #grads = tf.clip_by_global_norm(grads, clip_norm=clip_norm)
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1 is not None:
            v = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_v')
            v_t = mom1*v + (1. - mom1)*g
            v_hat = v_t / (1. - tf.pow(mom1,t))
            updates.append(v.assign(v_t))
        else:
            v_hat = g
        mg_t = mom2*mg + (1. - mom2)*tf.square(g)
        mg_hat = mg_t / (1. - tf.pow(mom2,t))
        g_t = v_hat / tf.sqrt(mg_hat + 1.e-8)
        p_t = p - lr * g_t
        updates.append(mg.assign(mg_t))
        updates.append(p.assign(p_t))
    updates.append(t.assign_add(1))
    return tf.group(*updates)


def experiment(is_training, feature, ground_truth, train_batch_norm, adapt_rates, 
               flip_fake, flip_real, learning_rate_ph, 
               discriminator_learning_rate_ph, beta1_ph):

    #flip_prob_fake_ph = flip_prob_real_ph = flip_ph

    #Generator
    feature = tf.reshape(feature, [-1, cropsize, cropsize, channels])
    truth = tf.reshape(ground_truth, [-1, cropsize, cropsize, channels])

    output = generator_architecture(feature, is_training, train_batch_norm=train_batch_norm)

    model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                     scope="GAN/Gen")

    #l2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
    #                                 scope="GAN/Gen/reg")

    concat = tf.concat([output, truth], axis=3)
    shapes = [(batch_size, cropsize//8, cropsize//8, channels),
              (batch_size, cropsize//8, cropsize//8, channels),
              (batch_size, cropsize//8, cropsize//8, channels)]
    multiscale_crops = get_multiscale_crops(concat, multiscale_channels=2)
    multiscale_crops = [tf.unstack(crop, axis=3) for crop in multiscale_crops]
    multiscale = [tf.reshape(unstacked[0], shape) 
                  for unstacked, shape in zip(multiscale_crops, shapes)]
    multiscale_natural = [tf.reshape(unstacked[1], shape) 
                          for unstacked, shape in zip(multiscale_crops, shapes)]

    discrimination = discriminator_architecture(multiscale, is_training, reuse=False)

    #Compare discrimination features for generation against those for a real image
    discrimination_natural = discriminator_architecture(multiscale_natural, 
                                                        is_training, 
                                                        reuse=True)

    x = discrimination[0] - discrimination_natural[0]
    #output_d = tf.sigmoid(output_d)

    output_d = tf.cond(x[0] < 0., lambda: 0.5*x**2+x+0.5, lambda: -0.5*x**2+x+0.5)

    natural_stat_losses = []
    for i in range(1, len(discrimination)):
        natural_stat_losses.append(
            tf.reduce_mean(
                tf.losses.absolute_difference(
                    discrimination[i], discrimination_natural[i])))
    natural_stat_loss = tf.add_n(natural_stat_losses)

    weight_natural_stats = 10.

    loss = -tf.log(tf.clip_by_value(output_d, 1.e-8, 1.))
    loss += weight_natural_stats*natural_stat_loss

    #Discriminator
    discr_model_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Discr")

    fake_label = tf.cond(flip_fake,
                         lambda: 0.,
                         lambda: tf.random_uniform((1,), minval=0.9, maxval=1.)[0])

    real_label = tf.cond(flip_real,
                          lambda: tf.random_uniform((1,), minval=0.9, maxval=1.)[0],
                          lambda: 0.)

    pred_fake = discrimination[0]#tf.sigmoid(discrimination[0])
    discr_tower_loss_fake = -tf.log(tf.clip_by_value(1.-tf.abs(fake_label-pred_fake), 1.e-8, 1.-1.e-8))

    pred_real = discrimination_natural[0]#tf.sigmoid(discrimination_natural[0])
    discr_tower_loss_real = -tf.log(tf.clip_by_value(1.-tf.abs(real_label-pred_real), 1.e-8, 1.-1.e-8))

    adapt_fake = tf.cond(flip_fake, lambda: np.float32(1.), lambda: adapt_rates)
    adapt_real = tf.cond(flip_fake, lambda: np.float32(1.), lambda: adapt_rates)

    discr_l2_loss = 1.e-5 * tf.add_n([tf.nn.l2_loss(v) for v in discr_model_params])
    adjusted_tower_loss = (adapt_fake*discr_tower_loss_fake + adapt_real*discr_tower_loss_real +
                           (adapt_fake+adapt_real)*discr_l2_loss)
    

    gen_train_op = adam_updates(model_params, loss, learning_rate_ph, beta1_ph)
    discr_train_op = tf.train.AdamOptimizer(discriminator_learning_rate_ph, beta1_ph).minimize(
        adjusted_tower_loss, var_list=discr_model_params)

    train_ops = [gen_train_op, discr_train_op]

    return {'rel_prob': output_d, 'real_prob': pred_real, 
            'fake_prob': pred_fake, 'train_ops': train_ops, 
            'output': output, 'did_it_flip': [flip_fake, flip_real]}

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


def load_image(addr, resizeSize=None, img_type=np.float32):
    """Read an image and make sure it is of the correct type. Optionally resize it"""
    
    try:
        img = imread(addr, mode='F')
    except:
        img = np.zeros((cropsize,cropsize))
        print("Image read failed")

    if img.shape != (cropsize, cropsize):
        img = np.zeros((cropsize,cropsize))
        print("Image had wrong shape")

    if resizeSize:
        img = cv2.resize(img, resizeSize, interpolation=cv2.INTER_AREA)

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

    img = norm_img(img)

    #img = cv2.resize(img, (cropsize, cropsize))

    return img

frac = 1./100 #np.random.uniform(0.01, 0.05)
np.random.seed(1)
select = np.random.random((cropsize, cropsize)) < frac

def gen_lq(img):

    lq = -np.ones(img.shape)
    lq[select] = img[select]

    return lq.astype(np.float32)

def record_parser(record):
    """Parse files and generate lower quality images from them"""

    img = flip_rotate(preprocess(load_image(record)))
    lq = gen_lq(img)
    if np.sum(np.isfinite(img)) != cropsize**2 or np.sum(np.isfinite(lq)) != cropsize**2:
        img = lq = np.zeros((cropsize,cropsize))*select

    return lq, img

def reshaper(img1, img2):
    img1 = tf.reshape(img1, [cropsize, cropsize, channels])
    img2 = tf.reshape(img2, [cropsize, cropsize, channels])
    return img1, img2

def input_fn(dir, subset, batch_size, num_shards):
    """Create a dataset from a list of filenames and shard batches from it"""

    with tf.device('/cpu:0'):

        dataset = tf.data.Dataset.list_files(dir+subset+"/"+"*.tif")
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.map(
            lambda file: tf.py_func(record_parser, [file], [tf.float32, tf.float32]),
            num_parallel_calls=num_parallel_calls)
        #print(dataset.output_shapes, dataset.output_types)
        dataset = dataset.map(reshaper, num_parallel_calls=num_parallel_calls)
        #print(dataset.output_shapes, dataset.output_types)
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
        #tower_gradvars = []
        #for tower_grad in kwargs['_tower_grads']:
        #    tower_gradvars.append(zip(tower_grad, tower_params))

        ## Now compute global loss and gradients.
        #gradvars = []
        #with tf.name_scope('gradient_averaging'):
        #    all_grads = {}
        #    for grad, var in itertools.chain(*tower_gradvars):
        #        if grad is not None:
        #            all_grads.setdefault(var, []).append(grad)
        #    for var, grads in six.iteritems(all_grads):
        #        # Average gradients on the same device as the variables
        #        # to which they apply.
        #        with tf.device(var.device):
        #            if len(grads) == 1:
        #                avg_grad = grads[0]
        #            else:
        #                avg_grad = tf.multiply(tf.add_n(grads), 1. / len(grads))

        #        gradvars.append((avg_grad, var))
                
        #global_step = tf.train.get_global_step()

        # Device that runs the ops to apply global gradient updates.
        consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
        with tf.device(consolidation_device):

            train_op = adam_updates(tower_params, kwargs['_tower_grads'][0], learning_rate_ph)
        #    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_ph, beta1=0.5)
        #    optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 40.0)

        ## Create single grouped train op
        #train_op = [optimizer.apply_gradients(gradvars, global_step=global_step)]

        #train_op.extend(update_ops)
        #train_op = tf.group(*train_op)

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
            #discriminator_optimizer = tf.contrib.estimator.clip_gradients_by_norm(
            #    discriminator_optimizer, 15.0)

        # Create single grouped train op
        discriminator_train_op = [
            discriminator_optimizer.apply_gradients(discriminator_gradvars, 
                                      global_step=discriminator_global_step)]

        discriminator_train_op.extend(discriminator_update_ops)
        discriminator_train_op = tf.group(*discriminator_train_op)

    return discriminator_train_op, discriminator_loss

def sigmoid(x):
  return 1 / (1 + np.exp(-np.array(x)))


def main(job_dir, data_dir, variable_strategy, num_gpus, log_device_placement, 
         num_intra_threads, **hparams):

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

            img, img_truth = input_fn(data_dir, 'train', batch_size, num_gpus)
            img_val, img_truth_val = input_fn(data_dir, 'val', batch_size, num_gpus)

            with tf.Session(config=sess_config) as sess: #Alternative is tf.train.MonitoredTrainingSession()

                print("Session started")

                sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                #sess.run( tf.global_variables_initializer())
                temp = set(tf.all_variables())

                ____img, ____img_truth = sess.run([img, img_truth])
                img_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img') 
                            for i in ____img]
                img_truth_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img_truth') 
                                for i in ____img_truth]

                is_training = True

                generator_model_fn = get_model_fn(
                    num_gpus, variable_strategy, num_workers, component="generator")
                discriminator_model_fn = get_model_fn(
                    num_gpus, variable_strategy, num_workers, component="discriminator")

                print("Dataflow established")

                flip_prob_fake_ph = tf.placeholder(tf.bool, name='flip_fake')
                flip_prob_real_ph = tf.placeholder(tf.bool, name='flip_real')
                batch_norm_on_ph = tf.placeholder(tf.bool, name='train_batch_norm')
                learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')
                beta1_ph = tf.placeholder(tf.float32, shape=(), name='beta1')
                discriminator_learning_rate_ph = tf.placeholder(tf.float32, name='discr_learning_rate')

                print("Generator flow established")

                adapt_rates_ph = tf.placeholder(tf.float32, name='adapt')

                #########################################################################################

                exp_dict = experiment(is_training, img_ph[0], img_truth_ph[0], batch_norm_on_ph, 
                                      adapt_rates_ph, flip_prob_fake_ph, flip_prob_real_ph, 
                                      learning_rate_ph, discriminator_learning_rate_ph, beta1_ph)

                sess.run(tf.initialize_variables(set(tf.all_variables()) - temp), feed_dict={beta1_ph: np.float32(0.9)})
                train_writer = tf.summary.FileWriter( logDir, sess.graph )

                #print(tf.all_variables())
                saver = tf.train.Saver()
                #saver.restore(sess, tf.train.latest_checkpoint(
                #    "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/gan-infilling-64-3/model/"))

                learning_rate = initial_learning_rate
                discriminator_learning_rate = initial_discriminator_learning_rate

                counter = 0
                save_counter = counter
                counter_init = counter+1

                b = 0.99
                avg_p = 0.5
                avg_p_fake = 0.5
                avg_p_real = 0.5

                base_rate = 0.0002
                flip_factor = 0.01/base_rate

                other_ops = [exp_dict['did_it_flip']]

                while True:
                    #Train for a couple of hours
                    time0 = time.time()

                    #base_rate = 0.0003
                    #if counter < 500000:
                    #    rate = base_rate
                    #    beta1 = 0.9
                    #    b = 0.99
                    #else:
                    #    if counter > 1000000:
                    #        saver.save(sess, save_path=model_dir+"model/", global_step=counter)
                    #        quit()
                    #    step =  (counter-500000) // 50000 + 1
                    #    rate = base_rate*(1-step/11)
                    #    beta1 = 0.5
                    #    b = 0.997

                    if counter < 500000:
                        rate = base_rate
                    else:
                        step = (counter-500000) // 50000 + 1
                        max_step = (1000000-500000) // 50000 + 1
                        rate = base_rate*(1.-step/max_step)

                    if counter > 1000000:
                        saver.save(sess, save_path=model_dir+"model/", global_step=counter)
                        quit()
                    beta1 = 0.9 if counter < 500000 else 0.5

                    train_batch_norm_on = counter < 0

                    learning_rate = np.float32(rate)
                    discriminator_learning_rate = learning_rate

                    while time.time()-time0 < modelSavePeriod:
                        counter += 1

                        flip_fake = np.random.rand() > np.float32(flip_factor*rate/(avg_p_fake+0.001))
                        flip_real = np.random.rand() > np.float32(flip_factor*rate/(1.-avg_p_real+0.001))
                        adapt = np.sqrt(avg_p_fake*(1.-avg_p_real)) / 0.5 #avg_p/0.5#1.
                        base_dict = {learning_rate_ph: learning_rate,
                                        discriminator_learning_rate_ph: discriminator_learning_rate,
                                        batch_norm_on_ph: train_batch_norm_on,
                                        flip_prob_fake_ph: np.bool(flip_fake),
                                        flip_prob_real_ph: np.bool(flip_real),
                                        adapt_rates_ph: np.float32(adapt),
                                        beta1_ph: np.float32(beta1)}

                        _img, _img_truth = sess.run([img, img_truth])

                        dict = base_dict.copy()
                        dict.update({img_ph[0]: _img[0],
                                        img_truth_ph[0]: _img_truth[0]})

                        #Save outputs occasionally
                        if counter <= 1 or not counter % save_result_every_n_batches or (counter < 10000 and not counter % 1000) or counter == counter_init:
                                    
                            results = sess.run(
                                exp_dict['train_ops'] +
                                [exp_dict['rel_prob'],
                                exp_dict['real_prob'],
                                exp_dict['fake_prob'],
                                exp_dict['output']] +
                                other_ops, feed_dict=dict)

                            rel_prob = results[2]
                            real_prob = results[3]
                            fake_prob = results[4]
                            output = results[5]
                                    
                            try:
                                save_input_loc = model_dir+"input-"+str(counter)+".tif"
                                save_truth_loc = model_dir+"truth-"+str(counter)+".tif"
                                save_output_loc = model_dir+"output-"+str(counter)+".tif"
                                Image.fromarray(scale0to1(_img[0]).reshape(cropsize, cropsize).astype(np.float32)).save( save_input_loc )
                                Image.fromarray(scale0to1(_img_truth[0]).reshape(cropsize, cropsize).astype(np.float32)).save( save_truth_loc )
                                Image.fromarray(scale0to1(output).reshape(cropsize, cropsize).astype(np.float32)).save( save_output_loc )
                            except:
                                print("Image save failed")
                        else:
                            results = sess.run(exp_dict['train_ops'] + 
                                                [exp_dict['rel_prob'],
                                                exp_dict['real_prob'],
                                                exp_dict['fake_prob']] +
                                                other_ops, feed_dict=dict)
                            rel_prob = results[2]
                            real_prob = results[3]
                            fake_prob = results[4]

                        print(results[5:], np.float32(flip_factor*rate/(avg_p_fake+0.001)), np.float32(flip_factor*rate/(1.-avg_p_real+0.001)))

                        #avg_p = b*avg_p + (1.-b)*rel_prob
                        avg_p_fake = b*avg_p_fake + (1.-b)*fake_prob
                        avg_p_real = b*avg_p_real + (1.-b)*real_prob

                        message = "Iter: {}, Rel prob: {}, Real prob: {}, Fake prob: {}".format(
                            counter, rel_prob, real_prob, fake_prob)
                        print(message)

                        try:
                            log.write(message)
                        except:
                            print("Write to log failed")

                    #Save the model
                    saver.save(sess, save_path=model_dir+"model/", global_step=counter)
                    save_counter = counter
    return


#def main(job_dir, data_dir, variable_strategy, num_gpus, log_device_placement, 
#         num_intra_threads, **hparams):

#    tf.reset_default_graph()

#    temp = set(tf.all_variables())

#    with open(discr_pred_file, 'a') as discr_pred_log:
#        discr_pred_log.flush()

#        with open(log_file, 'a') as log:
#            log.flush()

#            with open(val_log_file, 'a') as val_log:
#                val_log.flush()

#                # The env variable is on deprecation path, default is set to off.
#                os.environ['TF_SYNC_ON_FINISH'] = '0'
#                os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

#                #with tf.device("/cpu:0"):
#                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
#                with tf.control_dependencies(update_ops):

#                    # Session configuration.
#                    log_device_placement = False #Once placement is correct, this fills up too much of the cmd window...
#                    sess_config = tf.ConfigProto(
#                        allow_soft_placement=True,
#                        log_device_placement=log_device_placement,
#                        intra_op_parallelism_threads=num_intra_threads,
#                        gpu_options=tf.GPUOptions(force_gpu_compatible=True))

#                    config = RunConfig(
#                        session_config=sess_config, model_dir=job_dir)
#                    hparams=tf.contrib.training.HParams(
#                        is_chief=config.is_chief,
#                        **hparams)

#                    img, img_truth = input_fn(data_dir, 'train', batch_size, num_gpus)
#                    img_val, img_truth_val = input_fn(data_dir, 'val', batch_size, num_gpus)

#                    with tf.Session(config=sess_config) as sess: #Alternative is tf.train.MonitoredTrainingSession()

#                        print("Session started")

#                        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
#                        #sess.run( tf.global_variables_initializer())
#                        temp = set(tf.all_variables())

#                        ____img, ____img_truth = sess.run([img, img_truth])
#                        img_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img') 
#                                  for i in ____img]
#                        img_truth_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img_truth') 
#                                        for i in ____img_truth]

#                        is_training = True

#                        generator_model_fn = get_model_fn(
#                            num_gpus, variable_strategy, num_workers, component="generator")
#                        discriminator_model_fn = get_model_fn(
#                            num_gpus, variable_strategy, num_workers, component="discriminator")

#                        print("Dataflow established")

#                        #########################################################################################

#                        flip_ph = [tf.placeholder(tf.float32, shape=(1), name='flip')
#                                    for _ in range(len(preds))]
#                        batch_norm_on_ph = tf.placeholder(tf.bool, name='train_batch_norm')

#                        results = generator_model_fn(img_ph, 
#                                                     img_truth_ph,
#                                                     mode=is_training, 
#                                                     params=hparams,
#                                                     train_batch_norm=batch_norm_on_ph)
#                        _tower_preds = results[0]
#                        _discr_preds = results[1]
#                        update_ops = results[2]
#                        tower_grads = results[3:(3+batch_size)]

#                        learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')

#                        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
#                        temp = set(tf.all_variables())

#                        mini_batch_dict = {batch_norm_on_ph: False}
#                        for i in range(batch_size):
#                            _img, _img_truth = sess.run([img[i], img_truth[i]])
#                            mini_batch_dict.update({img_ph[i]: _img})
#                            mini_batch_dict.update({img_truth_ph[i]: _img_truth})

#                        gradvars_pry, preds = sess.run([tower_grads, _tower_preds], 
#                                                       feed_dict=mini_batch_dict)
#                        del mini_batch_dict

#                        tower_grads_ph = [[tf.placeholder(tf.float32, shape=t.shape, name='tower_grads') 
#                                           for t in gradvars_pry[0]] 
#                                           for _ in range(effective_batch_size)]
#                        del gradvars_pry
            
#                        train_op = _train_op(variable_strategy,
#                                             update_ops, 
#                                             learning_rate_ph,
#                                             _tower_grads=tower_grads_ph)

#                        print("Generator flow established")

#                        #########################################################################################

#                        pred_ph = [tf.placeholder(tf.float32, shape=i.shape, name='prediction')
#                                   for i in preds]
#                        label_ph = [tf.placeholder(tf.float32, shape=(1), name='label')
#                                    for _ in range(len(preds))]
#                        adapt_rates_ph = [tf.placeholder(tf.float32, shape=(1), name='adapt')
#                                    for _ in range(len(preds))]

#                        #The closer the label is to 1., the more confident the discriminator is that the image is real
#                        discriminator_results = discriminator_model_fn(
#                            pred_ph, label_ph, mode=is_training, params=hparams, adapt_rates=adapt_rates_ph)
#                        _discriminator_tower_losses = discriminator_results[0]
#                        _discriminator_tower_preds = discriminator_results[1]
#                        _discriminator_update_ops = discriminator_results[2]
#                        _discriminator_tower_grads = discriminator_results[3:(3+batch_size)]

#                        discriminator_tower_losses_ph = tf.placeholder(
#                            tf.float32, shape=(2*effective_batch_size,), 
#                            name='discriminator_tower_losses')
#                        discriminator_learning_rate_ph = tf.placeholder(
#                            tf.float32, name='discriminator_learning_rate')

#                        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
#                        temp = set(tf.all_variables())

#                        mini_batch_dict = {adapt_rates_ph[0]: np.array([1.])}
#                        for i in range(batch_size):
#                            mini_batch_dict.update({ph: np.random.rand(cropsize, cropsize, 1) for ph in pred_ph})
#                            mini_batch_dict.update({ph: val for ph, val in zip(pred_ph, preds)})
#                            mini_batch_dict.update({ph: np.array([0.5]) for ph in label_ph})
#                        discriminator_gradvars_pry = sess.run(_discriminator_tower_grads, feed_dict=mini_batch_dict)
#                        del preds
#                        del mini_batch_dict

#                        discriminator_tower_grads_ph = [[
#                            tf.placeholder(tf.float32, shape=t.shape, name='discriminator_tower_grads') 
#                            for t in discriminator_gradvars_pry[0]] for _ in range(2*effective_batch_size)]
#                        del discriminator_gradvars_pry
            
#                        discriminator_train_op, discriminator_get_loss = _discriminator_train_op(
#                            discriminator_tower_losses_ph,
#                            variable_strategy,
#                            _discriminator_update_ops,
#                            discriminator_learning_rate_ph,
#                            _tower_grads=discriminator_tower_grads_ph)

#                        print("Discriminator flows established")

#                        #########################################################################################

#                        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
#                        train_writer = tf.summary.FileWriter( logDir, sess.graph )

#                        #print(tf.all_variables())
#                        saver = tf.train.Saver()
#                        #saver.restore(sess, tf.train.latest_checkpoint(
#                        #    "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/gan-infilling-64-3/model/"))

#                        learning_rate = initial_learning_rate
#                        discriminator_learning_rate = initial_discriminator_learning_rate

#                        offset = 30
#                        train_gen = False
#                        avg_pred = 0. #To decide which to train
#                        avg_pred_real = 0. #To decide which to train
#                        num_since_change = 0.

#                        b = 0.99

#                        counter = 0
#                        save_counter = counter
#                        counter_init = counter+1
#                        pred_avg = 0.5
#                        pred_avg_real = 0.5
#                        while True:
#                            #Train for a couple of hours
#                            time0 = time.time()
                    
#                            #with open(model_dir+"learning_rate.txt") as lrf:
#                            #    try:
#                            #        learning_rates = [float(line) for line in lrf]
#                            #        learning_rate = np.float32(learning_rates[0])
#                            #        discriminator_learning_rate = np.float32(learning_rate[1])
#                            #        print("Using learning rates: {}, {}".format(learning_rate, 
#                            #                                                    discriminator_learning_rate))
#                            #    except:
#                            #        pass

#                            #if counter < 30000:
#                            #    rate = (0.002-0.0002)*(1-counter/100000) + 0.0002
#                            #if counter < 400000:
#                            #    rate = 0.0002
#                            #else:
#                            #    if counter > 800000:
#                            #        saver.save(sess, save_path=model_dir+"model/", global_step=counter)
#                            #        quit()
#                            #    step =  (counter-400000) // 50000 + 1
#                            #    rate = 0.0002*(1-step/9)

#                            step = counter // 50000
#                            max_step = 800000 // 50000
#                            rate = 0.0005*(1.-step/max_step)**2

#                            train_batch_norm_on = counter < 0

#                            learning_rate = np.float32(rate)
#                            discriminator_learning_rate = learning_rate / 5.

#                            generations = []
#                            while time.time()-time0 < modelSavePeriod:
#                                counter += 1

#                                #Apply the generator
#                                tower_preds_list = []
#                                tower_ground_truths_list = []
#                                if train_gen:
#                                    discriminator_tower_preds_list = []
#                                ph_dict = {}
#                                for j in range(increase_batch_size_by_factor):

#                                    mini_batch_dict = {}
#                                    __img, __img_truth = sess.run([img, img_truth])
#                                    tower_ground_truths_list += __img_truth

#                                    for i in range(batch_size):
#                                        mini_batch_dict.update({img_ph[i]: __img[i]})
#                                        mini_batch_dict.update({img_truth_ph[i]: __img_truth[i]})
#                                        mini_batch_dict.update({batch_norm_on_ph: False})

#                                        ph_dict.update({img_ph[i]: __img[i],
#                                                        img_truth_ph[i]: __img_truth[i]})

#                                        if train_gen:
#                                            mini_batch_results = sess.run([_tower_preds] +
#                                                                           tower_grads + 
#                                                                           [_discr_preds], 
#                                                                           feed_dict=mini_batch_dict)
#                                        else:
#                                            mini_batch_results = sess.run([_tower_preds] +
#                                                                           tower_grads, 
#                                                                           feed_dict=mini_batch_dict)

#                                    tower_preds_list += [x for x in mini_batch_results[0]]

#                                    for i in range(1, 1+batch_size):
#                                        ph_dict.update({ph: val for ph, val in 
#                                                        zip(tower_grads_ph[j], 
#                                                            mini_batch_results[i])})
#                                    if train_gen:
#                                        discriminator_tower_preds_list += [x for x in mini_batch_results[1+batch_size]]
#                                del mini_batch_dict

#                                #Save outputs occasionally
#                                if counter <= 1 or not counter % save_result_every_n_batches or (counter < 10000 and not counter % 1000) or counter == counter_init:
#                                    try:
#                                        save_input_loc = model_dir+"input-"+str(counter)+".tif"
#                                        save_truth_loc = model_dir+"truth-"+str(counter)+".tif"
#                                        save_output_loc = model_dir+"output-"+str(counter)+".tif"
#                                        Image.fromarray(scale0to1(__img[0]).reshape(cropsize, cropsize).astype(np.float32)).save( save_input_loc )
#                                        Image.fromarray(scale0to1(__img_truth[0]).reshape(cropsize, cropsize).astype(np.float32)).save( save_truth_loc )
#                                        Image.fromarray(scale0to1(tower_preds_list[0]).reshape(cropsize, cropsize).astype(np.float32)).save( save_output_loc )
#                                    except:
#                                        print("Image save failed")

#                                del mini_batch_results

#                                ph_dict.update({learning_rate_ph: learning_rate,
#                                                batch_norm_on_ph: train_batch_norm_on})

#                                #Train the generator
#                                if train_gen:
#                                    sess.run(train_op, feed_dict=ph_dict)
#                                    del ph_dict
#                                else:
#                                    discriminator_tower_losses_list = []
#                                    discriminator_tower_preds_list = []
#                                    discriminator_ph_dict = {}

#                                    #Apply the discriminator to the generated images
#                                    for j in range(increase_batch_size_by_factor):
#                                        mini_batch_dict = {}

#                                        for i in range(batch_size):
#                                            mini_batch_dict.update({pred_ph[i]: tower_preds_list[batch_size*j+i]})
#                                            try:
#                                                prob = 10*rate
#                                                no_flip = np.random.rand() > prob
#                                                label = 1.e-8 if no_flip else 0.85+0.15*np.random.rand()-1.e-8
#                                                if no_flip:
#                                                    adapt = 4*pred_avg**2
#                                                else:
#                                                    adapt = 1.
#                                                adapt = 1.
#                                            except:
#                                                label = 0.5
#                                                pred_avg = 0.5
#                                                adapt = 1.
#                                                print("Numerical error caught")
#                                            mini_batch_dict.update({label_ph[i]: np.reshape(label, (1,))})
#                                            mini_batch_dict.update({adapt_rates_ph[i]: np.reshape(adapt, (1,))})
#                                            discriminator_ph_dict.update({pred_ph[i]: tower_preds_list[batch_size*j+i]})

#                                        mini_batch_results = sess.run([_discriminator_tower_losses, 
#                                                                       _discriminator_tower_preds] +
#                                                                       _discriminator_tower_grads,
#                                                                       feed_dict=mini_batch_dict)

#                                        discriminator_tower_losses_list += [x for x in mini_batch_results[0]]
#                                        discriminator_tower_preds_list += [x for x in mini_batch_results[1]]

#                                        for i in range(2, 2+batch_size):
#                                            discriminator_ph_dict.update({ph: val for ph, val in 
#                                                                          zip(discriminator_tower_grads_ph[batch_size*j+i-2], 
#                                                                              mini_batch_results[i])})

#                                    del tower_preds_list
#                                    del mini_batch_dict

#                                    #print(mini_batch_results[2][0])

#                                    #Apply the discriminator to real images
#                                    for j in range(effective_batch_size):
#                                        mini_batch_dict = {}
#                                        for i in range(batch_size):
#                                            mini_batch_dict.update({pred_ph[i]: np.reshape(tower_ground_truths_list[batch_size*j+i],
#                                                                                           (cropsize, cropsize, 1))})
#                                            try:
#                                                prob = 10*rate
#                                                no_flip = np.random.rand() > prob
#                                                label = 0.85+0.15*np.random.rand()-1.e-8 if no_flip else 1.e-8
#                                                if no_flip:
#                                                    adapt = 4*pred_avg**2
#                                                else:
#                                                    adapt = 1.
#                                                adapt = 1.
#                                            except:
#                                                label = 0.5
#                                                pred_avg_real = 0.5
#                                                adapt = 1.
#                                                print("Numerical error caught")

#                                            mini_batch_dict.update({label_ph[i]: np.reshape(label, (1,))})
#                                            mini_batch_dict.update({adapt_rates_ph[i]: np.reshape(adapt, (1,))})

#                                        mini_batch_results = sess.run([_discriminator_tower_losses, 
#                                                                       _discriminator_tower_preds] +
#                                                                       _discriminator_tower_grads,
#                                                                       feed_dict=mini_batch_dict)

#                                        discriminator_tower_losses_list += [x for x in mini_batch_results[0]]
#                                        discriminator_tower_preds_list += [x for x in mini_batch_results[1]]

#                                        for i in range(2, 2+batch_size):
#                                            discriminator_ph_dict.update({ph: val for ph, val in 
#                                                                          zip(discriminator_tower_grads_ph[
#                                                                              batch_size*j+i-2+effective_batch_size], 
#                                                                              mini_batch_results[i])})
#                                    del mini_batch_dict

#                                    discriminator_ph_dict.update({discriminator_learning_rate_ph: discriminator_learning_rate})

#                                    discrimination_losses = np.reshape(np.asarray(discriminator_tower_losses_list),
#                                                                       (2*effective_batch_size,))
#                                    discriminator_ph_dict.update({discriminator_tower_losses_ph: discrimination_losses})

#                                    gen_losses = [np.reshape(x, (1,)) 
#                                                  for x in discriminator_tower_losses_list[:effective_batch_size]]

#                                    generation_losses = np.reshape(np.asarray(gen_losses), (effective_batch_size,))

#                                    _, discr_loss = sess.run([discriminator_train_op, discriminator_get_loss], 
#                                                             feed_dict=discriminator_ph_dict)

#                                    del discriminator_ph_dict

#                                    gen_loss = np.mean(generation_losses)
#                                    discr_loss = np.mean(discrimination_losses)

#                                    try:
#                                        log.write("Iter: {}, Gen loss: {:.8f}, Discr loss: {:.8f}".format(
#                                            counter, gen_loss, discr_loss))
#                                    except:
#                                        print("Failed to write to log")

#                                    #print("Iter: {}, Gen loss: {:.8f}, Discr loss: {:.8f}".format(
#                                    #      counter, gen_loss, discr_loss))

#                                avg_pred += np.sum(np.asarray(discriminator_tower_preds_list[:effective_batch_size]))
#                                avg_pred_real += np.sum(np.asarray(discriminator_tower_preds_list[effective_batch_size:]))

#                                print("Iter: {}, Preds: {}".format(counter, discriminator_tower_preds_list[:effective_batch_size]))
#                                discr_pred_log.write("Iter: {}, ".format(counter))
#                                for i, p in enumerate(discriminator_tower_preds_list):
#                                    discr_pred_log.write("{}, ".format(p))

#                                    if not np.isfinite(p) and not i:
#                                        print("Numerical issue")
#                                        saver.restore(sess, tf.train.latest_checkpoint(
#                                            model_dir+"model/"))
#                                        counter = save_counter

#                                discr_pred_log.write("\n")

#                                if not counter % val_skip_n:

#                                    tower_preds_list = []
#                                    tower_ground_truths_list = []

#                                    #Generate micrographs
#                                    mini_batch_dict = {}
#                                    for i in range(batch_size):
#                                        __img, __img_truth = sess.run([img_val[i], img_truth_val[i]])
#                                        tower_ground_truths_list.append(__img_truth)
#                                        mini_batch_dict.update({img_ph[i]: __img})
#                                        mini_batch_dict.update({img_truth_ph[i]: __img_truth})
#                                        mini_batch_dict.update({batch_norm_on_ph: False})

#                                    mini_batch_results = sess.run([_tower_preds] +
#                                                                   tower_grads, 
#                                                                   feed_dict=mini_batch_dict)

#                                    tower_preds_list += [x for x in mini_batch_results[0]]

#                                    del mini_batch_dict
#                                    del mini_batch_results

#                                    discriminator_tower_losses_list = []
#                                    _discriminator_tower_preds_list = []

#                                    #Apply discriminator to fake micrographs
#                                    mini_batch_dict = {}
#                                    j = 0
#                                    for i in range(batch_size):
#                                        mini_batch_dict.update({pred_ph[i]: np.reshape(tower_ground_truths_list[batch_size*j+i],
#                                                                                       (cropsize, cropsize, 1))})
#                                        mini_batch_dict.update({label_ph[i]: np.array([0.], dtype=np.float32)})

#                                    mini_batch_results = sess.run([_discriminator_tower_losses, 
#                                                                   _discriminator_tower_preds],
#                                                                   feed_dict=mini_batch_dict)

#                                    discriminator_tower_losses_list += [x for x in mini_batch_results[0]]
#                                    _discriminator_tower_preds_list += [x for x in mini_batch_results[1]]

#                                    #Apply discriminator to real micrographs
#                                    mini_batch_dict = {}
#                                    for i in range(batch_size):
#                                        mini_batch_dict.update({pred_ph[i]: np.reshape(tower_ground_truths_list[batch_size*j+i],
#                                                                                       (cropsize, cropsize, 1))})
#                                        mini_batch_dict.update({label_ph[i]: np.array([1.], dtype=np.float32)})

#                                    mini_batch_results = sess.run([_discriminator_tower_losses, 
#                                                                   _discriminator_tower_preds],
#                                                                   feed_dict=mini_batch_dict)

#                                    del mini_batch_dict

#                                    discriminator_tower_losses_list += [x for x in mini_batch_results[0]]
#                                    _discriminator_tower_preds_list += [x for x in mini_batch_results[1]]

#                                    discr_loss_val = np.mean(np.asarray(discriminator_tower_losses_list))
#                                    gen_loss_val = np.mean(np.asarray(discriminator_tower_losses_list[:batch_size]))

#                                    try:
#                                        val_log.write("Iter: {}, Gen Val: {:.8f}, Disc Val {:.8f}".format(
#                                            counter, gen_loss_val, discr_loss_val))
#                                    except:
#                                        print("Failed to write to val log")

#                                    #print("Iter: {}, Gen loss: {:.8f}, Discr loss: {:.8f}, Gen Val: {:.8f}, Discr Val: {:.8f}".format(
#                                    #      counter, gen_loss, discr_loss, gen_loss_val, discr_loss_val))

#                                if not counter % trainee_switch_skip_n:

#                                    avg_pred /= trainee_switch_skip_n*effective_batch_size
                                    
#                                    pred_avg = b*pred_avg + (1-b)*avg_pred

#                                    real_flip_prob = 0.
#                                    if avg_pred_real:
#                                        avg_pred_real /= trainee_switch_skip_n*effective_batch_size
#                                        avg_pred_real = 1. - avg_pred_real
#                                        pred_avg_real = b*pred_avg_real + (1-b)*avg_pred_real
#                                        real_flip_prob = 0.01

#                                    #print("Gen pred: {}, Discr Pred: {}".format(avg_pred, 1.-avg_pred_real))
#                                    #print("Iter: {}, Training {}, G flip: {}, R flip:{}".format(
#                                        #counter, "gen" if train_gen else "discr", 
#                                        #0.01,
#                                        #real_flip_prob))

#                                    if num_since_change >= max_num_since_training_change:
#                                        num_since_change = 1
#                                        train_gen = not train_gen
#                                    else:
#                                        if avg_pred < 0.3:
#                                            if train_gen:
#                                                num_since_change += 1
#                                            else:
#                                                num_since_change = 0
#                                            train_gen = True
#                                        elif avg_pred > 0.7:
#                                            if not train_gen:
#                                                num_since_change += 1
#                                            else:
#                                                num_since_change = 0
#                                            train_gen = False
#                                        else:
#                                            num_since_change = 0
#                                            train_gen = not train_gen

#                                    avg_pred = 0.
#                                    avg_pred_real = 0.

#                                #train_writer.add_summary(summary, counter)

#                            #Save the model
#                            saver.save(sess, save_path=model_dir+"model/", global_step=counter)
#                            save_counter = counter
#    return 

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
