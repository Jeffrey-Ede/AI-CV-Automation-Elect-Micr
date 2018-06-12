from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import numpy as np
import tensorflow as tf

import cv2
from scipy.misc import imread

import time

import os, random

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

slim = tf.contrib.slim #For depthwise separable strided atrous convolutions

tf.logging.set_verbosity(tf.logging.DEBUG)

features1 = 1#6
features2 = 3#2
features3 = 64
features4 = 1#28
features5 = 2#56
features6 = 7#68

gen_features0 = 32
gen_features1 = 64
gen_features2 = 64
gen_features3 = 32

nin_features1 = 64
nin_features2 = 128
nin_features3 = 256
nin_features4 = 768

nin_features_out1 = 512
nin_features_out2 = 258
nin_features_out3 = 128
nin_features_out4 = 64

num_global_enhancer_blocks = 1
num_local_enhancer_blocks = 1#5
num_discriminator_final_blocks = 4

#data_dir = "X:/Jeffrey-Ede/stills_all/"
data_dir = "E:/stills_hq-mini/"

modelSavePeriod = 12 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
#model_dir = "X:/Jeffrey-Ede/models/gan-multi-gpu-1/"
model_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-10/"

shuffle_buffer_size = 5000
num_parallel_calls = 8
num_parallel_readers = 5
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

weight_decay = 0.0#2.0e-4
initial_learning_rate = 0.001
initial_discriminator_learning_rate = 0.001
num_workers = 1

increase_batch_size_by_factor = 5
effective_batch_size = increase_batch_size_by_factor*batch_size

save_result_every_n_batches = 5000

val_skip_n = 10

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

    concat_axis = 3

    def pad(input, kernel_size):
        padding = tf.constant([[0, 0], 
                               [kernel_size//2, kernel_size//2],
                               [kernel_size//2, kernel_size//2],
                               [0, 0]])
        padded = tf.pad(input, padding, 'REFLECT')
        return padded

    ##Reusable blocks
    def _batch_norm_fn(input):
        batch_norm = tf.contrib.layers.batch_norm(
            input,
            center=True, scale=True,
            is_training=phase,
            fused=True,
            zero_debias_moving_mean=False,
            renorm=False)
        return batch_norm

    def batch_then_activ(input):
        batch_then_activ = _batch_norm_fn(input)
        batch_then_activ = tf.nn.relu6(batch_then_activ)
        return batch_then_activ

    def conv_block_not_sep(input, filters, kernel_size=3, phase=phase):

        conv_block = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=kernel_size,
            padding='SAME')
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
                           extra_batch_norm=True, kernel_size=3):
        
        strided_conv = slim.separable_convolution2d(
            inputs=input,
            num_outputs=filters,
            kernel_size=kernel_size,
            depth_multiplier=1,
            stride=stride,
            data_format='NHWC',
            padding='SAME',
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
            trainable=True,
            scope=None)
        strided_conv = batch_then_activ(strided_conv)

        return strided_conv

    def residual_conv(input, filters):

        residual = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=1,
            strides=2,
            padding="SAME")
        residual = batch_then_activ(residual)

        return residual

    def deconv_block(input, filters):
        '''Transpositionally convolute a feature space to upsample it'''
        
        deconv_block = tf.layers.conv2d_transpose(
            inputs=input,
            filters=filters,
            kernel_size=3,
            strides=2,
            padding='SAME')
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

        nin = strided_conv_block(input, nin_features1, 2, 1)
        
        nin = strided_conv_block(nin, nin_features2, 2, 1)
        
        nin = strided_conv_block(nin, nin_features2, 1, 1)
        nin = strided_conv_block(nin, nin_features3, 2, 1)

        nin = xception_encoding_block_diff(nin, nin_features3, nin_features4)

        for _ in range(num_global_enhancer_blocks):
            nin = xception_middle_block(nin, nin_features4)

        nin = deconv_block(nin, nin_features_out1)

        nin = strided_conv_block(nin, nin_features_out2, 1, 1)
        nin = strided_conv_block(nin, nin_features_out2, 1, 1)
        nin = deconv_block(nin, nin_features_out2)

        nin = strided_conv_block(nin, nin_features_out3, 1, 1)
        nin = deconv_block(nin, nin_features_out3)

        nin = strided_conv_block(nin, nin_features_out4, 1, 1)
        nin = deconv_block(nin, nin_features_out4)

        return nin

    ##Model building
    input_layer = tf.reshape(inputs, [-1, generator_input_size, generator_input_size, channels])

    enc = strided_conv_block(input=input_layer,
                             filters=gen_features0,
                             stride=2,
                             kernel_size = 9)
    enc = strided_conv_block(enc, gen_features1, 1, 1)

    enc += network_in_network(enc)

    for _ in range(num_local_enhancer_blocks):
        enc = xception_middle_block(enc, gen_features2)

    enc = deconv_block(enc, gen_features3)
    enc = strided_conv_block(enc, gen_features3, 1, 1)

    #Create final image with 1x1 convolutions
    final = conv_block_not_sep(enc, 1)

    #Image values will be between 0 and 1
    output = tf.clip_by_value(
        final,
        clip_value_min=0.,
        clip_value_max=1.)

    return output

def discriminator_architecture(inputs, ground_truth, phase=False, params=None, gen_loss=0.):
    """Discriminates between real and fake data"""

    #phase = mode == tf.estimator.ModeKeys.TRAIN #phase is true during training
    concat_axis = 3

    ##Reusable blocks
    def _batch_norm_fn(input):
        batch_norm = tf.contrib.layers.batch_norm(
            input,
            center=True, scale=True,
            is_training=phase,
            fused=True,
            zero_debias_moving_mean=False,
            renorm=False)
        return batch_norm

    ##Reusable blocks
    def _batch_norm_fn(input):
        batch_norm = tf.contrib.layers.batch_norm(
            input,
            center=True, scale=True,
            is_training=phase,
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

        conv_block = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=kernel_size,
            padding="SAME")
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

        residual = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=1,
            strides=2,
            padding="SAME")
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

    def shared_flow(input):

        shared = xception_encoding_block_diff(input, features2, features3)
        shared = xception_encoding_block_diff(shared, features3, features4)
        
        shared = xception_encoding_block(shared, features5)
        shared = xception_encoding_block(shared, features6)

        return shared

    def terminating_fc(input):

        fc = _batch_norm_fn(input)
        fc = tf.contrib.layers.flatten(fc)
        fc = tf.contrib.layers.fully_connected(inputs=fc,
                                               num_outputs=1,
                                               activation_fn=tf.nn.tanh)
        return fc

    '''Model building'''

    input_layer = tf.reshape(inputs, [-1, cropsize, cropsize, channels])

    with tf.variable_scope("small-start", reuse=False):
        small = tf.random_crop(
            input_layer,
            size=(batch_size, cropsize//4, cropsize//4, 1))
        small = strided_conv_block(small, features1, 1, 1)
        small = strided_conv_block(small, features2, 1, 1)

    with tf.variable_scope("medium-start", reuse=False):
        medium = tf.random_crop(
            input_layer,
            size=(batch_size, cropsize//2, cropsize//2, 1))
        medium = tf.image.resize_images(medium, 
                                        (cropsize//4, cropsize//4), 
                                        method=tf.image.ResizeMethod.AREA)
        medium = strided_conv_block(medium, features1, 1, 1)
        medium = strided_conv_block(medium, features2, 1, 1)

    with tf.variable_scope("large-start", reuse=False):
        large = input_layer
        large = tf.image.resize_images(large, 
                                        (cropsize//4, cropsize//4), 
                                        method=tf.image.ResizeMethod.AREA)
        large = strided_conv_block(large, features1, 1, 1)
        large = strided_conv_block(large, features2, 1, 1)

        #s1 = shared_flow(small)
        #s2 = shared_flow(small)
        #assert s1 is s2

    with tf.variable_scope("shared") as shared_scope:
        small = shared_flow(small)
    with tf.variable_scope(shared_scope, reuse=True):
        medium = shared_flow(medium)
    with tf.variable_scope(shared_scope, reuse=True):
        large = shared_flow(large)

    with tf.variable_scope("small-end", reuse=False):
        for _ in range(num_discriminator_final_blocks):
            small = xception_middle_block(small, features6)
        small = terminating_fc(small)

    with tf.variable_scope("medium-end", reuse=False):
        for _ in range(num_discriminator_final_blocks):
            medium = xception_middle_block(medium, features6)
        medium = terminating_fc(medium)

    with tf.variable_scope("large", reuse=False):
        for _ in range(num_discriminator_final_blocks):
            large = xception_middle_block(large, features6)
        large = terminating_fc(large)

    output = small + medium + large
    output = tf.clip_by_value(output,
                              clip_value_min=0.,
                              clip_value_max=1.)

    return output

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
            raise RuntimeError(
                'Global step should be created to use StepCounterHook.')

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
        def _model_fn(features, mode=None, params=None, losses_des=None):
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
                    with tf.name_scope('generator_tower_%d' % i) as name_scope:
                        with tf.device(device_setter):
                            grads, preds = _generator_tower_fn(
                                is_training, tower_features[i], losses_des[i])

                            tower_grads.append(grads)
                            tower_preds.append(preds)
                            if i == 0:
                                # Only trigger batch_norm moving mean and variance update from
                                # the 1st tower. Ideally, we should grab the updates from all
                                # towers but these stats accumulate extremely fast so we can
                                # ignore the other stats from the other towers without
                                # significant detriment.
                                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)

            _tower_preds_tmp = tf.stack(preds)
        
            return [_tower_preds_tmp, update_ops] + tower_grads

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
                    with tf.name_scope('discriminator_tower_%d' % i) as name_scope:
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

def _generator_tower_fn(is_training, feature, loss):
    """Build computation tower.
        Args:
        is_training: true if is training graph.
        feature: a Tensor.
        Returns:
        A tuple with the loss for the tower, the gradients and parameters, and
        predictions.
    """

    #phase = tf.estimator.ModeKeys.TRAIN if is_training else tf.estimator.ModeKeys.EVAL
    output = generator_architecture(feature[0], is_training)

    model_params = tf.trainable_variables()

    loss += weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in model_params])
    
    loss_reshaped = tf.reshape(loss, (1,))
    tower_grad = tf.gradients(loss_reshaped, model_params)

    return tower_grad, output

def _discriminator_tower_fn(is_training, feature, ground_truth):
    """Build computation tower.
        Args:
        is_training: true if is training graph.
        feature: a Tensor.
        Returns:
        A tuple with the loss for the tower, the gradients and parameters, and
        predictions.
    """

    #phase = tf.estimator.ModeKeys.TRAIN if is_training else tf.estimator.ModeKeys.EVAL
    output = discriminator_architecture(feature, ground_truth, is_training)

    model_params = tf.trainable_variables()

    small_const = 1.e-6
    tower_loss = -tf.cond(ground_truth[0] > 0., 
                          lambda: tf.log(1.-output+small_const), 
                          lambda: tf.log(output+small_const))
    tower_loss += weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in model_params])

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


def gen_lq(img, img_type=np.float32):
    '''Generate low quality image'''

    choice = np.random.randint(0, 3)
    mark = 0.5

    if choice == 0: #Black rectangle in middle
        half_side1 = np.random.randint(0, cropsize//2+1)//2
        half_side2 = np.random.randint(0, cropsize//2+1)//2
        img[half_side1:(cropsize-half_side1), half_side2:(cropsize-half_side2)] = mark
    #if choice == 1: #Completely random
    #    frac = np.random.rand()
    #    mask = np.random.uniform(0, 1, img.shape)
    #    mask[mask>frac] = mark
    #    img = np.ceil(mask)*img
    if choice == 1: #Black side
        img = flip_rotate(img)
        side = np.random.randint(0, cropsize+1)
        img[0:side, 0:side] = mark
    #if choice == 3: #Splodges from thresholding image
    #    thresh = np.random.rand()
    #    if np.random.rand() > 0.5:
    #        mask = img
    #        mask[img<thresh] = mark
    #    else:
    #        mask = img
    #        mask[img>1.-thresh] = mark
    #    img[flip_rotate(mask)>0.] = mark
    if choice == 2: #Black corner square
        half_side1 = np.random.randint(0, cropsize//2+1)//2
        half_side2 = np.random.randint(0, cropsize//2+1)//2
        img[:half_side1, :half_side2] = mark

    return flip_rotate(img.astype(np.float32))


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

def preprocess(img):

    img[np.isnan(img)] = 0.5
    img[np.isinf(img)] = 0.5

    return scale0to1(flip_rotate(img))

def record_parser(record):
    """Parse files and generate lower quality images from them"""

    img = preprocess(load_image(record))
    lq = gen_lq(img)

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
        dataset = dataset.map(reshaper, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

        iter = dataset.make_one_shot_iterator()
        img_batch = iter.get_next()

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

def get_experiment_fn(data_dir,
                      num_gpus,
                      variable_strategy):
    """Returns an experiment function
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
        """Generates a 'Unique Identifier' based on all internal fields.
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

def _train_op(tower_losses_ph, 
              variable_strategy, 
              update_ops, 
              learning_rate_ph, 
              **kwargs):

    with tf.variable_scope("generator"):

        tower_losses = tf.unstack(tower_losses_ph, effective_batch_size)
    
        tower_params = tf.trainable_variables()
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

            loss = tf.reduce_mean(tower_losses, name='loss')
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_ph, beta1=0.5)

        # Create single grouped train op
        train_op = [optimizer.apply_gradients(gradvars, global_step=global_step)]

        train_op.extend(update_ops)
        train_op = tf.group(*train_op)

    return train_op, loss


def _discriminator_train_op(discriminator_tower_losses_ph, 
                            discriminator_variable_strategy, 
                            discriminator_update_ops, 
                            discriminator_learning_rate_ph, 
                            **kwargs):

    with tf.variable_scope("discriminator"):
        discriminator_tower_losses = tf.unstack(discriminator_tower_losses_ph, 2*effective_batch_size)

        discriminator_tower_params = tf.trainable_variables()
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
            discriminator_optimizer = tf.train.AdamOptimizer(
                learning_rate=discriminator_learning_rate_ph, beta1=0.5)

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

                img, img_truth = input_fn(data_dir, 'train', batch_size, num_gpus)
                img_val, img_val_truth = input_fn(data_dir, 'val', batch_size, num_gpus)

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

                    _losses_descended_ph = [tf.placeholder(tf.float32, 
                                                           shape=(1,),
                                                           name="loss_descended")
                                           for _ in ____img]

                    is_training = True
                    generator_model_fn = get_model_fn(
                        num_gpus, variable_strategy, num_workers, component="generator")
                    discriminator_model_fn = get_model_fn(
                        num_gpus, variable_strategy, num_workers, component="discriminator")

                    print("Dataflow established")

                    #########################################################################################

                    results = generator_model_fn(img_ph, mode=is_training, 
                                                 params=hparams, losses_des=_losses_descended_ph)
                    _tower_preds = results[0]
                    update_ops = results[1]
                    tower_grads = results[2:(2+batch_size)]

                    tower_losses_ph = tf.placeholder(tf.float32, shape=(effective_batch_size,), 
                                                     name='tower_losses')
                    learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')

                    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                    temp = set(tf.all_variables())

                    mini_batch_dict = {}
                    for i in range(batch_size):
                        _img, _img_truth = sess.run([img[i], img_truth[i]])
                        mini_batch_dict.update({img_ph[i]: _img})
                        mini_batch_dict.update({img_truth_ph[i]: _img_truth})
                        mini_batch_dict.update({_losses_descended_ph[i]: 
                                                np.array([1.23], dtype=np.float32)})

                    gradvars_pry, preds = sess.run([tower_grads, _tower_preds], 
                                                   feed_dict=mini_batch_dict)
                    del mini_batch_dict

                    tower_grads_ph = [[tf.placeholder(tf.float32, shape=t.shape, name='tower_grads') 
                                       for t in gradvars_pry[0]] 
                                       for _ in range(effective_batch_size)]
                    del gradvars_pry
            
                    train_op, get_loss = _train_op(tower_losses_ph,
                                                    variable_strategy, update_ops, 
                                                    learning_rate_ph,
                                                    _tower_grads=tower_grads_ph)

                    print("Generator flows established")

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
                    gradvars_pry = sess.run(_discriminator_tower_grads, feed_dict=mini_batch_dict)
                    del preds
                    del mini_batch_dict

                    discriminator_tower_grads_ph = [[
                        tf.placeholder(tf.float32, shape=t.shape, name='discriminator_tower_grads') 
                        for t in gradvars_pry[0]] for _ in range(2*effective_batch_size)]
                    del gradvars_pry
            
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

                    learning_rate = initial_learning_rate
                    discriminator_learning_rate = initial_discriminator_learning_rate

                    counter = 0
                    cycleNum = 0
                    while True:
                        cycleNum += 1
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
                            ph_dict = {}
                            for j in range(increase_batch_size_by_factor):

                                mini_batch_dict = {}
                                for i in range(batch_size):
                                    __img, __img_truth = sess.run([img[i], img_truth[i]])
                                    tower_ground_truths_list.append(__img_truth)
                                    mini_batch_dict.update({img_ph[i]: __img})
                                    mini_batch_dict.update({img_truth_ph[i]: __img_truth})

                                mini_batch_results = sess.run([_tower_preds] +
                                                               tower_grads, 
                                                               feed_dict=mini_batch_dict)

                                tower_preds_list += [x for x in mini_batch_results[0]]

                                for i in range(1, 1+batch_size):
                                    ph_dict.update({ph: val for ph, val in 
                                                    zip(tower_grads_ph[j], 
                                                        mini_batch_results[i])})

                                del mini_batch_dict
                                del mini_batch_results

                            ph_dict.update({learning_rate_ph: learning_rate,
                                            img_ph[0]: __img,
                                            img_truth_ph[0]: __img_truth})

                            discriminator_tower_losses_list = []
                            discriminator_tower_preds_list = []
                            discriminator_ph_dict = {}

                            #Apply the discriminator to the generated images
                            for j in range(increase_batch_size_by_factor):
                                mini_batch_dict = {}

                                for i in range(batch_size):
                                    mini_batch_dict.update({pred_ph[i]: tower_preds_list[batch_size*j+i]})
                                    mini_batch_dict.update({label_ph[i]: np.array([1.], dtype=np.float32)})
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

                            #Apply the discriminator to real images
                            for j in range(effective_batch_size):
                                mini_batch_dict = {}
                                for i in range(batch_size):
                                    mini_batch_dict.update({pred_ph[i]: np.reshape(tower_ground_truths_list[batch_size*j+i],
                                                                                   (cropsize, cropsize, 1))})
                                    mini_batch_dict.update({label_ph[i]: np.array([0.], dtype=np.float32)})

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

                            discriminator_ph_dict.update({discriminator_learning_rate_ph: discriminator_learning_rate})

                            #if counter <= 1 or not counter % save_result_every_n_batches:
                            #    try:
                            #        save_input_loc = model_dir+"input-"+str(counter)+".tif"
                            #        save_output_loc = model_dir+"output-"+str(counter)+".tif"
                            #        Image.fromarray(__img.reshape(cropsize, cropsize).astype(np.float32)).save( save_input_loc )
                            #        Image.fromarray(__img_truth.reshape(cropsize, cropsize).astype(np.float32)).save( save_output_loc )
                            #    except:
                            #        print("Image save failed")

                            gen_losses = [-x for x in discriminator_tower_losses_list[:effective_batch_size]]
                            ph_dict.update({ph: np.reshape(val, (batch_size,)) for ph, val in zip(_losses_descended_ph, gen_losses)})

                            generation_losses = np.reshape(np.asarray(gen_losses), (effective_batch_size,))
                            ph_dict.update({tower_losses_ph: generation_losses})

                            discrimination_losses = np.reshape(np.asarray(discriminator_tower_losses_list),
                                                               (2*effective_batch_size,))
                            discriminator_ph_dict.update({discriminator_tower_losses_ph: discrimination_losses})

                            #Train the discriminator
                            _, discr_loss = sess.run([discriminator_train_op, discriminator_get_loss],
                                                              feed_dict=discriminator_ph_dict)
                            del discriminator_ph_dict

                            #Train the generator
                            _, gen_loss = sess.run([train_op, get_loss], feed_dict=ph_dict)
                            del ph_dict

                            #Save outputs occasionally
                            if counter <= 1 or not counter % save_result_every_n_batches:
                                try:
                                    save_input_loc = model_dir+"input-"+str(counter)+".tif"
                                    save_output_loc = model_dir+"output-"+str(counter)+".tif"
                                    Image.fromarray(__img.reshape(cropsize, cropsize).astype(np.float32)).save( save_input_loc )
                                    Image.fromarray(__img_truth.reshape(cropsize, cropsize).astype(np.float32)).save( save_output_loc )
                                except:
                                    print("Image save failed")

                            try:
                                log.write("Iter: {}, Gen loss: {:.8f}, Discr loss: {:.8f}".format(
                                    counter, gen_loss, discr_loss))
                            except:
                                print("Failed to write to log")

                            if not counter % val_skip_n:

                                #Generate micrographs
                                mini_batch_dict = {}
                                for i in range(batch_size):
                                    __img, __img_truth = sess.run([img[i], img_truth[i]])
                                    tower_ground_truths_list.append(__img_truth)
                                    mini_batch_dict.update({img_ph[i]: __img})
                                    mini_batch_dict.update({img_truth_ph[i]: __img_truth})

                                mini_batch_results = sess.run([_tower_preds] +
                                                               tower_grads, 
                                                               feed_dict=mini_batch_dict)

                                tower_preds_list += [x for x in mini_batch_results[0]]

                                for i in range(1, 1+batch_size):
                                    ph_dict.update({ph: val for ph, val in 
                                                    zip(tower_grads_ph[j], 
                                                        mini_batch_results[i])})
                                    j += 1

                                del mini_batch_dict
                                del mini_batch_results

                                discriminator_tower_losses_list = []
                                discriminator_tower_preds_list = []
                                discriminator_ph_dict = {}

                                #Apply discriminator to fake micrographs
                                mini_batch_dict = {}
                                for i in range(batch_size):
                                    mini_batch_dict.update({pred_ph[i]: np.reshape(tower_ground_truths_list[batch_size*j+i],
                                                                                   (cropsize, cropsize, 1))})
                                    mini_batch_dict.update({label_ph[i]: np.array([0.], dtype=np.float32)})

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

                                #Apply discriminator to real micrographs
                                mini_batch_dict = {}
                                for i in range(batch_size):
                                    mini_batch_dict.update({pred_ph[i]: np.reshape(tower_ground_truths_list[batch_size*j+i],
                                                                                   (cropsize, cropsize, 1))})
                                    mini_batch_dict.update({label_ph[i]: np.array([0.], dtype=np.float32)})

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

                                gen_losses = [-x for x in discriminator_tower_losses_list[:effective_batch_size]]
                                ph_dict.update({ph: np.reshape(val, (batch_size,)) 
                                                for ph, val in zip(_losses_descended_ph, gen_losses)})

                                generation_losses = np.reshape(np.asarray(gen_losses), (effective_batch_size,))
                                ph_dict.update({tower_losses_ph: generation_losses})

                                discrimination_losses = np.reshape(np.asarray(discriminator_tower_losses_list),
                                                                    (2*effective_batch_size,))
                                discriminator_ph_dict.update({discriminator_tower_losses_ph: discrimination_losses})

                                #Train the discriminator
                                _, discr_loss_val = sess.run([discriminator_train_op, discriminator_get_loss],
                                                                    feed_dict=discriminator_ph_dict)
                                del discriminator_ph_dict

                                #Train the generator
                                _, gen_loss_val = sess.run([train_op, get_loss], feed_dict=ph_dict)
                                del ph_dict

                                try:
                                    val_log.write("Iter: {}, Gen Val: {:.8f}, Disc Val".format(
                                        counter, gen_loss_val, discr_loss_val))
                                except:
                                    print("Failed to write to val log")

                                print("Iter: {}, Gen loss: {:.8f}, Discr loss: {:.8f}, Gen Val: {:.8f}, Discr Val: {:.8f}".format(
                                      counter, gen_loss, discr_loss, gen_loss_val, discr_loss_val))
                            else:
                                print("Iter: {}, Gen loss: {:.8f}, Discr loss: {:.8f}".format(
                                    counter, gen_loss, discr_loss))

                            #train_writer.add_summary(summary, counter)

                        #Save the model
                        saver.save(sess, save_path=model_dir+"model/", global_step=counter)
                        #tf.saved_model.simple_save(
                        #    session=sess,
                        #    export_dir=model_dir+"model-"+str(counter)+"/",
                        #    inputs={"img": img[0][0]},
                        #    outputs={"prediction": prediction})
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




