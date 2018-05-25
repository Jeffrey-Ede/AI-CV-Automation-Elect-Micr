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

features0 = 32
features1 = 64
features2 = 128
features3 = 192
features4 = 384
features5 = 384

gen_features1 = 444
gen_features2 = 444
gen_features3 = 444 #Middle flow
gen_features4 = 444
gen_features5 = 444

extra_gen_middle_blocks = 8 #_ + 1 in total
extra_dis_middle_blocks = 6 #_ + 1 in total

data_dir = "E:/stills_hq-mini/"

modelSavePeriod = 6 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-7/"

shuffle_buffer_size = 5000
num_parallel_calls = 8
num_parallel_readers = 5
prefetch_buffer_size = 20
batch_size = 2
num_gpus = 2

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
log_every = 1 #Log every _ examples
cumProbs = np.array([]) #Indices of the distribution plus 1 will be correspond to means

numMeans = 64 // batch_size
scaleMean = 4 #Each means array index increment corresponds to this increase in the mean
numDynamicGrad = 1 #Number of gradients to calculate for each possible mean when dynamically updating training
lossSmoothingBoxcarSize = 5

channels = 1 #Greyscale input image

#Sidelength of images to feed the neural network
cropsize = 512
height_crop = width_crop = cropsize

#hparams = experiment_hparams(train_batch_size=batch_size, eval_batch_size=16)

weight_decay = 0.0#2.0e-4
initial_learning_rate = 0.001
num_workers = 1

increase_batch_size_by_factor = 5
effective_batch_size = increase_batch_size_by_factor*batch_size

generator_input_size = 128

perceptual_loss_lambda = 0.1 #Scale perceptual loss by _ before addition to contextual loss

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

def generator_architecture(inputs, ground_truth, phase=False, params=None):
    """Generates fake data to try and fool the discrimator"""

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

    def batch_then_activ(input):

        batch_then_activ = _batch_norm_fn(input)
        batch_then_activ = tf.nn.relu(batch_then_activ)

        return batch_then_activ

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
            trainable=True,
            scope=None)
        strided_conv = batch_then_activ(strided_conv)

        return strided_conv

    def deconv_block(input, filters):
        '''Transpositionally convolute a feature space to upsample it'''
        
        deconv_block = tf.layers.conv2d_transpose(
            inputs=input,
            filters=filters,
            kernel_size=3,
            strides=2,
            padding="same")
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

    ##Model building
    input_layer = tf.reshape(inputs, [-1, generator_input_size, generator_input_size, channels])

    enc1 = xception_encoding_block(input_layer, gen_features1)
    enc2 = xception_encoding_block(enc1, gen_features2)

    middle_flow = xception_middle_block(enc2, gen_features3)
    for _ in range(extra_gen_middle_blocks):
        middle_flow = xception_middle_block(middle_flow, gen_features3)
    middle_flow = deconv_block(middle_flow, gen_features2)

    concat1 = tf.concat(
        values=[enc1, middle_flow],
        axis=concat_axis)
    
    dec0 = conv_block(concat1, gen_features2)
    dec = conv_block(dec0, gen_features2)
    dec = conv_block(dec0, gen_features2)
    dec += dec0
    dec = deconv_block(middle_flow, gen_features3)

    res_input = conv_block_not_sep(input, gen_features3, 1)
    concat2 = tf.concat(
        values=[dec, res_input],
        axis=concat_axis)
    
    dec0 = conv_block(concat2, gen_features3)
    dec = conv_block(dec0, gen_features3)
    dec = conv_block(dec0, gen_features3)
    dec += dec0

    deconv = deconv_block(dec, gen_features4)
    dec0 = conv_block(deconv, gen_features4)
    dec = conv_block(dec0, gen_features4)
    dec = conv_block(dec0, features4)
    dec += dec0

    deconv = deconv_block(dec, gen_features5)
    dec0 = conv_block(deconv, gen_features5)
    dec = conv_block(dec0, gen_features5)
    dec = conv_block(dec0, gen_features5)
    dec += dec0

    #Create final image with 1x1 convolutions
    final = conv_block_not_sep(deconv0, 1)

    #Image values will be between 0 and 1
    output = tf.clip_by_value(
        final,
        clip_value_min=0.,
        clip_value_max=1.,
        name='clipper')

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

    '''Model building'''
    input_layer = tf.reshape(inputs, [-1, cropsize, cropsize, channels])

    #Encoding block 0
    cnn0 = conv_block(
        input=input_layer, 
        filters=features0)
    cnn0_last = conv_block(
        input=cnn0, 
        filters=features0)
    cnn0_strided = strided_conv_block(
        input=cnn0_last,
        filters=features0,
        stride=2)

    residual0 = residual_conv(input_layer, features0)
    cnn0_strided += residual0

    #Encoding block 1
    cnn1 = conv_block(
        input=cnn0_strided, 
        filters=features1)
    cnn1_last = conv_block(
        input=cnn1, 
        filters=features1)
    cnn1_strided = strided_conv_block(
        input=cnn1_last,
        filters=features1,
        stride=2)

    residual1 = residual_conv(cnn0_strided, features1)
    cnn1_strided += residual1

    #Encoding block 2
    cnn2 = conv_block(
        input=cnn1_strided,
        filters=features2)
    cnn2_last = conv_block(
        input=cnn2,
        filters=features2)
    cnn2_strided = strided_conv_block(
        input=cnn2_last,
        filters=features2,
        stride=2)

    residual2 = residual_conv(cnn1_strided, features2)
    cnn2_strided += residual2

    #Encoding block 3
    cnn3 = conv_block(
        input=cnn2_strided,
        filters=features3)
    cnn3_last = conv_block(
        input=cnn3,
        filters=features3)
    cnn3_strided = strided_conv_block(
        input=cnn3_last,
        filters=features3,
        stride=2)

    residual3 = residual_conv(cnn2_strided, features3)
    cnn3_strided += residual3

    #Encoding block 4
    cnn4 = conv_block(
        input=cnn3_strided,
        filters=features4)
    cnn4_last = conv_block(
        input=cnn4,
        filters=features4)
    cnn4_strided = strided_conv_block(
        input=cnn4_last,
        filters=features4,
        stride=2)

    residual4 = residual_conv(cnn3_strided, features4)
    cnn4_strided += residual4

    #Encoding block 5
    cnn5 = conv_block(
        input=cnn4_strided,
        filters=features5)
    cnn5 = conv_block(
        input=cnn5,
        filters=features5)
    cnn5_last = conv_block(
        input=cnn5,
        filters=features5)

    cnn5_last += cnn4_strided

    for _ in range(extra_dis_middle_blocks):
        cnn5_last = xception_middle_block(cnn5_last, features5)

    fc = _batch_norm_fn(cnn5_last)
    fc = tf.tanh(fc)
    fc = tf.layers.dense(
        inputs=fc,
        units=1)

    output = tf.clip_by_value(
        fc,
        clip_value_min=0.,
        clip_value_max=1.,
        name='discriminator_clipper')

    return fc

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
            tower_mses = []

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
                            loss, grads, preds, mse = _generator_tower_fn(
                                is_training, tower_features[i], tower_labels[i])

                            tower_losses.append(loss)
                            tower_grads.append(grads)
                            tower_preds.append(preds)
                            tower_mses.append(mse)
                            if i == 0:
                                # Only trigger batch_norm moving mean and variance update from
                                # the 1st tower. Ideally, we should grab the updates from all
                                # towers but these stats accumulate extremely fast so we can
                                # ignore the other stats from the other towers without
                                # significant detriment.
                                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)


            _tower_losses_tmp = tf.tuple(tower_losses)
            _tower_mses_tmp = tf.tuple(tower_mses)
            _tower_preds_tmp = tf.stack(preds)
        
            return [_tower_losses_tmp, _tower_preds_tmp, _tower_mses_tmp, update_ops] + _tower_grads

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
                            loss, grads, preds, params = _discriminator_tower_fn(
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

def _generator_tower_fn(is_training, feature, ground_truth):
    """Build computation tower.
        Args:
        is_training: true if is training graph.
        feature: a Tensor.
        Returns:
        A tuple with the loss for the tower, the gradients and parameters, and
        predictions.
    """

    #phase = tf.estimator.ModeKeys.TRAIN if is_training else tf.estimator.ModeKeys.EVAL
    output = architecture(feature[0], ground_truth[0], is_training)

    model_params = tf.trainable_variables()

    tower_pred = output

    out = tf.reshape(output, [-1, cropsize, cropsize, channels])
    truth = tf.reshape(ground_truth[0], [-1, cropsize, cropsize, channels])
    
    mse = tf.reduce_mean(tf.losses.mean_squared_error(out, truth))
    tower_loss = tf.cond(mse < 0.001, lambda: 1000.*mse, lambda: tf.sqrt(1000.*mse))

    #tower_loss += 1.0-tf_ssim(out, truth) #Don't need to unstack for batch size of 2

    tower_loss += weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in model_params])

    tower_grad = tf.gradients(tower_loss, model_params)

    return tower_loss, tower_grad, tower_pred, mse

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
    output = architecture(feature[0], ground_truth[0], is_training)

    model_params = tf.trainable_variables()

    tower_pred = output

    tower_loss = weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in model_params])

    tower_grad = tf.gradients(tower_loss, model_params)

    return tower_loss, tower_grad, tower_pred


def get_scale():
    return 25.+np.random.exponential(75.)


def gen_lq(img, scale, img_type=np.float32):
    '''Generate low quality image'''

    #Ensure that the seed is random
    np.random.seed(int(np.random.rand()*(2**32-1)))

    #Adjust the image scale so that the image has the
    # correct average counts
    lq = np.random.poisson( img * scale )

    return scale0to1(lq).astype(img_type)


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

def preprocess(img):

    img[np.isnan(img)] = 0.5
    img[np.isinf(img)] = 0.5

    return scale0to1(flip_rotate(img))


def record_parser(record):
    """Parse files and generate lower quality images from them"""

    img = load_image(record)
    img = preprocess(img)

    lq = gen_lq(img, scale=get_scale())
    rescaled_img = (np.mean(lq)/np.mean(img))*img

    ##TODO: Sketch generator rather than lq generator

    return lq, rescaled_img


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

        if num_shards <= 1:
            # No GPU available or only 1 GPU.
            return [img_batch], [img_batch]
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
              tower_mses_ph, 
              variable_strategy, 
              update_ops, 
              learning_rate_ph, 
              **kwargs):

    tower_losses = tf.unstack(tower_losses_ph, effective_batch_size)
    tower_mses = tf.unstack(tower_mses_ph, effective_batch_size)
    #tower_losses = [tower_loss_ph for tower_loss_ph in tower_losses_ph]
    #tower_mses = [tower_mse_ph for tower_mse_ph in tower_mses_ph]

    #tower_grads = [ for tower_loss in tower_losses]
    
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
        loss_mse = tf.reduce_mean(tower_mses, name='loss_mse')
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_ph, beta1=0.2)
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_ph, 
                                               momentum=0.9, 
                                               use_nesterov=True)

    # Create single grouped train op
    train_op = [optimizer.apply_gradients(gradvars, global_step=global_step)]

    train_op.extend(update_ops)
    train_op = tf.group(*train_op)

    return train_op, loss, loss_mse


def _discriminator_train_op(discriminator_tower_losses_ph, 
                            discriminator_variable_strategy, 
                            discriminator_update_ops, 
                            discriminator_learning_rate_ph, 
                            **kwargs):

    discriminator_tower_losses = tf.unstack(tower_losses_ph, effective_batch_size)

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
    consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
    with tf.device(consolidation_device):

        discriminator_loss = tf.reduce_mean(tower_losses, name='loss')
        #discriminator_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_ph, beta1=0.2)
        discriminator_optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate_ph,
            momentum=0.9,
            use_nesterov=True)

    # Create single grouped train op
    discriminator_train_op = [
        discriminator_optimizer.apply_gradients(discriminator_gradvars, 
                                  global_step=discriminator_global_step)]

    discriminator_train_op.extend(update_ops)
    discriminator_train_op = tf.group(*train_op)

    return discriminator_train_op, discriminator_loss


def main(job_dir, data_dir, variable_strategy, num_gpus, log_device_placement, 
         num_intra_threads, **hparams):

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

                    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                    #sess.run( tf.global_variables_initializer())
                    temp = set(tf.all_variables())

                    ____img, ____img_truth = sess.run([img, img_truth])
                    img_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img') 
                              for i in ____img]
                    img_truth_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img_truth') 
                                    for i in ____img_truth]
                    del ____img
                    del ____img_truth

                    is_training = True
                    generator_model_fn = get_model_fn(
                        num_gpus, variable_strategy, num_workers, component="generator")
                    discriminator_model_fn = get_model_fn(
                        num_gpus, variable_strategy, num_workers, component="discriminator")

                    #########################################################################################

                    results = generator_model_fn(img_ph, img_truth_ph, mode=is_training, params=hparams)
                    _tower_losses = results[0]
                    _tower_preds = results[1]
                    _tower_mses = results[2]
                    update_ops = results[3]
                    tower_grads = results[4:(4+batch_size)]

                    tower_losses_ph = tf.placeholder(tf.float32, shape=(effective_batch_size,), 
                                                     name='tower_losses')
                    tower_mses_ph = tf.placeholder(tf.float32, shape=(effective_batch_size,), 
                                                   name='tower_mses')
                    learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')

                    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                    temp = set(tf.all_variables())

                    mini_batch_dict = {}
                    for i in range(batch_size):
                        _img, _img_truth = sess.run([img[i], img_truth[i]])
                        mini_batch_dict.update({img_ph[i]: _img})
                        mini_batch_dict.update({img_truth_ph[i]: _img_truth})
                    gradvars_pry, preds = sess.run([tower_grads, _tower_preds], feed_dict=mini_batch_dict)
                    del mini_batch_dict

                    tower_grads_ph = [[tf.placeholder(tf.float32, shape=t.shape, name='tower_grads') 
                                       for t in gradvars_pry[0]] 
                                       for _ in range(effective_batch_size)]
                    del gradvars_pry
            
                    train_op, get_loss, get_loss_mse = _train_op(tower_losses_ph, tower_mses_ph, 
                                                         variable_strategy, update_ops, learning_rate_ph,
                                                         _tower_grads=tower_grads_ph)

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
                    _discriminator_tower_params = discriminator_results[(3+batch_size):]

                    discriminator_tower_losses_ph = tf.placeholder(
                        tf.float32, shape=(2*effective_batch_size,), name='discriminator_tower_losses')
                    discriminator_learning_rate_ph = tf.placeholder(
                        tf.float32, name='discriminator_learning_rate')

                    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                    temp = set(tf.all_variables())

                    mini_batch_dict = {}
                    for i in range(batch_size):
                        mini_batch_dict.update({ph: val for ph, val in zip(pred_ph, preds)}
                        mini_batch_dict.update({ph: val for ph, val in zip(pred_ph, np.random.rand())}
                    gradvars_pry = sess.run(_discriminator_tower_grads, feed_dict=mini_batch_dict)
                    del preds
                    del mini_batch_dict

                    discriminator_tower_grads_ph = [[
                        tf.placeholder(tf.float32, shape=t.shape, name='discriminator_tower_grads') 
                        for t in gradvars_pry[0]] for _ in range(effective_batch_size)]
                    del gradvars_pry
            
                    discriminator_train_op, discriminator_get_loss, discriminator_get_loss_mse = _discriminator_train_op(
                        discriminator_tower_losses_ph,
                        variable_strategy,
                        _discriminator_update_ops,
                        discriminator_learning_rate_ph,
                        _tower_grads=discriminator_tower_grads_ph)

                    #########################################################################################

                    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

                    train_writer = tf.summary.FileWriter( logDir, sess.graph )

                    #print(tf.all_variables())
                    saver = tf.train.Saver()

                    counter = 0
                    cycleNum = 0
                    while True:
                        cycleNum += 1
                        #Train for a couple of hours
                        time0 = time.time()
                    
                        with open(model_dir+"learning_rate.txt") as lrf:
                            try:
                                learning_rate = [float(line) for line in lrf]
                                learning_rate = np.float32(learning_rate[0])
                                print("Using learning rate: {}".format(learning_rate))
                            except:
                                pass

                        generations = []
                        while time.time()-time0 < modelSavePeriod:
                            counter += 1

                            #Apply the generator
                            tower_losses_list = []
                            tower_preds_list = []
                            tower_mses_list = []
                            tower_ground_truths_list = []
                            ph_dict = {}
                            j = 0
                            for _ in range(increase_batch_size_by_factor):
                                mini_batch_dict = {}
                                for i in range(batch_size):
                                    __img, __img_truth = sess.run([img[i], img_truth[i]])
                                    tower_ground_truths_list += [img for img in ___img_truth]
                                    mini_batch_dict.update({img_ph[i]: __img})
                                    mini_batch_dict.update({img_truth_ph[i]: __img_truth})

                                mini_batch_results = sess.run([_discriminator_tower_losses, 
                                                               _discriminator_tower_preds, 
                                                               _discriminator_tower_mses] +
                                                               _discriminator_tower_grads + 
                                                               _discriminator_tower_params, 
                                                               feed_dict=mini_batch_dict)

                                tower_losses_list += [x for x in mini_batch_results[0]]
                                tower_preds_list += [x for x in mini_batch_results[1]]
                                tower_mses_list += [x for x in mini_batch_results[2]]

                                for i in range(3, 3+batch_size):
                                    ph_dict.update({ph: val for ph, val in 
                                                    zip(tower_grads_ph[j], 
                                                        mini_batch_results[i])})
                                    j += 1

                                del mini_batch_dict
                                del mini_batch_results

                            ph_dict.update({tower_losses_ph: np.asarray(tower_losses_list),
                                            tower_mses_ph: np.asarray(tower_mses_list),
                                            learning_rate_ph: learning_rate,
                                            img_ph[0]: __img,
                                            img_ph[1]: __img,
                                            img_truth_ph[0]: __img_truth,
                                            img_truth_ph[1]: __img_truth})
                    
                            del tower_losses_list
                            del tower_mses_list

                            discriminator_tower_losses_list = []
                            discriminator_tower_preds_list = []
                            discriminator_ph_dict = {}

                            #Apply the discriminator to the generated images
                            for j in range(effective_batch_size):
                                mini_batch_dict = {}
                                mini_batch_dict.update({pred_ph: tower_preds_list[j]})
                                mini_batch_dict.update({label_ph: np.array([0.], dtype=np.float32)})

                                mini_batch_results = sess.run([_discriminator_tower_losses, 
                                                               _discriminator_tower_preds] +
                                                               _discriminator_tower_grads,
                                                               feed_dict=mini_batch_dict)

                                discriminator_tower_losses_list += [x for x in mini_batch_results[0]]
                                discriminator_tower_preds_list += [x for x in mini_batch_results[1]]

                                for i in range(2, 2+batch_size):
                                    discriminator_ph_dict.update({ph: val for ph, val in 
                                                                  zip(discriminator_tower_grads_ph[j], 
                                                                      mini_batch_results[i])})

                            del tower_preds_list
                            del mini_batch_dict

                            #Apply the discriminator to real images
                            for j in range(effective_batch_size):
                                mini_batch_dict = {}
                                mini_batch_dict.update({pred_ph: tower_ground_truths_list[j]})
                                mini_batch_dict.update({label_ph: np.array([1.], dtype=np.float32)})

                                mini_batch_results = sess.run([_discriminator_tower_losses, 
                                                               _discriminator_tower_preds] +
                                                               _discriminator_tower_grads,
                                                               feed_dict=mini_batch_dict)

                                discriminator_tower_losses_list += [x for x in mini_batch_results[0]]
                                discriminator_tower_preds_list += [x for x in mini_batch_results[1]]

                                for i in range(2, 2+batch_size):
                                    discriminator_ph_dict.update({ph: val for ph, val in 
                                                                  zip(discriminator_tower_grads_ph[j+effective_batch_size], 
                                                                      mini_batch_results[i])})

                            generation_losses = np.asarray(discriminator_tower_losses_list[:effective_batch_size])
                            discrimination_losses = np.asarray(discriminator_tower_losses_list[:effective_batch_size] + 
                                                               discriminator_tower_losses_list[effective_batch_size:])

                            _, discriminator_loss = sess.run([discriminator_train_op, discriminator_get_loss],
                                                              feed_dict=discriminator_ph_dict)

                            _, actual_loss, loss_mse = sess.run([train_op, get_loss, get_loss_mse],
                                                                  feed_dict=ph_dict)
                            del ph_dict

                            try:
                                log.write("Iter: {}, Loss: {:.8f}".format(counter, float(loss_mse)))
                            except:
                                print("Failed to write to log")

                            if not counter % val_skip_n:
                                mini_batch_dict = {}
                                for i in range(batch_size):
                                    ___img, ___img_truth = sess.run([img_val[i], img_val_truth[i]])
                                    mini_batch_dict.update({img_ph[i]: ___img})
                                    mini_batch_dict.update({img_truth_ph[i]: ___img_truth})

                                mini_batch_results = sess.run([_tower_losses, _tower_preds, _tower_mses] +
                                                                tower_grads,
                                                                feed_dict=mini_batch_dict)
                                val_loss = np.mean(np.asarray([x for x in mini_batch_results[2]]))

                                try:
                                    val_log.write("Iter: {}, Loss: {:.8f}".format(counter, float(val_loss)))
                                except:
                                    print("Failed to write to val log")

                                print("Iter: {}, Loss: {:.6f}, Huberised loss: {:.6f}, Val loss: {:.6f}".format(
                                      counter, loss_value, actual_loss, val_loss))
                            else:
                                print("Iter: {}, Loss: {:.6f}, Huberised loss: {:.6f}".format(
                                      counter, loss_value, actual_loss))

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




