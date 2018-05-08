from __future__ import division
from __future__ import print_function

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

filters00 = 32
filters01 = 64
filters1 = 128
filters2 = 256
filters3 = 728
filters4 = 728
filters5 = 1024
filters6 = 1536
filters7 = 2048
numMiddleXception = 8

fc_features = 4096

#trainDir = "F:/stills_all/train/"
#valDir = "F:/stills_all/val/"
#testDir = "F:/stills_all/test/"

data_dir = "E:/stills_all/stills_all/"

modelSavePeriod = 3 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/autoencoder_xception-multi-gpu-6/"

shuffle_buffer_size = 1000
num_parallel_calls = 4
num_parallel_readers = 4
prefetch_buffer_size = 8
batch_size = 12
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
log_every = 1 #Log every _ examples
cumProbs = np.array([]) #Indices of the distribution plus 1 will be correspond to means

numMeans = 64 // batch_size
scaleMean = 4 #Each means array index increment corresponds to this increase in the mean
numDynamicGrad = 1 # Number of gradients to calculate for each possible mean when dynamically updating training
lossSmoothingBoxcarSize = 5

#Dimensions of images in the dataset
height = width = 2048
channels = 1 #Greyscale input image

#Sidelength of images to feed the neural network
cropsize = 299
height_crop = width_crop = cropsize

#hparams = experiment_hparams(train_batch_size=batch_size, eval_batch_size=16)

weight_decay = 0.0#2.0e-4
learning_rate = 0.001
num_workers = 1

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

##Modified aligned xception
def architecture(inputs, ground_truth, phase=False, params=None):
    """
    Atrous convolutional encoder-decoder noise-removing network
    phase - True during training
    """

    #phase = mode == tf.estimator.ModeKeys.TRAIN #phase is true during training
    concat_axis = 3

    ##Reusable blocks
    def batch_then_activ(input):

        batch_then_activ = tf.contrib.layers.batch_norm(
            input,
            center=True, scale=True,
            is_training=phase,
            fused=True)
        batch_then_activ = tf.nn.relu(batch_then_activ)

        return batch_then_activ

    def conv_block(input, filters, phase=phase):
        """
        Convolution -> batch normalisation -> leaky relu
        phase defaults to true, meaning that the network is being trained
        """

        conv_block = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=3,
            padding="SAME",
            activation=tf.nn.relu)
        conv_block = batch_then_activ(conv_block)

        return conv_block


    def strided_conv_block(input, filters, stride, rate=1, phase=phase):
        
        return slim.separable_convolution2d(
            inputs=input,
            num_outputs=filters,
            kernel_size=3,
            depth_multiplier=1,
            stride=stride,
            padding='SAME',
            data_format='NHWC',
            rate=rate,
            activation_fn=tf.nn.relu,
            normalizer_fn=tf.contrib.layers.batch_norm,
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

    def deconv_block(input, filters, phase=phase):
        '''Transpositionally convolute a feature space to upsample it'''
        
        #Residual
        residual = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=1,
            strides=2,
            padding="SAME")
        residual = batch_then_activ(residual)

        #Main flow
        main_flow = strided_conv_block(
            input=input,
            filters=filters,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters,
            stride=1)
        main_flow = tf.layers.conv2d_transpose(
            inputs=main_flow,
            filters=filters,
            kernel_size=3,
            strides=2,
            padding="SAME")
        main_flow = batch_then_activ(main_flow)

        return deconv_block + residual

    def xception_entry_flow(input):

        #Entry flow 0
        entry_flow = tf.layers.conv2d(
            inputs=input,
            filters=filters00,
            kernel_size=3,
            strides = 2,
            padding="SAME")
        entry_flow = batch_then_activ(entry_flow)
        entry_flow = tf.layers.conv2d(
            inputs=entry_flow,
            filters=filters01,
            kernel_size=3,
            padding="SAME")
        entry_flow = batch_then_activ(entry_flow)

        #Residual 1
        residual1 = tf.layers.conv2d(
            inputs=entry_flow,
            filters=filters1,
            kernel_size=1,
            strides=2,
            padding="SAME")
        residual1 = batch_then_activ(residual1)
       
        #Main flow 1
        main_flow1 = strided_conv_block(
            input=entry_flow,
            filters=filters1,
            stride=1)
        main_flow1 = strided_conv_block(
            input=main_flow1,
            filters=filters1,
            stride=1)
        main_flow1_strided = strided_conv_block(
            input=main_flow1,
            filters=filters1,
            stride=2)

        residual_connect1 = main_flow1_strided + residual1

        #Residual 2
        residual2 = tf.layers.conv2d(
            inputs=residual_connect1,
            filters=filters2,
            kernel_size=1,
            strides=2,
            padding="SAME")
        residual2 = batch_then_activ(residual2)
       
        #Main flow 2
        main_flow2 = strided_conv_block(
            input=residual_connect1,
            filters=filters2,
            stride=1)
        main_flow2 = strided_conv_block(
            input=main_flow2,
            filters=filters2,
            stride=1)
        main_flow2_strided = strided_conv_block(
            input=main_flow2,
            filters=filters2,
            stride=2)

        residual_connect2 = main_flow2_strided + residual2

        #Residual 3
        residual3 = tf.layers.conv2d(
            inputs=residual_connect2,
            filters=filters3,
            kernel_size=1,
            strides=2,
            padding="SAME")
        residual3 = batch_then_activ(residual3)
       
        #Main flow 3
        main_flow3 = strided_conv_block(
            input=residual_connect2,
            filters=filters3,
            stride=1)
        main_flow3 = strided_conv_block(
            input=main_flow3,
            filters=filters3,
            stride=1)
        main_flow3_strided = strided_conv_block(
            input=main_flow3,
            filters=filters3,
            stride=2)

        residual_connect3 = main_flow3_strided + residual3

        return residual_connect3


    def xception_middle_block(input):
        
        main_flow = strided_conv_block(
            input=input,
            filters=filters4,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters4,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters4,
            stride=1)

        return main_flow + input


    def xception_exit_flow(input):

        #Residual
        residual = tf.layers.conv2d(
            inputs=input,
            filters=filters5,
            kernel_size=1,
            strides=1,
            padding="SAME")
        residual = batch_then_activ(residual)

        #Main flow
        main_flow = main_flow = strided_conv_block(
            input=input,
            filters=filters4,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters5,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters5,
            stride=2)

        #Residual connection
        main_flow = main_flow + residual

        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters6,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters6,
            stride=1,
            rate=2) # Swap stride and rate values to to decimate further
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters7,
            stride=1)

        return main_flow

    '''Model building'''
    input_layer = tf.reshape(inputs, [-1, cropsize, cropsize, channels])

    #Build Xception
    main_flow = xception_entry_flow(input_layer)
    
    for _ in range(numMiddleXception):
        main_flow = xception_middle_block(main_flow)

    main_flow = xception_exit_flow(main_flow)

    main_flow = tf.reduce_mean(main_flow, [1,2]) #Global average pooling

    fc = tf.layers.flatten(main_flow)
    fc = tf.contrib.layers.fully_connected(
        fc,
        fc_features)
    fc = tf.contrib.layers.fully_connected(
        fc,
        fc_features)

    logits = tf.layers.dense(fc, units=30)

    output = tf.nn.softmax(logits)

    return output

##########################################################################################################

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


def get_model_fn(num_gpus, variable_strategy, num_workers):
    """Returns a function that will build the model."""

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
        tower_gradvars = []
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
            with tf.variable_scope('pellet', reuse=bool(i != 0)):
                with tf.name_scope('tower_%d' % i) as name_scope:
                    with tf.device(device_setter):
                        loss, gradvars, preds = _tower_fn(
                            is_training, tower_features[i], tower_labels[i])

                        tower_losses.append(loss)
                        tower_gradvars.append(gradvars)
                        tower_preds.append(preds)
                        if i == 0:
                            # Only trigger batch_norm moving mean and variance update from
                            # the 1st tower. Ideally, we should grab the updates from all
                            # towers but these stats accumulate extremely fast so we can
                            # ignore the other stats from the other towers without
                            # significant detriment.
                            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS,
                                                            name_scope)

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

        # Device that runs the ops to apply global gradient updates.
        consolidation_device = '/gpu:0' if variable_strategy == 'GPU' else '/cpu:0'
        with tf.device(consolidation_device):
            num_batches_per_epoch = num_examples_per_epoch(
                'train') // (params.train_batch_size * num_workers)
            boundaries = [
                num_batches_per_epoch * x
                for x in np.array([82, 123, 300]+[300 for _ in range(15)], dtype=np.int64)
            ]
            staged_lr = [params.learning_rate * x for x in [1, 0.1, 0.01, 0.002]+[0.002 for _ in range(15)]]


            #learning_rate = tf.train.piecewise_constant(tf.train.get_global_step(),
            #                                      boundaries, staged_lr)

            loss = tf.reduce_mean(tower_losses, name='loss')

            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        # Create single grouped train op
        train_op = [
            optimizer.apply_gradients(
                gradvars, global_step=tf.train.get_global_step())
        ]
        train_op.extend(update_ops)
        train_op = tf.group(*train_op)

        predictions = tf.concat([p['output'] for p in tower_preds], axis=0)

        return loss, predictions, train_op

    return _model_fn


def _tower_fn(is_training, feature, ground_truth):
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

    tower_pred = {
        'output': output
    }

    out = tf.reshape(output, [-1, cropsize, cropsize, channels])
    
    #tower_loss = tf.reduce_mean(tf.losses.mean_squared_error(out, truth))

    out_batch = tf.unstack(out, num=batch_size//2, axis=0)

    similarity = tf.losses.cosine_distance(out_batch[0], out_batch[1])
    for i in range(2, len(out_batch), 2):
        similarity += tf.acos(tf.losses.cosine_distance(out_batch[i], out_batch[i+1]))
    similarity /= 0.5*len(out_batch)

    dissimilarity_losses = []
    for i in range(len(out_batch)):
        for j in range(i+1, len(out_batch)):
            dissimilarity_losses.append( tf.acos(tf.losses.cosine_distance(out_batch[i], out_batch[j])) )
    dissimilarity = tf_median( tf.concat(dissimilarity_losses, axis=0) )

    tower_loss = 1000. - similarity - dissimilarity
    
    #mse = tf.losses.mean_squared_error(out, truth)
    #tower_loss = tf.reduce_mean(mse)

    tower_loss += weight_decay * tf.add_n(
        [tf.nn.l2_loss(v) for v in model_params])

    tower_grad = tf.gradients(tower_loss, model_params)

    #print(tower_grad)

    return tower_loss, zip(tower_grad, model_params), tower_pred


def load_image(addr, resizeSize=None, imgType=np.float32):
    """Read an image and make sure it is of the correct type. Optionally resize it"""
    
    try:
        img = imread(addr, mode='F')
    except:
        img = 0.5*np.ones((2048,2048))

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


def flip_rotate(img, not_same=False):
    """Applies a random flip || rotation to the image, possibly leaving it unchanged"""

    choice = 1+int(7*np.random.rand()) if not_same else int(8*np.random.rand())
    
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
    if choice == 7 or choice == 8: #Include 8 just in case...
        return np.flip(np.rot90(img, 1), 1)


def preprocess(img):
    """
    Threshold the image to remove dead or very bright pixels.
    Then crop a region of the image of a random size and resize it.
    """

    size = int(cropsize + np.random.rand()*(height-cropsize))
    topLeft_x = int(np.random.rand()*(height-size))
    topLeft_y = int(np.random.rand()*(height-size))

    crop = img[topLeft_y:(topLeft_y+cropsize), topLeft_x:(topLeft_x+cropsize)]

    resized = cv2.resize(crop, (cropsize, cropsize), interpolation=cv2.INTER_AREA)

    resized[np.isnan(resized)] = 0.5
    resized[np.isinf(resized)] = 0.5

    return scale0to1(flip_rotate(resized))


def record_parser(record):
    """Parse files and generate lower quality images from them"""

    img = load_image(record)
    img = preprocess(img)

    return np.dstack((img, flip_rotate(img, not_same=True)))

def reshaper(img):
    return tf.reshape(img, [cropsize, cropsize, channels])

def input_fn(dir, subset, batch_size, num_shards):
    """Create a dataset from a list of filenames and shard batches from it"""

    with tf.device('/cpu:0'):

        dataset = tf.data.Dataset.list_files(dir+subset+"/"+"*.tif")
        dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
        dataset = dataset.map(
            lambda file: tf.py_func(record_parser, [file], [tf.float32]),
            num_parallel_calls=num_parallel_calls)
        dataset = dataset.map(reshaper, num_parallel_calls=num_parallel_calls)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
        dataset = dataset.repeat(num_epochs)

        iter = dataset.make_one_shot_iterator()

        img_batch = iter.get_next()

        image_batch = tf.unstack(img_batch, num=batch_size, axis=0)
        images = []
        for i in range(batch_size):
            split1, split2 =  tf.split(image_batch[i], num_or_size_splits=2, axis=3)
            images.append(split1)
            images.append(split2)
        features_shard = [tf.parallel_stack(images)]

        return [features_shard], [features_shard]


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


def main(job_dir, data_dir, variable_strategy, num_gpus, log_device_placement, num_intra_threads,
         **hparams):

    temp = set(tf.all_variables())
    log = open(log_file, 'a')

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

        is_training = True
        model_fn = get_model_fn(num_gpus, variable_strategy, num_workers)

        loss, prediction, train_op = model_fn(img, img, mode=is_training, params=hparams)

        with tf.Session(config=sess_config) as sess: #Alternative is tf.train.MonitoredTrainingSession()

            init = tf.global_variables_initializer()
            sess.run(init)

            sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

            train_writer = tf.summary.FileWriter( logDir, sess.graph )

            #Set up mean probabilities to be dynamically adjusted during training
            probs = np.ones(numMeans, dtype=np.float32)
            losses0 = np.empty([])

            #print(tf.all_variables())

            counter = 0
            cycleNum = 0
            while True:
                cycleNum += 1
                #Train for a couple of hours
                time0 = time.time()
                while time.time()-time0 < modelSavePeriod:
                    counter += 1

                    #merge = tf.summary.merge_all()

                    _, loss_value = sess.run([train_op, loss])
                    #img, pred = sess.run([img, prediction])
                    #print(np.min(img), '\n', np.min(pred))
                    print("Iter: {}, Loss: {:.6f}".format(counter, loss_value))
                    log.write("Iter: {}, Loss: {:.6f}".format(counter, loss_value))
                    #train_writer.add_summary(summary, counter)

                #Save the model
                #saver.save(sess, save_path=model_dir+"model", global_step=counter)
                tf.saved_model.simple_save(
                    session=sess,
                    export_dir=model_dir+"model-"+str(counter)+"/",
                    inputs={"img": img[0][0]},
                    outputs={"prediction": prediction})

                #predict_fn = tf.contrib.predictor.from_saved_model(model_dir+"model-"+str(counter)+"/")

                #loaded_img = imread("E:/stills_hq/reaping1.tif", mode='F')
                #loaded_img = scale0to1(cv2.resize(loaded_img, (cropsize, cropsize), interpolation=cv2.INTER_AREA))
                #cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
                #cv2.imshow("dfsd", loaded_img)
                #cv2.waitKey(0)

                #prediction1 = predict_fn({"lq": loaded_img})

                #cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
                #cv2.imshow("dfsd", prediction1['prediction'].reshape(cropsize, cropsize))
                #cv2.waitKey(0)

                #Evaluate the model and use the results to dynamically adjust the training process
                losses = np.zeros(numMeans, dtype=np.float32)
                for i in range(numMeans):
                    for _ in range(numDynamicGrad):
                        losses[i] += sess.run(loss)
                        print(i, losses[i])
                    losses[i] /= numDynamicGrad

                np.save(model_dir+"losses-"+str(counter), losses)
                losses0 = losses
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


