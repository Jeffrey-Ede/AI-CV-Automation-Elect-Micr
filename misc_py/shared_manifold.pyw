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

features1 = 25
features2 = 50
features3 = 100
features4 = 200
features5 = 400

gen_features1 = 32 #256
gen_features2 = 64 #128
gen_features3 = 128 #64
gen_features4 = 256 #32
gen_features5 = 8 #32

dec_features2 = 128 #16
dec_features3 = 256 #32
dec_features4 = 128 #64
dec_features5 = 64 #128
dec_features6 = 32 #256

num_global_enhancer_blocks = 8
num_local_enhancer_blocks = 4

data_dir = "X:/Jeffrey-Ede/stills_all/"
#data_dir = "E:/stills_hq-mini/"

modelSavePeriod = 4 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "X:/Jeffrey-Ede/models/fract-recur-conv-autoencoder-1/"

shuffle_buffer_size = 5000
num_parallel_calls = 8
num_parallel_readers = 8
prefetch_buffer_size = 20
batch_size = 1
num_gpus = 1

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
batch_decay_gen = 0.9997
batch_decay_discr = 0.9997
initial_learning_rate = 0.001
initial_discriminator_learning_rate = 0.001
num_workers = 1

increase_batch_size_by_factor = 1
effective_batch_size = increase_batch_size_by_factor*batch_size

save_result_every_n_batches = 100000

val_skip_n = 100
trainee_switch_skip_n = 1

flip_weight = 20

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

def generator_architecture(inputs, phase, batch_norm_on, params=None):
    """Generates fake data to try and fool the discrimator"""

    concat_axis = 3

    depth = 4
    turns = 1
    features = 384

    def _instance_norm(net, train=phase):
        batch, rows, cols, channels = [i.value for i in net.get_shape()]
        var_shape = [channels]
        mu, sigma_sq = tf.nn.moments(net, [1,2], keep_dims=True)
        shift = tf.Variable(tf.zeros(var_shape), trainable=train)
        scale = tf.Variable(tf.ones(var_shape), trainable=train)
        epsilon = 1e-3
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

    ##Reusable blocks
    def _batch_norm_fn(input, train=True):
        batch_norm = tf.contrib.layers.batch_norm(
            input,
            decay=batch_decay_gen,
            epsilon=0.001,
            center=True, 
            scale=True,
            is_training=batch_norm_on,
            fused=True,
            zero_debias_moving_mean=False,
            renorm=False)
        return batch_norm

    def batch_then_activ(input):
        _batch_then_activ = tf.nn.relu(input)
        return _batch_then_activ

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

    def conv_block(input, filters, phase=phase, pad_size=None, dec=False):
        """
        Convolution -> batch normalisation -> activation
        phase defaults to true, meaning that the network is being trained
        """

        conv_block = strided_conv_block(input, filters, 1, 1, pad_size=pad_size, dec=dec)

        return conv_block

    def strided_conv_block(input, filters, stride, rate=1, phase=phase, 
                            extra_batch_norm=True, kernel_size=3, pad_size=None,
                            dec=False):
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
        if dec:
            strided_conv = batch_then_activ(strided_conv)
        else:
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

    def deconv_block(input, filters):
        '''Transpositionally convolute a feature space to upsample it'''

        deconv = slim.conv2d_transpose(
            inputs=input,
            num_outputs=filters,
            kernel_size=3,
            stride=2,
            padding='SAME',
            activation_fn=None)
        deconv = batch_then_activ(deconv)

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

    def xception_middle_block(input, features, pad_size=None, dec=False):
        
        main_flow = strided_conv_block(
            input=input,
            filters=features,
            stride=1, 
            pad_size=pad_size,
            dec=dec)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=features,
            stride=1,
            pad_size=pad_size,
            dec=dec)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=features,
            stride=1,
            pad_size=pad_size,
            dec=dec)

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

    def embedding(input, features, pad_size=None, kernel_size=3):
        embed = strided_conv_block(input, features, 1, kernel_size=kernel_size, pad_size=pad_size)
        return embed

    def reconstruction(input, features, pad_size=None, kernel_size=3):
        recon = strided_conv_block(input, features, 1, kernel_size=kernel_size, pad_size=pad_size)
        return recon

    def reconstruction_final_enc(input, pad_size=None):
        recon = strided_conv_block(input, features, 1, kernel_size=3, pad_size=pad_size)
        recon = tf.image.resize_images(recon, (8, 8))
        return recon

    def reconstruction_final_dec(input, pad_size=None):
        recon = strided_conv_block(input, 128, 1, pad_size=pad_size)
        recon = strided_conv_block(input, 64, 1, pad_size=pad_size)
        recon = strided_conv_block(input, 1, 1, pad_size=pad_size)
        return recon


    def fract_recur_conv(input, turns, depth, embed_kernel_size=3, recon_kernel_size=3):

        embed_scope = None
        recur_scope = None
        recon_scope = None

        def _fract_recur_conv(input=input, turns=turns, depth=depth-1, 
                                embed_kernel_size=embed_kernel_size, 
                                recon_kernel_size=recon_kernel_size):

            nonlocal embed_scope
            nonlocal recur_scope
            nonlocal recon_scope

            if embed_scope:
                with tf.variable_scope(embed_scope, reuse=True) as scope0:
                    embed = embedding(input, features)
            else:
                default_embed_scope="embedding"
                with tf.variable_scope(default_embed_scope) as scope0:
                    embed_scope = scope0
                    embed = embedding(input, features)

            #Perform recursive convolutions
            if not depth:
                recur_convs = []
                if recur_scope:
                    with tf.variable_scope(recur_scope, reuse=True) as scope1:
                        recur_frac_conv = strided_conv_block(embed, features, 1)
                else:
                    default_recur_scope="recur_conv"
                    with tf.variable_scope(default_recur_scope) as scope1:
                        recur_scope = scope1
                        recur_frac_conv = strided_conv_block(embed, features, 1)
                recur_convs.append(recur_frac_conv)

                for _ in range(1, turns):
                    with tf.variable_scope(scope1, reuse=True):
                            recur_frac_conv = strided_conv_block( recur_frac_conv, features, 1 )
                    recur_convs.append(recur_frac_conv)
            else:
                recur_convs = []
                for _ in range(turns):
                    recur_frac_conv = _fract_recur_conv(depth=depth-1)
                    recur_convs.append(recur_frac_conv)

            output = 0.
            for conv in recur_convs:
                concat = tf.concat([input, conv], axis=3)
                
                if recon_scope:
                    with tf.variable_scope(recon_scope, reuse=True) as scope2:
                        output += reconstruction(concat, features)
                else:
                    default_recon_scope="reconstruction"
                    with tf.variable_scope(default_recon_scope) as scope2:
                        recon_scope = scope2
                        output += reconstruction(concat, features)

            return output

        output = _fract_recur_conv()

        return output

    ##Model building
    input_layer = tf.reshape(inputs, [-1, generator_input_size, generator_input_size, channels])

    with tf.variable_scope("GAN/Gen/enc") as inner_scope:

        #256
        enc = strided_conv_block(input=input_layer,
                                 filters=64,
                                 stride=2,
                                 kernel_size=3)

        enc = strided_conv_block(input=enc,
                                    filters=128,
                                    stride=1,
                                    kernel_size=3)

        enc = fract_recur_conv(enc, turns=4, depth=4, embed_kernel_size=5, recon_kernel_size=3)

        enc = reconstruction_final_enc(enc)

        enc = tf.contrib.layers.fully_connected(enc, 8, activation_fn=None)
        enc = batch_then_activ(enc)
        logits = tf.contrib.layers.flatten(enc)


    with tf.variable_scope("GAN/Gen/dec") as inner_scope:

        dec = tf.contrib.layers.fully_connected(enc, 32, activation_fn=None)
        dec = batch_then_activ(dec)

        dec = deconv_block(dec, 64) #16
        dec = deconv_block(dec, 64)
        dec = deconv_block(dec, 64) #64
        dec = deconv_block(dec, 64)
        dec = deconv_block(dec, 64) #256

        dec = slim.conv2d(inputs=dec,
                        num_outputs=1,
                        kernel_size=3,
                        padding="SAME",
                        activation_fn=None)

        dec = tf.tanh(dec)

    return logits, dec


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


def adam_updates(params, cost_or_grads, lr=0.001, mom1=np.array([0.5]), mom2=np.array([0.999]), clip_norm=None):
    ''' Adam optimizer '''
    updates = []
    if type(cost_or_grads) is not list:
        grads = tf.gradients(cost_or_grads, params)
    else:
        grads = cost_or_grads

    if clip_norm:
        raise Exception('Functionality not added yet')
        #grads = tf.clip_by_global_norm(grads, clip_norm=clip_norm)

    #Weight normalized ADAM updates
    t = tf.Variable(1., 'adam_t')
    for p, g in zip(params, grads):
        mg = tf.Variable(tf.zeros(p.get_shape()), p.name + '_adam_mg')
        if mom1>0:
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


def super_confuser(input, reuse=False):

    features1 = 128
    features2 = 256

    with tf.variable_scope("GAN/SuperConfuser", reuse=reuse):
        
        x = strided_conv_block(input, features1, 1, 1)

        x = strided_conv_block(x, features2, 2, 1)
        x0 = x
        x = strided_conv_block(x, features2, 1, 1)
        x = strided_conv_block(x, features2, 1, 1)
        x += x0
        x0 = x
        x = strided_conv_block(x, features2, 1, 1)
        x = strided_conv_block(x, features2, 1, 1)
        x += x0
        x = deconv_block(x, features1)

        x = slim.conv2d(
                inputs=x,
                num_outputs=1,
                kernel_size=3,
                padding="SAME",
                activation_fn=None,
                weights_initializer=None,
                biases_initializer=None)

    return x

def confuser(input, reuse=False):

    features1 = 32
    features2 = 64
    features3 = 128
    features4 = 256
    features5 = 384
    features6 = 512

    with tf.variable_scope("GAN/Confuser", reuse=reuse):

        with tf.variable_scope('start'):
            x = strided_conv_block(input, features1, 2, 1)
            x = strided_conv_block(x, features2, 2, 1)

        with tf.variable_scope('mid'):
            x = strided_conv_block(x, features3, 2, 1)

            partway = x #To go to super=confuser

        with tf.variable_scope('end'):
            x = strided_conv_block(x, features4, 2, 1)

            x = strided_conv_block(x, features5, 2, 1)
            x = strided_conv_block(x, features6, 2, 1)

            x = tf.reshape(x, (-1, features6*(cropsize//(2**6)**2)))
            x = tf.contrib.layers.fully_connected(inputs=x,
                                                    num_outputs=1,
                                                    activation_fn=None)
        x = tf.sigmoid(x)

    return partway, x

def discriminator(input, reuse=False, name="Discr1"):

    features1 = 64
    features2 = 128
    features3 = 256
    features4 = 512
    features5 = features4

    with tf.variable_scope("GAN/"+name, reuse=reuse):

        '''Model building'''        
        layers = []

        with tf.variable_scope("small", reuse=reuse) as small_scope:
            small = inputs[0]
            small = strided_conv_block(small, features1, 2, 1, kernel_size=4)
            layers.append(small)
            small = strided_conv_block(small, features2, 2, 1, kernel_size=4)
            layers.append(small)
            small = strided_conv_block(small, features3, 2, 1, kernel_size=4)
            layers.append(small)
            small = strided_conv_block(small, features4, 2, 1, kernel_size=4)
            layers.append(small)
            #small = tf.reduce_mean(small, [1,2])
            small = tf.reshape(small, (-1, features5*16))
            small = tf.contrib.layers.fully_connected(inputs=small,
                                                      num_outputs=1,
                                                      activation_fn=None)

        with tf.variable_scope("medium", reuse=reuse) as medium_scope:
            medium = inputs[1]
            #medium = tf.nn.avg_pool(medium,
            #                        [1, 2, 2, 1],
            #                        strides=[1, 2, 2, 1],
            #                        padding='SAME')
            medium = strided_conv_block(medium, features1, 2, 1, kernel_size=4)
            layers.append(medium)
            medium = strided_conv_block(medium, features2, 2, 1, kernel_size=4)
            layers.append(medium)
            medium = strided_conv_block(medium, features3, 2, 1, kernel_size=4)
            layers.append(medium)
            medium = strided_conv_block(medium, features4, 2, 1, kernel_size=4)
            layers.append(medium)
            #medium = tf.reduce_mean(medium, [1,2])
            medium = tf.reshape(medium, (-1, features5*16))
            medium = tf.contrib.layers.fully_connected(inputs=medium,
                                                       num_outputs=1,
                                                       activation_fn=None)

        with tf.variable_scope("large", reuse=reuse) as large_scope:
            large = inputs[2]
            large = strided_conv_block(large, features1, 2, 1, kernel_size=4)
            layers.append(large)
            large = strided_conv_block(large, features2, 2, 1, kernel_size=4)
            layers.append(large)
            large = strided_conv_block(large, features3, 2, 1, kernel_size=4)
            layers.append(large)
            large = strided_conv_block(large, features4, 2, 1, kernel_size=4)
            layers.append(large)
            #large = tf.reduce_mean(large, [1,2])
            large = tf.reshape(large, (-1, features5*16))
            large = tf.contrib.layers.fully_connected(inputs=large,
                                                   num_outputs=1,
                                                   activation_fn=None)

        output = tf.sigmoid(tf.reduce_max(tf.concat([small, medium, large], axis=1), axis=1))

    return [output] + layers


def distiller(input, reuse=False, name="Distiller"):

    features1 = 32
    features2 = 64
    features3 = 128
    features4 = 256
    features5 = 512
    features6 = 512

    concat_axis = 3

    with tf.variable_scope("GAN/"+name, reuse=reuse):

        with tf.variable_scope("main", reuse=reuse):

            x = strided_conv_block(input, features1, 2, 1, kernel_size=4) # 512
            x1 = x
            x = strided_conv_block(x, features2, 2, 1, kernel_size=4)
            x2 = x
            x = strided_conv_block(x, features3, 2, 1, kernel_size=4) #128
            x3 = x
            x = strided_conv_block(x, features4, 2, 1, kernel_size=4)
            x4 = x
            x = strided_conv_block(x, features5, 2, 1, kernel_size=4) #32
            x5 = x

            x = strided_conv_block(x, features6, 2, 1, kernel_size=4) 
            x6 = x
            x = strided_conv_block(x, features6, 2, 1, kernel_size=4) #8
            x7 = x
            x = strided_conv_block(x, features6, 2, 1, kernel_size=4)
            x8 = x

            x = strided_conv_block(x, features6, 2, 1, kernel_size=4)

            x = deconv_block(x, features6)
            x = tf.concat([x, x8], axis=concat_axis, kernel_size=4)
            x = deconv_block(x, features6)
            x = tf.concat([x, x7], axis=concat_axis, kernel_size=4)
            x = deconv_block(x, features6)
            x = tf.concat([x, x6], axis=concat_axis, kernel_size=4)
            x = deconv_block(x, features6)

            x = tf.concat([x, x5], axis=concat_axis, kernel_size=4)
            x = deconv_block(x, features5)
            x = tf.concat([x, x4], axis=concat_axis, kernel_size=4)
            x = deconv_block(x, features4)
            x = tf.concat([x, x3], axis=concat_axis, kernel_size=4)
            x = deconv_block(x, features3)
            x = tf.concat([x, x2], axis=concat_axis, kernel_size=4)
            x = deconv_block(x, features2)
            x = tf.concat([x, x1], axis=concat_axis, kernel_size=4)

        with tf.variable_scope("end", reuse=reuse):
            x = deconv_block(x, features1)

        x = slim.conv2d(
                inputs=x,
                num_outputs=1,
                kernel_size=3,
                padding="SAME",
                activation_fn=None,
                weights_initializer=None,
                biases_initializer=None)

    return x


def generator(input, reuse=False, name="Gen1"):

    features1 = 32
    features2 = 64
    features3 = 128
    features4 = 256

    with tf.variable_scope("GAN/"+name, reuse=reuse):

        x = strided_conv_block(input, features1, 2, 1, kernel_size=3) # 512

        x1 = x
        x = strided_conv_block(x, features2, 2, 1, kernel_size=3)
        x2 = x
        x = strided_conv_block(x, features3, 2, 1, kernel_size=3) #128
        x3 = x
        x = strided_conv_block(x, features4, 2, 1, kernel_size=3)
        x4 = x

        x = strided_conv_block(x, features4, 2, 1, kernel_size=3)

        x = deconv_block(x, features4)
        x = tf.concat([x, x4], axis=concat_axis, kernel_size=3)
        x = deconv_block(x, features4)
        x = tf.concat([x, x3], axis=concat_axis, kernel_size=3)
        x = deconv_block(x, features3)
        x = tf.concat([x, x2], axis=concat_axis, kernel_size=3)
        x = deconv_block(x, features2)
        x = tf.concat([x, x1], axis=concat_axis, kernel_size=3)
        x = deconv_block(x, features1)

        x = slim.conv2d(
                inputs=x,
                num_outputs=1,
                kernel_size=3,
                padding="SAME",
                activation_fn=None,
                weights_initializer=None,
                biases_initializer=None)

    return x

def experiment(manifold1, manifold2, lr_gen1_ph, lr_gen2_ph, lr_discr1_ph, lr_discr2_ph, 
               lr_distiller_ph, lr_confuser_ph, flip_fake1_ph, flip_real1_ph,
               flip_fake2_ph, flip_real2_ph, flip_confuser1_ph, flip_confuser2_ph, 
               sc_grad_mean_ph, c_grad_mean_ph,
               sc_grad_beta_ph, c_grad_beta_ph, gan1_grad_mean_ph, gan2_grad_mean_ph,
               confuser_grad_mean_ph, gan1_grad_beta_ph, gan2_grad_beta_ph,
               confuser_grad_beta_ph, entropy_term1_ph, entropy_term2_ph,
               beta1_distiller_ph, beta1_gen1_ph, beta1_gen2_ph, beta1_discr1_ph, 
               beta1_discr2_ph, beta1_confuser_ph, beta1_super_confuser_ph,
               adapt_rate1_ph, adapt_rate2_ph, adapt_confusion_ph, entropy_confusion_ph):

    eps = 1.e-8

    manifold1 = tf.reshape(manifold1, [-1, cropsize, cropsize, channels])
    manifold2 = tf.reshape(manifold2, [-1, cropsize, cropsize, channels])

    ##Networking
    distillation1 = distiller(manifold1, reuse=False, name="Distiller1")
    distillation2 = distiller(manifold2, reuse=False, name="Distiller2")

    gen1 = generator(distillation1, reuse=False, name="Gen1")
    gen2 = generator(distillation2, reuse=False, name="Gen2")

    concat = tf.concat([gen1, gen2, manifold1, manifold2], axis=3)
    shapes = [(1, cropsize//8, cropsize//8, 1),
              (1, cropsize//8, cropsize//8, 1),
              (1, cropsize//8, cropsize//8, 1)]
    multiscale_crops = get_multiscale_crops(concat, multiscale_channels=4)
    multiscale_crops = [tf.unstack(crop, axis=3) for crop in multiscale_crops]

    multiscale_fake1 = [tf.reshape(unstacked[0], shape)
                  for unstacked, shape in zip(multiscale_crops, shapes)]
    multiscale_fake2 = [tf.reshape(unstacked[1], shape)
                  for unstacked, shape in zip(multiscale_crops, shapes)]
    multiscale_real1 = [tf.reshape(unstacked[2], shape)
                  for unstacked, shape in zip(multiscale_crops, shapes)]
    multiscale_real2 = [tf.reshape(unstacked[3], shape)
                  for unstacked, shape in zip(multiscale_crops, shapes)]

    discr_fake1 = discriminator(multiscale_fake1, reuse=False, name="Discr1")
    discr_real1 = discriminator(multiscale_real1, reuse=True, name="Discr1")

    discr_fake2 = discriminator(multiscale_fake2, reuse=False, name="Discr2")
    discr_real2 = discriminator(multiscale_real2, reuse=True, name="Discr2")

    pred_fake1 = discr_fake1[0]
    pred_real1 = discr_real1[0]

    pred_fake2 = discr_fake2[0]
    pred_real2 = discr_real2[0]

    _, confused_fake1 = confuser(distillation1, reuse=False)
    partway_real1, confused_real1 = confuser(manifold1, reuse=True)
    _, confused_fake2 = confuser(distillation2, reuse=True)
    partway_real2, confused_real2 = confuser(manifold2, reuse=True)

    super_confused1 = super_confuser(partway_real1, reuse=False)
    super_confused2 = super_confuser(partway_real2, reuse=True)

    distillation1_mini = tf.image.resize_images(distillation1, (cropsize//8, cropsize//8))
    distillation2_mini = tf.image.resize_images(distillation2, (cropsize//8, cropsize//8))

    super_confused1_loss = tf.losses.mean_squared_error(super_confused1, distillation1_mini)
    super_confused2_loss = tf.losses.mean_squared_error(super_confused2, distillation2_mini)

    ##Losses
    distiller1_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Distiller1/main")
    distiller1_params_end = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Distiller1/end")

    distiller2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Distiller2/main")
    distiller2_params_end = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Distiller2/end")

    confuser_end_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Confuser/end")
    confuser_mid_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Confuser/mid")
    confuser_start_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Confuser/start")

    super_confuser_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/SuperConfuser")
    
    gen1_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Gen1")
    gen2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Gen2")

    discr1_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Discr1")
    discr2_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="GAN/Discr2")

    super_confused_loss = super_confused1_loss + super_confused2_loss

    #Natural statistics
    natural_stat_losses1 = []
    for i in range(1, len(discr_fake1)):
        natural_stat_losses1.append(
            tf.reduce_mean(
                tf.losses.absolute_difference(
                    discr_fake1[i], discr_real1[i])))
    natural_stat_loss1 = tf.add_n(natural_stat_losses1)

    natural_stat_losses2 = []
    for i in range(1, len(discr_fake2)):
        natural_stat_losses2.append(
            tf.reduce_mean(
                tf.losses.absolute_difference(
                    discr_fake2[i], discr_real2[i])))
    natural_stat_loss2 = tf.add_n(natural_stat_losses2)

    weight_natural_stats = 10.

    #Relativistic generation
    x1 = pred_fake1[0] - pred_real1[0]
    rel_pred1 = tf.cond(x1[0] < 0., lambda: 0.5*x1**2+x1+0.5, lambda: -0.5*x1**2+x1+0.5)
    x2 = pred_fake2[0] - pred_real2[0]
    rel_pred2 = tf.cond(x2[0] < 0., lambda: 0.5*x2**2+x2+0.5, lambda: -0.5*x2**2+x2+0.5)

    gen1_pred_loss = -tf.log(tf.clip_by_value(rel_pred1, 1.e-8, 1.))
    gen2_pred_loss = -tf.log(tf.clip_by_value(rel_pred2, 1.e-8, 1.))

    gen1_loss = gen1_pred_loss + weight_natural_stats*natural_stat_loss1
    gen2_loss = gen2_pred_loss + weight_natural_stats*natural_stat_loss2

    #Flipping phs are False for flip; True otherwise
    fake_label1 = tf.cond(flip_fake1_ph,
                          lambda: 0.,
                          lambda: 1.)

    real_label1 = tf.cond(flip_real1_ph,
                          lambda: tf.random_uniform((1,), minval=0.9, maxval=1.),
                          lambda: 0.)

    fake_label2 = tf.cond(flip_fake2_ph,
                          lambda: 0.,
                          lambda: 1.)

    real_label2 = tf.cond(flip_real2_ph,
                          lambda: tf.random_uniform((1,), minval=0.9, maxval=1.),
                          lambda: 0.)

    #Discriminator training
    discr1_decay = 1.e-5 * tf.add_n( [tf.nn.l2_loss(v) for v in discr1_params] )
    discr2_decay = 1.e-5 * tf.add_n( [tf.nn.l2_loss(v) for v in discr2_params] )

    discr_fake1_loss = -tf.log(1.-tf.clip_by_value(tf.abs(pred_fake1-fake_label1), 0., 1.-1.e-8))
    discr_real1_loss = -tf.log(1.-tf.clip_by_value(tf.abs(pred_real1-real_label1), 0., 1.-1.e-8))

    discr_fake2_loss = -tf.log(1.-tf.clip_by_value(tf.abs(pred_fake2-fake_label2), 0., 1.-1.e-8))
    discr_real2_loss = -tf.log(1.-tf.clip_by_value(tf.abs(pred_real2-real_label2), 0., 1.-1.e-8))

    #Adapt discriminator learning rates to generator performances
    discr_fake1_adapt = tf.cond(flip_fake1_ph, 
                                lambda: adapt_rate1_ph, 
                                lambda: np.array([1.], dtype=np.float32))

    discr_real1_adapt = tf.cond(flip_real1_ph, 
                                lambda: adapt_rate1_ph, 
                                lambda: np.array([1.], dtype=np.float32))

    discr_fake2_adapt = tf.cond(flip_fake2_ph, 
                                lambda: adapt_rate2_ph, 
                                lambda: np.array([1.], dtype=np.float32))

    discr_real2_adapt = tf.cond(flip_real2_ph, 
                                lambda: adapt_rate2_ph, 
                                lambda: np.array([1.], dtype=np.float32))

    discr1_loss = (discr_fake1_adapt*discr_fake1_loss + discr_real1_adapt*discr_real1_loss + 
                   discr1_decay + entropy_term1_ph)
    discr2_loss = (discr_fake2_adapt*discr_fake2_loss + discr_real2_adapt*discr_real2_loss + 
                   discr2_decay + entropy_term2_ph)

    confused1_label = np.array([0.], dtype=np.float32)
    confused2_label = np.array([1.], dtype=np.float32)
    midway = 0.5*(confused1_label+confused2_label)

    confuser_fake1_loss = -tf.log(1.-tf.clip_by_value(tf.abs(confused_fake1-confused1_label), 0., 1.-1.e-8))
    confuser_real1_loss = -tf.log(1.-tf.clip_by_value(tf.abs(confused_real1-confused1_label), 0., 1.-1.e-8))

    confuser_fake2_loss = -tf.log(1.-tf.clip_by_value(tf.abs(confused_fake2-confused2_label), 0., 1.-1.e-8))
    confuser_real2_loss = -tf.log(1.-tf.clip_by_value(tf.abs(confused_real2-confused2_label), 0., 1.-1.e-8))

    confuser_loss = confuser_real1_loss + confuser_real2_loss

    #TODO: These need to be replaced with grad manipulations
    distillation1_loss = -tf.log(tf.clip_by_value(tf.abs(confused_fake1-midway), 1.e-8, 1.))
    distillation2_loss = -tf.log(tf.clip_by_value(tf.abs(confused_fake2-midway), 1.e-8, 1.))
    distillation_loss = confusion_rate_ph*(distillation1_loss+distillation2_loss) + gen1_loss+gen2_loss

    ##Training operations with gradient balancing

    #Gradients back to middle of confuser
    super_confuser_to_mid = super_confuser_params+confuser_params_mid
    super_confuser_grads = tf.gradients(super_confused_loss, super_confuser_to_mid)
    confuser_to_mid = confuser_params_end+confuser_params_mid
    confuser_grads = tf.gradients(confuser_loss, confuser_to_mid)

    #Get middle tensor that balanced gradients go through
    idx_super_confuser_mid = []
    for i, v in enumerate(super_confuser_grads):
        if 'mid' in v.name:
            idx_super_confuser_mid.append(i)

    idx_confuser_mid = []
    for i, v in enumerate(confuser_grads):
        if 'mid' in v.name:
            idx_confuser_mid.append(i)

    #Get the confuser gradients
    sc_grad_mean = []
    c_grad_mean = []
    num_mid_grads = len(idx_super_confuser_mid)
    for i in range(num_mid_grads):
        sc_grad_mean.append(sc_grad_beta_ph*sc_grad_mean_ph[i] + 
                           (1-sc_grad_beta_ph)*tf.reduce_mean(tf.abs(
                               super_confuser_grads[idx_super_confuser_mid[i]])))
        c_grad_mean.append(c_grad_beta_ph*c_grad_mean_ph[i] + 
                           (1-c_grad_beta_ph)*tf.reduce_mean(tf.abs(
                               confuser_grads[idx_confuser_mid[i]])))

    #Balance confuser gradents
    mid_grads = []
    for i in num_mid_grads:
        geometric_mean = tf.sqrt(sc_grad_mean[i]*c_grad_mean[i])

        mid_grad = (confuser_grads[i] - 
                    c_grad_mean[i] * super_confuser_grads[i] / (sc_grad_mean[i]+eps) )

        mid_grads.append(mid_grad)

    confuser_grads_start = tf.gradients(confuser_params_mid, 
                                        confuser_params_start,
                                        grad_ys=mid_grads)

    #Get the distiller gradients
    distiller_gan1_grads_end = tf.gradients(distiller_params_end, natural_stat_losses1)
    distiller_gan2_grads_end = tf.gradients(distiller_params_end, natural_stat_losses2)
    confuser_grads_end = tf.gradients(distiller_params_end, natural_stat_losses2)
    
    gan1_grad_mean = []
    gan2_grad_mean = []
    confuser_grad_mean = []
    distiller_num_mid_grads = len(distiller_gan1_grads_end)
    for i in range(num_mid_grads):
        gan1_grad_mean.append(gan1_grad_beta_ph*gan1_grad_mean_ph[i] + 
                              (1-gan1_grad_beta_ph)*tf.reduce_mean(tf.abs(
                              distiller_gan1_grads_end[i])))
        gan2_grad_mean.append(gan2_grad_beta_ph*gan2_grad_mean_ph[i] + 
                              (1-gan2_grad_beta_ph)*tf.reduce_mean(tf.abs(
                              distiller_gan2_grads_end[i])))
        confuser_grad_mean.append(confuser_grad_beta_ph*confuser_grad_mean_ph[i] + 
                                  (1-confuser_grad_beta_ph)*tf.reduce_mean(tf.abs(
                                  confuser_grads_end[i])))

    #Balance distiller gradients (from GANs and confuser)
    distiller_mid_grads = []
    for i in num_mid_grads:
        #gan_average = 0.5 * ( gan1_grad_mean[i] + gran2_grad_mean[i] )
        #geometric_mean = tf.sqrt( confuser_grad_mean[i] * gan_average )

        mid_grad = ( confuser_grad_mean[i] * distiller_gan1_grads_end[i] / (gan1_grad_mean[i]+eps) +
                     confuser_grad_mean[i] * distiller_gan2_grads_end[i] / (gan2_grad_mean[i]+eps) - 
                     confuser_grads_end[i] )

        distiller_mid_grads.append(mid_grad)

    distiller_grads = tf.gradients(distiller_end,
                                   distiller_params,
                                   grad_ys=distiller_mid_grads)

    optimizer_confuser = tf.train.AdamOptimizer(learning_rate=lr_confuser_ph, beta1=beta1_confuser_ph)
    train_op_confuser = optimizer_confuser.apply_gradients(confuser_grads)

    #train_op_confuser = adam_updates(confuser_params_start+confuser_params_mid+
    #                                       confuser_params_end, confuser_grads_start+
    #                                       confuser_grads_mid+confuser_grads_end, 
    #                                       lr=lr_confuser_ph)

    train_op_super_confuser = adam_updates(super_confuser_params, super_confuser_grads, 
                                           lr=lr_super_confuser_ph, beta1=beta1_super_confuser_ph)

    train_op_gen1 = adam_updates(gen1_params, gen1_loss, lr=lr_gen1_ph, beta1=beta1_gen1_ph)
    train_op_gen2 = adam_updates(gen2_params, gen2_loss, lr=lr_gen2_ph, beta1=beta1_gen2_ph)

    optimizer_distiller = tf.train.AdamOptimizer(learning_rate=lr_distiller_ph, beta1=beta1_distiller_ph)
    train_op_distiller = optimizer_distiller.apply_gradients(distiller_grads)

    optimizer_discr1 = tf.train.AdamOptimizer(learning_rate=lr_discr1_ph, beta1=beta1_discr1_ph)
    train_op_discr1 = optimizer_discr1.minimize(discr1_loss, var_list=discr1_params)
    optimizer_discr2 = tf.train.AdamOptimizer(learning_rate=lr_discr2_ph, beta1=beta1_discr2_ph)
    train_op_discr2 = optimizer_discr2.minimize(discr2_loss, var_list=discr2_params)

    return {'distillations': [distillation1, distillation2], 'gens': [gen1, gen2],
            'discr_preds': [pred_fake1, pred_real1, pred_fake2, pred_real2], 
            'confuser_preds': [confuser_real1, confuser_real2, confused_fake2, confused_real2],
            'train_ops': [train_op_gen1, train_op_gen2, train_op_discr1, train_op_discr2, train_op_distiller, 
                          train_op_confuser, train_op_super_confuser]}


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
        img = 0.5*np.ones((cropsize,cropsize))
        print("Image read failed")

    if resizeSize:
        img = cv2.resize(img, resizeSize, interpolation=cv2.INTER_AREA)

    return img.astype(img_type)


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
    img = preprocess(flip_rotate(load_image(record)))

    if np.sum(np.isfinite(img)) != cropsize**2:
        img = np.zeros((cropsize, cropsize))

    return img

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


def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return

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

def main():

    tf.reset_default_graph()

    temp = set(tf.all_variables())

    with open(discr_pred_file, 'a') as discr_pred_log:
        discr_pred_log.flush()

        with open(val_log_file, 'a') as val_log:
            val_log.flush()

            # The env variable is on deprecation path, default is set to off.
            #os.environ['TF_SYNC_ON_FINISH'] = '0'
            #os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
            with tf.control_dependencies(update_ops):

                #Session configuration.
                log_device_placement = False #Once placement is correct, this fills up too much of the cmd window...
                sess_config = tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=log_device_placement,
                    intra_op_parallelism_threads=1,
                    gpu_options=tf.GPUOptions(force_gpu_compatible=True))

                config = RunConfig(
                    session_config=sess_config, model_dir=model_dir)

                img1 = input_fn(data_dir1, '', batch_size, num_gpus)
                img2 = input_fn(data_dir2, '', batch_size, num_gpus)

                with tf.Session(config=sess_config) as sess:

                    print("Session started")

                    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                    temp = set(tf.all_variables())

                    __img = sess.run(img)
                    img_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img') 
                                for i in __img]
                    del __img

                    is_training = True

                    adapt_rate1_ph = tf.placeholder(tf.float32, name='adapt_rate1')
                    adapt_rate2_ph = tf.placeholder(tf.float32, name='adapt_rate2')
                    adapt_confusion_ph = tf.placeholder(tf.float32, name='adapt_confusion')

                    #Learning rates
                    lr_gen1_ph, lr_discr1_ph, lr_gen2_ph, lr_discr2_ph, \
                        lr_confuser_ph, lr_super_confuser_ph = [
                        tf.placeholder(tf.float32) for _ in range(6)]

                    flip_fake1_ph, flip_real1_ph, flip_fake2_ph, flip_real2_ph, \
                        flip_confuser1_ph, flip_confuser2_ph = [
                        tf.placeholder(tf.float32) for _ in range(6)]

                    #Mid confuser gradient balancing
                    sc_grad_mean_ph = tf.placeholder(tf.float32, name='sc_grad_mean')
                    c_grad_mean_ph = tf.placeholder(tf.float32, name='c_grad_mean')
                    sc_grad_beta_ph = tf.placeholder(tf.float32, name='sc_grad_beta')
                    c_grad_beta_ph = tf.placeholder(tf.float32, name='c_grad_beta')

                    #To distiller gradient balancing
                    gan1_grad_mean_ph = tf.placeholder(tf.float32, name='gan1_grad_mean')
                    gan2_grad_mean_ph = tf.placeholder(tf.float32, name='gan2_grad_mean')
                    gan1_grad_beta_ph = tf.placeholder(tf.float32, name='gan1_grad_beta')
                    gan2_grad_beta_ph = tf.placeholder(tf.float32, name='gan2_grad_beta')
                    confuser_grad_mean_ph = tf.placeholder(tf.float32, name='confuser_grad_mean')
                    confuser_grad_beta_ph = tf.placeholder(tf.float32, name='confuser_grad_beta')

                    entropy_term1_ph = tf.placeholder(tf.float32, name='entropy_term1')
                    entropy_term2_ph = tf.placeholder(tf.float32, name='entropy_term2')
                    entropy_confusion_ph = tf.placeholder(tf.float32, name='entropy_confusion')

                    #ADAM beta1
                    beta1_distiller_ph = tf.placeholder(tf.float32, shape=(), 
                                                        name='beta1_distiller')
                    beta1_gen1_ph = tf.placeholder(tf.float32, shape=(), 
                                                   name='beta1_gen1')
                    beta1_gen2_ph = tf.placeholder(tf.float32, shape=(), 
                                                   name='beta1_gen2')
                    beta1_discr1_ph = tf.placeholder(tf.float32, shape=(), 
                                                     name='beta1_discr1')
                    beta1_discr2_ph = tf.placeholder(tf.float32, shape=(), 
                                                     name='beta1_discr2')
                    beta1_confuser_ph = tf.placeholder(tf.float32, shape=(), 
                                                       name='beta1_confuser')
                    beta1_super_confuser_ph = tf.placeholder(tf.float32, shape=(), 
                                                             name='beta1_super_confuser')

                    exp_dict = experiment(
                        manifold1, manifold2, lr_gen1_ph, lr_gen2_ph, lr_discr1_ph, lr_discr2_ph, 
                        lr_distiller_ph, lr_confuser_ph, flip_fake1_ph, flip_real1_ph,
                        flip_fake2_ph, flip_real2_ph, flip_confuser1_ph, flip_confuser2_ph,
                        sc_grad_mean_ph, c_grad_mean_ph,
                        sc_grad_beta_ph, c_grad_beta_ph, gan1_grad_mean_ph, gan2_grad_mean_ph,
                        confuser_grad_mean_ph, gan1_grad_beta_ph, gan2_grad_beta_ph,
                        confuser_grad_beta_ph, entropy_term1_ph, entropy_term2_ph,
                        beta1_distiller_ph, beta1_gen1_ph, beta1_gen2_ph, beta1_discr1_ph, 
                        beta1_discr2_ph, beta1_confuser_ph, beta1_super_confuser_ph,
                        adapt_rate1_ph, adapt_rate2_ph, adapt_confusion_ph, entropy_confusion_ph)

                    #########################################################################################

                    sess.run( tf.initialize_variables(set(tf.all_variables()) - temp) )
                    train_writer = tf.summary.FileWriter( logDir, sess.graph )

                    #print(tf.all_variables())
                    saver = tf.train.Saver()
                    #saver.restore(sess, tf.train.latest_checkpoint(model_dir+"model/"))

                    #Store hard examples in a buffer for reuse
                    buffer1_size = 10
                    buffer1_difficulty = 0.
                    hard_buffer1 = [2*np.random.rand(cropsize, cropsize, 1)-1 for _ in range(buffer1_size)]
                    buffer2_size = 10
                    buffer2_difficulty = 0.
                    hard_buffer2 = [2*np.random.rand(cropsize, cropsize, 1)-1 for _ in range(buffer2_size)]

                    avg_p_real1 = 0.5
                    avg_p_fake1 = 0.5
                    avg_p_real2 = 0.5
                    avg_p_fake2 = 0.5
                    avg_p_1_confuser = 0.5
                    avg_p_2_confuser = 0.5
                    b = 0.99 #For prediction decaying avgs

                    bad_buffer_size1 = 15
                    bad_buffer_truth1 = [np.zeros((1, 1, cropsize, cropsize, 1)) for _ in range(bad_buffer_size)]
                    bad_buffer1 = [gen_lq(img.reshape((cropsize, cropsize))).reshape((1, 1, cropsize, cropsize, 1)) 
                                  for img in bad_buffer_truth]
                    bad_buffer_prob1 = 0.1
                    bad_buffer_beta1 = 0.999
                    bad_buffer_thresh1 = 0.05
                    bad_buffer_tracker1 = bad_buffer_prob
                    bad_buffer_tracker_beta1 = 0.999

                    bad_buffer_size2 = 15
                    bad_buffer_truth2 = [np.zeros((1, 1, cropsize, cropsize, 1)) for _ in range(bad_buffer_size)]
                    bad_buffer2 = [gen_lq(img.reshape((cropsize, cropsize))).reshape((1, 1, cropsize, cropsize, 1)) 
                                   for img in bad_buffer_truth]
                    bad_buffer_prob2 = 0.1
                    bad_buffer_beta2 = 0.999
                    bad_buffer_thresh2 = 0.05
                    bad_buffer_tracker2 = bad_buffer_prob
                    bad_buffer_tracker_beta2 = 0.999

                    nat_stat1_mean_beta = 0.999
                    nat_stat2_std_dev_beta1 = 0.999
                    nat_stat_mean1 = 1.
                    nat_stat2_mean1 = 4.

                    nat_stat2_mean_beta = 0.999
                    nat_stat2_std_dev_beta2 = 0.999
                    nat_stat_mean2 = 1.
                    nat_stat2_mean2 = 4.

                    entropy1_bins = 100
                    max_entropy1 = 0.0
                    binned_preds1 = np.ones((entropy1_bins,), dtype=np.float32) / entropy1_bins
                    entropy1_b = 0.99

                    entropy2_bins = 100
                    max_entropy2 = 0.0
                    binned_preds2 = np.ones((entropy2_bins,), dtype=np.float32) / entropy2_bins
                    entropy2_b = 0.99

                    entropy_confuser_bins = 100
                    max_entropy_confuser = 0.0
                    binned_preds_confuser = np.ones((entropy_confuser_bins,), dtype=np.float32) / entropy_confuser_bins
                    entropy_confuser_b = 0.99

                    flip_base = 0.002
                    base_rate = 0.0002

                    counter = 0
                    save_counter = counter
                    counter_init = counter+1
                    max_counter = 20000000

                    while True:
                        #Train for a couple of hours
                        time0 = time.time()

                        if counter < max_counter/2:
                            rate = base_rate
                        elif counter > max_counter:
                            saver.save(sess, save_path=model_dir+"model/", global_step=counter)
                            quit()
                        else:
                            step = (counter-max_counter/2) // 50000 + 1
                            max_step = (max_counter-max_counter/2) // 50000 + 1
                            rate = base_rate*(1.-step/max_step)

                        lr = np.array([0.0002])

                        lr_gen1 = lr_gen2 = lr_discr1 = lr_discr2 = lr_confuser = lr_super_confuser = lr

                        while time.time()-time0 < modelSavePeriod:
                            counter += 1

                            linear_ramp = counter/max_counter
                            quadratic_decay = (1-counter/max_counter)**2

                            #Flipping
                            fake_actv1 = np.float32(flip_base/(avg_p_fake1+0.001))*(1-counter/max_counter)**2
                            flip_fake1 = np.random.rand() > fake_actv1
                            real_actv1 = np.float32(flip_base/(1.-avg_p_real1+0.001))*(1-counter/max_counter)**2
                            flip_real1 = np.random.rand() > real_actv1

                            fake_actv2 = np.float32(flip_base/(avg_p_fake2+0.001))*(1-counter/max_counter)**2
                            flip_fake2 = np.random.rand() > fake_actv2
                            real_actv2 = np.float32(flip_base/(1.-avg_p_real2+0.001))*(1-counter/max_counter)**2
                            flip_real2 = np.random.rand() > real_actv2

                            fake_actv_confuser1 = np.float32(flip_base/(avg_p_fake2+0.001))*(1-counter/max_counter)**2
                            flip_confuser1 = np.random.rand() > fake_actv2
                            real_actv_confuser2 = np.float32(flip_base/(1.-avg_p_real2+0.001))*(1-counter/max_counter)**2
                            flip_confuser2 = np.random.rand() > real_actv2

                            #Adaption
                            adapt1 = np.sqrt(4*avg_p_fake1*(1.-avg_p_real1)) * quadratic_decay + (1.-quadratic_decay)
                            adapt2 = np.sqrt(4*avg_p_fake2*(1.-avg_p_real2)) * quadratic_decay + (1.-quadratic_decay)
                            adapt_confuser = np.sqrt(4*avg_p_1_confuser*(1.-avg_p_2_confuser)) * \
                                quadratic_decay + (1.-quadratic_decay)

                            #Entropy
                            freq = binned_preds1/np.sum(binned_preds1) + 1.e-8
                            entropy = np.sum(-freq*np.log(freq))
                            if entropy > max_entropy1:
                                max_entropy1 = entropy
                            entropy_term1 = (max_entropy1/entropy-1.)**2 * quadratic_decay

                            freq = binned_preds2/np.sum(binned_preds2) + 1.e-8
                            entropy = np.sum(-freq*np.log(freq))
                            if entropy > max_entropy2:
                                max_entropy2 = entropy
                            entropy_term2 = (max_entropy2/entropy-1.)**2 * quadratic_decay

                            freq = binned_preds_confuser/np.sum(binned_preds_confuser) + 1.e-8
                            entropy = np.sum(-freq*np.log(freq))
                            if entropy > max_entropy_confuser:
                                max_entropy_confuser = entropy
                            entropy_term_confuser = (max_entropy_confuser/entropy-1.)**2 * quadratic_decay

                            beta1 = 0.9 if counter < max_counter/2 else 0.5

                            confuser_bal_beta = 0.997
                            gen1_bal_beta = 0.997
                            gen2_bal_beta = 0.997

                            beta1_distiller = beta1_gen1 = beta1_gen2 = beta1_discr1 \
                                = beta1_discr2 = beta1_confuser = beta1_super_confuser = beta1

                            if np.random.rand() < bad_buffer_prob1:
                                _img1 = bad_buffer1[np.random.randint(0, buffer_size1)]
                            else:
                                _img1 = sess.run(img1[0])
                            if np.random.rand() < bad_buffer_prob2:
                                _img2 = bad_buffer2[np.random.randint(0, buffer_size2)]
                            else:
                                _img2 = sess.run(img2[0])

                            feed_dict = {lr_gen1_ph: np.array([lr_gen1]),
                                         lr_gen2_ph: np.array([lr_gen2]),
                                         lr_discr1_ph: np.array([lr_discr1]),
                                         lr_discr2_ph: np.array([lr_discr2]),
                                         lr_confuser_ph: np.array([lr_confuser]),
                                         lr_super_confuser_ph: np.array([lr_super_confuser]),
                                         flip_fake1_ph: np.bool(flip_fake1),
                                         flip_real1_ph: np.bool(flip_real1),
                                         flip_fake2_ph: np.bool(flip_fake2),
                                         flip_real2_ph: np.bool(flip_real2),
                                         flip_confuser1_ph: np.bool(flip_confuser1),
                                         flip_confuser2_ph: np.bool(flip_confuser2),
                                         adapt_rate1_ph: np.float32(adapt1),
                                         adapt_rate2_ph: np.float32(adapt2),
                                         adapt_confusion_ph: np.float32(adapt_confuser),
                                         entropy_term1_ph: np.float32(entropy_term1),
                                         entropy_term2_ph: np.float32(entropy_term2),
                                         entropy_confusion_ph: np.float32(entropy_confusion),
                                         sc_grad_mean_ph: np.float32(sc_grad_mean),
                                         c_grad_mean_ph: np.float32(c_grad_mean),
                                         sc_grad_beta_ph: np.float32(sc_grad_beta),
                                         c_grad_beta_ph: np.float32(c_grad_beta),
                                         gan1_grad_mean_ph: np.float32(gan1_grad_mean),
                                         gan2_grad_mean_ph: np.float32(gan2_grad_mean),
                                         gan1_grad_beta_ph: np.float32(gan1_grad_beta),
                                         gan2_grad_beta_ph: np.float32(gan2_grad_beta),
                                         confuser_grad_mean_ph: np.float32(confuser_grad_mean),
                                         confuser_grad_beta_ph: np.float32(confuser_grad_beta),
                                         beta1_distiller_ph: np.float32(beta1_distiller),
                                         beta1_gen1_ph: np.float32(beta1_gen1),
                                         beta1_gen2_ph: np.float32(beta1_gen2),
                                         beta1_discr1_ph: np.float32(beta1_discr1),
                                         beta1_discr2_ph: np.float32(beta1_discr2),
                                         beta1_confuser_ph: np.float32(beta1_confuser),
                                         beta1_super_confuser_ph: np.float32(beta1_super_confuser),
                                         img1_ph[0]: _img1,
                                         img2_ph[0]: _img2}

                            if counter <= 1 or not counter % save_result_every_n_batches or (counter < 10000 and not counter % 1000) or counter == counter_init:

                                _, _, _, prediction_loss, prediction, prediction_real, output_img = sess.run( 
                                    preds_ops+output_ops, feed_dict=feed_dict )
                              
                                try:
                                    save_input_loc = model_dir+"input-"+str(counter)+".tif"
                                    save_output_loc = model_dir+"output-"+str(counter)+".tif"
                                    Image.fromarray((0.5*_img+0.5).reshape(cropsize, cropsize).astype(np.float32)).save( save_input_loc )
                                    Image.fromarray((0.5*output_img+0.5).reshape(cropsize, cropsize).astype(np.float32)).save( save_output_loc )
                                except:
                                    print("Image save failed")
                            else:
                                _, _, _, prediction_loss, prediction, prediction_real = sess.run( 
                                    std_gan_ops, feed_dict=feed_dict )


                            nat_stat_mean1 = (nat_stat1_mean_beta*nat_stat_mean1 + 
                                             (1.-nat_stat1_mean_beta)*nat_stat1_loss)
                            nat_stat_mean2 = (nat_stat2_mean_beta*nat_stat_mean2 + 
                                             (1.-nat_stat2_mean_beta)*nat_stat2_loss)

                            nat_stat2_mean1 = (nat_stat2_std_dev_beta1*nat_stat2_mean1 + 
                                              (1.-nat_stat2_std_dev_beta1)*nat_stat2_loss**2)
                            nat_stat2_mean2 = (nat_stat2_std_dev_beta2*nat_stat2_mean2 + 
                                              (1.-nat_stat2_std_dev_beta2)*nat_stat2_loss**2)

                            nat_stat_std_dev1 = np.sqrt(nat_stat2_mean1 - nat_stat_mean1**2)
                            nat_stat_std_dev2 = np.sqrt(nat_stat2_mean2 - nat_stat_mean2**2)

                            avg_p_fake1 = b*avg_p_fake1 + (1.-b)*fake1_prob
                            avg_p_real1 = b*avg_p_real1 + (1.-b)*real1_prob
                            avg_p_fake2 = b*avg_p_fake2 + (1.-b)*fake2_prob
                            avg_p_real2 = b*avg_p_real2 + (1.-b)*real2_prob

                            avg_p_1_confuser = b*avg_p_1_confuser + (1.-b)*confuser_1_prob
                            avg_p_2_confuser = b*avg_p_2_confuser + (1.-b)*confuser_2_prob

                            #Update entropy
                            binned_preds1 = entropy1_b*binned_preds1
                            idx1 = int(entropy1_bins*rel_prob1)
                            binned_preds1[idx1] += (1.-entropy1_b)

                            binned_preds2 = entropy2_b*binned_preds2
                            idx2 = int(entropy2_bins*rel_prob2)
                            binned_preds2[idx2] += (1.-entropy2_b)

                            binned_preds_confuser = entropy_confuser_b*binned_preds_confuser
                            idx_confuser = int(entropy_confuser_bins*rel_prob_confuser)
                            binned_preds_confuser[idx_confuser] += (1.-entropy_confuser_b)

                            #Decide whether or not to add using natural statistics
                            if nat_stat1_loss > bad_buffer_thresh1:
                                idx = np.random.randint(0, bad_buffer_size1)
                                bad_buffer1[idx] = _img1
                            else:
                                bad_buffer_tracker1 = bad_buffer_tracker_beta1*bad_buffer_tracker1

                            if bad_buffer_tracker1 < bad_buffer_prob1:
                                step = nat_stat_mean1-3*nat_stat_std_dev1
                                bad_buffer_thresh1 = bad_buffer_beta1*bad_buffer_thresh1 + (1.-bad_buffer_beta1)*step

                            if bad_buffer_tracker1 >= bad_buffer_prob1:
                                step = nat_stat_mean1+3*nat_stat_std_dev1
                                bad_buffer_thresh1 = bad_buffer_beta1*bad_buffer_thresh1 + (1.-bad_buffer_beta1)*step

                            if nat_stat2_loss > bad_buffer_thresh2:
                                idx = np.random.randint(0, bad_buffer_size2)
                                bad_buffer2[idx] = _img2
                            else:
                                bad_buffer_tracker2 = bad_buffer_tracker_beta2*bad_buffer_tracker2

                            if bad_buffer_tracker2 < bad_buffer_prob2:
                                step = nat_stat_mean2-3*nat_stat_std_dev2
                                bad_buffer_thresh2 = bad_buffer_beta2*bad_buffer_thresh2 + (1.-bad_buffer_beta2)*step

                            if bad_buffer_tracker2 >= bad_buffer_prob2:
                                step = nat_stat_mean2+3*nat_stat_std_dev2
                                bad_buffer_thresh2 = bad_buffer_beta2*bad_buffer_thresh2 + (1.-bad_buffer_beta2)*step

                            #Update save files
                            try:
                                discr_pred_log.write("Iter: {}, {}".format(counter, float(prediction)))
                                if prediction == 0.5:
                                    saver.restore(sess, tf.train.latest_checkpoint(model_dir+"model/"))
                                    counter = save_counter
                            except:
                                print("Write to discr pred file failed")

                            if not counter % val_skip_n:

                                _img = sess.run(img_val[0])

                                feed_dict.update({flip_prob_fake_ph: pred_avg_d1, flip_prob_real_ph: pred_avg_d2,
                                                    img_ph[0]: _img,
                                                    batch_norm_on_ph: False})

                                _, _, _, prediction_loss, prediction, prediction_real = sess.run( 
                                    std_gan_ops, feed_dict=feed_dict )

                                try:
                                    val_log.write("Iter: {}, {}".format(counter, float(prediction)))
                                except:
                                    print("Write to val log file failed")

                            #Save the model
                            saver.save(sess, save_path=model_dir+"model/", global_step=counter)
                            save_counter = counter
    return 

if __name__ == '__main__':

    main()


