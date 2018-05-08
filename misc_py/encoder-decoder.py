from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import cv2
from scipy.misc import imread

import time

import os, random

import warnings

slim = tf.contrib.slim #For depthwise separable strided atrous convolutions

tf.logging.set_verbosity(tf.logging.DEBUG)

features0 = 32
features1 = 2*features0 #Number of features to use after 1st convolution
features2 = 2*features1 #Number of features after 2nd convolution
features3 = 3*features1 #Number of features after 3rd convolution
features4 = 4*features1 #Number of features after 4th convolution
aspp_filters = features4 #Number of features for atrous convolutional spatial pyramid pooling

aspp_rateSmall = 6
aspp_rateMedium = 12
aspp_rateLarge = 18

trainDir = "E:/stills_hq/train/"
valDir = "E:/stills_hq/val/"
testDir = "E:/stills_hq/test/"

modelSavePeriod = 1 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "E:/models/noise2/"

shuffle_buffer_size = 10000
num_parallel_calls = 6
num_parallel_readers = 6
prefetch_buffer_size = 64

#batch_size = 8 #Batch size to use during training
num_epochs = 1000000 #Dataset repeats indefinitely

logDir = "C:/dump/train/"
log_file = model_dir+"log.txt"
log_every = 1 #Log every _ examples
cumProbs = np.array([]) #Indices of the distribution plus 1 will be correspond to means

#Remove extreme intensities
removeLower = 0.01
removeUpper = 0.01

numMeans = 64
scaleMean = 4 #Each means array index increment corresponds to this increase in the mean
numDynamicGrad = 10 # Number of gradients to calculate for each possible mean when dynamically updating training
lossSmoothingBoxcarSize = 5

#Dimensions of images in the dataset
height = width = 2048
channels = 1 #Greyscale input image

#Sidelength of images to feed the neural network
cropsize = 1024
height_crop = width_crop = cropsize

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
    mu2 = tf.nn.conv2d(img2, window, strides=[1,1,1,1],padding='VALID')
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

####Noise1
### Initial idea: aspp, batch norm + Leaky RELU, residual connection and lower feature numbers
#def architecture(lq, img=None, mode=None):
#    """Atrous convolutional encoder-decoder noise-removing network"""

#    phase = mode == tf.estimator.ModeKeys.TRAIN #phase is true during training
#    concat_axis = 3

#    ##Reusable blocks

#    def conv_block(input, filters, phase=phase):
#        """
#        Convolution -> batch normalisation -> leaky relu
#        phase defaults to true, meaning that the network is being trained
#        """

#        conv_block = tf.layers.conv2d(
#            inputs=input,
#            filters=filters,
#            kernel_size=3,
#            padding="SAME",
#            activation=tf.nn.relu)

#        #conv_block = tf.contrib.layers.batch_norm(
#        #    conv_block, 
#        #    center=True, scale=True, 
#        #    is_training=phase)

#        #conv_block = tf.nn.leaky_relu(
#        #    features=conv_block,
#        #    alpha=0.2)
#        #conv_block = tf.nn.relu(conv_block)

#        return conv_block

#    def aspp_block(input, phase=phase):
#        """
#        Atrous spatial pyramid pooling
#        phase defaults to true, meaning that the network is being trained
#        """

#        #Convolutions at multiple rates
#        conv1x1 = tf.layers.conv2d(
#            inputs=input,
#            filters=aspp_filters,
#            kernel_size=1,
#            padding="same",
#            activation=tf.nn.relu,
#            name="1x1")
#        #conv1x1 = tf.contrib.layers.batch_norm(
#        #    conv1x1, 
#        #    center=True, scale=True, 
#        #    is_training=phase)

#        conv3x3_rateSmall = tf.layers.conv2d(
#            inputs=input,
#            filters=aspp_filters,
#            kernel_size=3,
#            padding="same",
#            dilation_rate=aspp_rateSmall,
#            activation=tf.nn.relu,
#            name="lowRate")
#        #conv3x3_rateSmall = tf.contrib.layers.batch_norm(
#        #    conv3x3_rateSmall, 
#        #    center=True, scale=True, 
#        #    is_training=phase)

#        conv3x3_rateMedium = tf.layers.conv2d(
#            inputs=input,
#            filters=aspp_filters,
#            kernel_size=3,
#            padding="same",
#            dilation_rate=aspp_rateMedium,
#            activation=tf.nn.relu,
#            name="mediumRate")
#        #conv3x3_rateMedium = tf.contrib.layers.batch_norm(
#        #    conv3x3_rateMedium, 
#        #    center=True, scale=True, 
#        #    is_training=phase)

#        conv3x3_rateLarge = tf.layers.conv2d(
#            inputs=input,
#            filters=aspp_filters,
#            kernel_size=3,
#            padding="same",
#            dilation_rate=aspp_rateLarge,
#            activation=tf.nn.relu,
#            name="highRate")
#        #conv3x3_rateLarge = tf.contrib.layers.batch_norm(
#        #    conv3x3_rateLarge, 
#        #    center=True, scale=True, 
#        #    is_training=phase)

#        #Image-level features
#        pooling = tf.nn.pool(
#            input=input,
#            window_shape=(2,2),
#            pooling_type="AVG",
#            padding="SAME",
#            strides=(2, 2))
#        #Use 1x1 convolutions to project into a feature space the same size as the atrous convolutions'
#        pooling = tf.layers.conv2d(
#            inputs=pooling,
#            filters=aspp_filters,
#            kernel_size=1,
#            padding="SAME",
#            name="imageLevel")
#        pooling = tf.image.resize_images(pooling, [64, 64])
#        #pooling = tf.contrib.layers.batch_norm(
#        #    pooling,
#        #    center=True, scale=True,
#        #    is_training=phase)

#        #Concatenate the atrous and image-level pooling features
#        concatenation = tf.concat(
#            values=[conv1x1, conv3x3_rateSmall, conv3x3_rateMedium, conv3x3_rateLarge, pooling],
#            axis=concat_axis)

#        #Reduce the number of channels
#        reduced = tf.layers.conv2d( #Not sure if this is the correct way to reshape...
#            inputs=concatenation,
#            filters=aspp_filters,
#            kernel_size=1,
#            padding="SAME")

#        return reduced


#    def strided_conv_block(input, filters, stride, rate=1, phase=phase):
        
#        return slim.separable_convolution2d(
#            inputs=input,
#            num_outputs=filters,
#            kernel_size=3,
#            depth_multiplier=1,
#            stride=stride,
#            padding='SAME',
#            data_format='NHWC',
#            rate=rate,
#            activation_fn=tf.nn.relu,
#            normalizer_fn=None,
#            normalizer_params=None,
#            weights_initializer=tf.contrib.layers.xavier_initializer(),
#            weights_regularizer=None,
#            biases_initializer=tf.zeros_initializer(),
#            biases_regularizer=None,
#            reuse=None,
#            variables_collections=None,
#            outputs_collections=None,
#            trainable=True,
#            scope=None)


#    def deconv_block(input, filters, phase=phase):
#        '''Transpositionally convolute a feature space to upsample it'''
        
#        deconv_block = tf.layers.conv2d_transpose(
#            inputs=input,
#            filters=filters,
#            kernel_size=3,
#            strides=2,
#            padding="SAME",
#            activation=tf.nn.relu)

#        #deconv_block = tf.contrib.layers.batch_norm(
#        #    deconv_block, 
#        #    center=True, scale=True, 
#        #    is_training=phase)

#        #deconv_block = tf.nn.leaky_relu(
#        #    features=deconv_block,
#        #    alpha=0.2)
#        #deconv_block = tf.nn.relu(deconv_block)

#        return deconv_block

#    '''Model building'''
#    input_layer = tf.reshape(lq, [-1, cropsize, cropsize, channels])

#    #Encoding block 0
#    cnn0_last = conv_block(
#        input=input_layer, 
#        filters=features0)
#    cnn0_strided = strided_conv_block(
#        input=cnn0_last,
#        filters=features0,
#        stride=2)

#    #Encoding block 1
#    cnn1_last = conv_block(
#        input=cnn0_strided, 
#        filters=features1)
#    cnn1_strided = strided_conv_block(
#        input=cnn1_last,
#        filters=features1,
#        stride=2)

#    #Encoding block 2
#    cnn2_last = conv_block(
#        input=cnn1_strided,
#        filters=features2)
#    cnn2_strided = strided_conv_block(
#        input=cnn2_last,
#        filters=features2,
#        stride=2)

#    #Encoding block 3
#    #cnn3 = conv_block(
#    #    input=cnn2_strided,
#    #    filters=features3)
#    #cnn3_last = conv_block(
#    #    input=cnn3,
#    #    filters=features3)
#    cnn3_last = conv_block(
#        input=cnn2_strided,
#        filters=features3)
#    cnn3_strided = strided_conv_block(
#        input=cnn3_last,
#        filters=features3,
#        stride=2)

#    #Encoding block 4
#    #cnn4 = conv_block(
#    #    input=cnn3_strided,
#    #    filters=features4)
#    #cnn4_last = conv_block(
#    #    input=cnn4,
#    #    filters=features4)
#    cnn4_last = conv_block(
#        input=cnn3_strided,
#        filters=features4)

#    #cnn4_strided = split_separable_conv2d(
#    #    inputs=cnn4_last,
#    #    filters=features4,
#    #    rate=2,
#    #    stride=2)

#    #Prepare for aspp
#    aspp_input = strided_conv_block(
#        input=cnn4_last,
#        filters=features4,
#        stride=1,
#        rate=2)
#    aspp_input = conv_block(
#        input=aspp_input,
#        filters=features4)

#    ##Atrous spatial pyramid pooling
#    aspp = aspp_block(aspp_input)

#    #Upsample the semantics by a factor of 4
#    #upsampled_aspp = tf.image.resize_bilinear(
#    #    images=aspp,
#    #    tf.shape(aspp)[1:3],
#    #    align_corners=True)

#    #Decoding block 1 (deepest)
#    deconv4 = conv_block(aspp, features4)
#    #deconv4 = conv_block(deconv4, features4)
    
#    #Decoding block 2
#    deconv4to3 = deconv_block(deconv4, features4)
#    concat3 = tf.concat(
#        values=[deconv4to3, cnn3_last],
#        axis=concat_axis)
#    deconv3 = conv_block(concat3, features3)
#    #deconv3 = conv_block(deconv3, features3)

#    #Decoding block 3
#    deconv3to2 = deconv_block(deconv3, features3)
#    concat2 = tf.concat(
#        values=[deconv3to2, cnn2_last],
#        axis=concat_axis)
#    deconv2 = conv_block(concat2, features2)
    
#    #Decoding block 4
#    deconv2to1 = deconv_block(deconv2, features2)
#    concat1 = tf.concat(
#        values=[deconv2to1, cnn1_last],
#        axis=concat_axis)
#    deconv1 = conv_block(concat1, features1)

#    #Decoding block 5
#    deconv1to0 = deconv_block(deconv1, features1)
#    concat0 = tf.concat(
#        values=[deconv1to0, cnn0_last],
#        axis=concat_axis)
#    deconv1 = conv_block(concat0, features0)

#    #Create final image with 1x1 convolutions
#    deconv_final = tf.layers.conv2d_transpose(
#        inputs=deconv1,
#        filters=1,
#        kernel_size=3,
#        padding="SAME",
#        activation=tf.nn.relu)


#    #Residually connect the input to the output
#    output = deconv_final#+input_layer

#    #Image values will be between 0 and 1
#    output = tf.clip_by_value(
#        output,
#        clip_value_min=0,
#        clip_value_max=1)

#    if phase: #Calculate loss during training
#        ground_truth = tf.reshape(img, [-1, cropsize, cropsize, channels])
#        loss = 1.0-tf_ssim(output, ground_truth)#cropsize*cropsize*tf.reduce_mean(tf.squared_difference(output, ground_truth))
        
#        #tf.log(cropsize*cropsize*tf.reduce_mean(tf.squared_difference(output, ground_truth))+1)
#        #tf.summary.histogram("loss", loss)
#    else:
#        loss = -1

#    return loss, output

###Second noise architecture
###More convolutions between strides
def architecture(lq, img=None, mode=None):
    """Atrous convolutional encoder-decoder noise-removing network"""

    phase = mode == tf.estimator.ModeKeys.TRAIN #phase is true during training
    concat_axis = 3

    ##Reusable blocks

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

        #conv_block = tf.contrib.layers.batch_norm(
        #    conv_block, 
        #    center=True, scale=True, 
        #    is_training=phase)

        #conv_block = tf.nn.leaky_relu(
        #    features=conv_block,
        #    alpha=0.2)
        #conv_block = tf.nn.relu(conv_block)

        return conv_block

    def aspp_block(input, phase=phase):
        """
        Atrous spatial pyramid pooling
        phase defaults to true, meaning that the network is being trained
        """

        #Convolutions at multiple rates
        conv1x1 = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=1,
            padding="same",
            activation=tf.nn.relu,
            name="1x1")
        #conv1x1 = tf.contrib.layers.batch_norm(
        #    conv1x1, 
        #    center=True, scale=True, 
        #    is_training=phase)

        conv3x3_rateSmall = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateSmall,
            activation=tf.nn.relu,
            name="lowRate")
        #conv3x3_rateSmall = tf.contrib.layers.batch_norm(
        #    conv3x3_rateSmall, 
        #    center=True, scale=True, 
        #    is_training=phase)

        conv3x3_rateMedium = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateMedium,
            activation=tf.nn.relu,
            name="mediumRate")
        #conv3x3_rateMedium = tf.contrib.layers.batch_norm(
        #    conv3x3_rateMedium, 
        #    center=True, scale=True, 
        #    is_training=phase)

        conv3x3_rateLarge = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateLarge,
            activation=tf.nn.relu,
            name="highRate")
        #conv3x3_rateLarge = tf.contrib.layers.batch_norm(
        #    conv3x3_rateLarge, 
        #    center=True, scale=True, 
        #    is_training=phase)

        #Image-level features
        pooling = tf.nn.pool(
            input=input,
            window_shape=(2,2),
            pooling_type="AVG",
            padding="SAME",
            strides=(2, 2))
        #Use 1x1 convolutions to project into a feature space the same size as the atrous convolutions'
        pooling = tf.layers.conv2d(
            inputs=pooling,
            filters=aspp_filters,
            kernel_size=1,
            padding="SAME",
            name="imageLevel")
        pooling = tf.image.resize_images(pooling, [64, 64])
        #pooling = tf.contrib.layers.batch_norm(
        #    pooling,
        #    center=True, scale=True,
        #    is_training=phase)

        #Concatenate the atrous and image-level pooling features
        concatenation = tf.concat(
            values=[conv1x1, conv3x3_rateSmall, conv3x3_rateMedium, conv3x3_rateLarge, pooling],
            axis=concat_axis)

        #Reduce the number of channels
        reduced = tf.layers.conv2d( #Not sure if this is the correct way to reshape...
            inputs=concatenation,
            filters=aspp_filters,
            kernel_size=1,
            padding="SAME")

        return reduced


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
            normalizer_fn=None,
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
        
        deconv_block = tf.layers.conv2d_transpose(
            inputs=input,
            filters=filters,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)

        #deconv_block = tf.contrib.layers.batch_norm(
        #    deconv_block, 
        #    center=True, scale=True, 
        #    is_training=phase)

        #deconv_block = tf.nn.leaky_relu(
        #    features=deconv_block,
        #    alpha=0.2)
        #deconv_block = tf.nn.relu(deconv_block)

        return deconv_block

    '''Model building'''
    input_layer = tf.reshape(lq, [-1, cropsize, cropsize, channels])

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

    #Encoding block 3
    cnn3 = conv_block(
        input=cnn2_strided,
        filters=features3)
    cnn3 = conv_block(
        input=cnn3,
        filters=features3)
    cnn3_last = conv_block(
        input=cnn3,
        filters=features3)
    cnn3_strided = strided_conv_block(
        input=cnn3_last,
        filters=features3,
        stride=2)

    #Encoding block 4
    cnn4 = conv_block(
        input=cnn3_strided,
        filters=features4)
    cnn4 = conv_block(
        input=cnn4,
        filters=features4)
    cnn4_last = conv_block(
        input=cnn4,
        filters=features4)

    #cnn4_strided = split_separable_conv2d(
    #    inputs=cnn4_last,
    #    filters=features4,
    #    rate=2,
    #    stride=2)

    #Prepare for aspp
    aspp_input = strided_conv_block(
        input=cnn4_last,
        filters=features4,
        stride=1,
        rate=2)
    aspp_input = conv_block(
        input=aspp_input,
        filters=features4)

    ##Atrous spatial pyramid pooling
    aspp = aspp_block(aspp_input)

    #Upsample the semantics by a factor of 4
    #upsampled_aspp = tf.image.resize_bilinear(
    #    images=aspp,
    #    tf.shape(aspp)[1:3],
    #    align_corners=True)

    #Decoding block 1 (deepest)
    deconv4 = conv_block(aspp, features4)
    deconv4 = conv_block(deconv4, features4)
    deconv4 = conv_block(deconv4, features4)
    
    #Decoding block 2
    deconv4to3 = deconv_block(deconv4, features4)
    concat3 = tf.concat(
        values=[deconv4to3, cnn3_last],
        axis=concat_axis)
    deconv3 = conv_block(concat3, features3)
    deconv3 = conv_block(deconv3, features3)
    deconv3 = conv_block(deconv3, features3)

    #Decoding block 3
    deconv3to2 = deconv_block(deconv3, features3)
    concat2 = tf.concat(
        values=[deconv3to2, cnn2_last],
        axis=concat_axis)
    deconv2 = conv_block(concat2, features2)
    deconv2 = conv_block(deconv2, features2)
    
    #Decoding block 4
    deconv2to1 = deconv_block(deconv2, features2)
    concat1 = tf.concat(
        values=[deconv2to1, cnn1_last],
        axis=concat_axis)
    deconv1 = conv_block(concat1, features1)
    deconv1 = conv_block(deconv1, features1)

    #Decoding block 5
    deconv1to0 = deconv_block(deconv1, features1)
    concat0 = tf.concat(
        values=[deconv1to0, cnn0_last],
        axis=concat_axis)
    deconv0 = conv_block(concat0, features0)
    deconv0 = conv_block(deconv0, features0)

    #Create final image with 1x1 convolutions
    deconv_final = tf.layers.conv2d_transpose(
        inputs=deconv0,
        filters=1,
        kernel_size=3,
        padding="SAME",
        activation=tf.nn.relu)


    #Residually connect the input to the output
    output = deconv_final#+input_layer

    #Image values will be between 0 and 1
    output = tf.clip_by_value(
        output,
        clip_value_min=0,
        clip_value_max=1)

    if phase: #Calculate loss during training
        ground_truth = tf.reshape(img, [-1, cropsize, cropsize, channels])
        loss = 1.0-tf_ssim(output, ground_truth)#cropsize*cropsize*tf.reduce_mean(tf.squared_difference(output, ground_truth))
        
        #tf.log(cropsize*cropsize*tf.reduce_mean(tf.squared_difference(output, ground_truth))+1)
        #tf.summary.histogram("loss", loss)
    else:
        loss = -1

    return loss, output

def load_image(addr, resizeSize=None, imgType=np.float32):
    """Read an image and make sure it is of the correct type. Optionally resize it"""
    
    img = imread(addr, mode='F')
    if resizeSize:
        img = cv2.resize(img, resizeSize, interpolation=cv2.INTER_CUBIC)
    img = img.astype(imgType)

    return img


def scale0to1(img):
    """Rescale image between 0 and 1"""

    min = np.min(img)
    max = np.max(img)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)


def gen_lq(img, scale):
    '''Generate low quality image'''

    #Ensure that the seed is random
    np.random.seed(int(np.random.rand()*(2**32-1)))

    #Adjust the image scale so that the image has the
    # correct average counts
    lq = np.random.poisson( img * scale )

    return scale0to1(lq)


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
    """
    Threshold the image to remove dead or very bright pixels.
    Then crop a region of the image of a random size and resize it.
    """

    sorted = np.sort(img, axis=None)
    min = sorted[int(removeLower*sorted.size)]
    max = sorted[int((1.0-removeUpper)*sorted.size)]

    size = int(cropsize + np.random.rand()*(height-cropsize))
    topLeft_x = int(np.random.rand()*(height-size))
    topLeft_y = int(np.random.rand()*(height-size))

    crop = np.clip(img[topLeft_y:(topLeft_y+cropsize), topLeft_x:(topLeft_x+cropsize)], min, max)

    resized = cv2.resize(crop, (cropsize, cropsize), interpolation=cv2.INTER_AREA)

    resized[np.isnan(resized)] = 0.5
    resized[np.isinf(resized)] = 0.5

    return scale0to1(flip_rotate(resized))


def get_scale():
    """Generate a mean from the cumulative probability distribution"""

    
    return 0.5


def parser(record, dir):
    """Parse files and generate lower quality images from them"""

    with warnings.catch_warnings():
        try:
            img = load_image(record)
            img = preprocess(img)

            scale = get_scale()
            lq = gen_lq(img, scale)

            img = (np.mean(lq) * img / np.mean(img)).clip(0.0, 1.0)

            #cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
            #cv2.imshow("dfsd", lq)
            #cv2.waitKey(0)
            #cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
            #cv2.imshow("dfsd", img)
            #cv2.waitKey(0)

        except RuntimeWarning as e:
            print("Catching this RuntimeWarning is getting personal...")
            print(e)
            lq, img = parser(dir+random.choice(os.listdir(dir)), dir)

    return lq, img


def input_fn(dir):
    """Create a dataset from a list of filenames"""

    dataset = tf.data.Dataset.list_files(dir+"*.tif")
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(
        lambda file: tuple(tf.py_func(parser, [file, dir], [tf.float32, tf.float32])),
        num_parallel_calls=num_parallel_calls)
    #dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    dataset = dataset.repeat(num_epochs)
    
    iter = dataset.make_one_shot_iterator()

    lq, img = iter.get_next()

    return lq, img


def movingAverage(values, window):

    weights = np.repeat(1.0, window)/window
    ma = np.convolve(values, weights, 'same')

    return ma


def get_training_probs(losses0, losses1):
    """
    Returns cumulative probabilities of means being selected for loq-quality image syntheses
    losses0 - previous losses (smoothed)
    losses1 - losses after the current training run
    """

    diffs = movingAverage(losses0, lossSmoothingBoxcarSize) - movingAverage(losses1, lossSmoothingBoxcarSize)
    diffs[diffs < 0] = 0
    max_diff = np.max(diffs)
    
    if max_diff == 0:
        max_diff = 1

    diffs += 0.05*max_diff
    cumDiffs = np.cumsum(diffs)
    cumProbs = cumDiffs / np.max(cumDiffs, axis=None)

    return cumProbs.astype(np.float32)


def main(unused_argv=None):

    temp = set(tf.all_variables())

    log = open(log_file, 'a')

    #with tf.device("/gpu:0"):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
    with tf.control_dependencies(update_ops):

        lq, img = input_fn(trainDir)

        loss, prediction = architecture(lq, img, tf.estimator.ModeKeys.TRAIN)
        train_op = tf.train.AdamOptimizer().minimize(loss)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        #config.gpu_options.per_process_gpu_memory_fraction = 0.7

        #saver = tf.train.Saver(max_to_keep=-1)

        tf.add_to_collection("train_op", train_op)
        tf.add_to_collection("update_ops", update_ops)
        with tf.Session(config=config) as sess: #Alternative is tf.train.MonitoredTrainingSession()

            init = tf.global_variables_initializer()

            sess.run(init)
            sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

            train_writer = tf.summary.FileWriter( logDir, sess.graph )

            #Set up mean probabilities to be dynamically adjusted during training
            probs = np.ones(numMeans, dtype=np.float32)
            losses0 = np.empty([])
            global cumProbs
            cumProbs = np.cumsum(probs)
            cumProbs /= np.max(cumProbs)

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
                    print("Iter: {}, Loss: {:.6f}".format(counter, loss_value))
                    log.write("Iter: {}, Loss: {:.6f}".format(counter, loss_value))
                    #train_writer.add_summary(summary, counter)

                #Save the model
                #saver.save(sess, save_path=model_dir+"model", global_step=counter)
                tf.saved_model.simple_save(
                    session=sess,
                    export_dir=model_dir+"model-"+str(counter)+"/",
                    inputs={"lq": lq},
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

                #cumProbs = get_training_probs(losses0, losses)
                losses0 = losses
    return 

if __name__ == "__main__":
    tf.app.run()
