import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import numpy as np
import scipy.misc
import scipy.io
import math
import tensorflow as tf
from sys import stderr
from functools import reduce

import functools
import itertools

import collections
import six

import cv2

from scipy.misc import imread

slim = tf.contrib.slim

num_gpus = 1

batch_size = 1
shuffle_buffer_size = 5000
num_epochs = 10000 #Large number so the dataset effectively repeats forever
num_parallel_calls = 2
num_parallel_readers = 2
prefetch_buffer_size = 2

cropsize = 224
channels = 1

iters_with_batch = 60000
iters_with_batch_frozen = 60000

step_learning_rate_every_n_batch = 5000
step_learning_rate_every_n_batch_frozen = 5000

initial_learning_rate_batch = 0.005
initial_learning_rate_batch_frozen = 0.001

learning_rates_batch = [initial_learning_rate_batch*(
    1-i*step_learning_rate_every_n_batch/iters_with_batch)**0.9 
                        for i in range(iters_with_batch//step_learning_rate_every_n_batch)]
learning_rates_batch_frozen = [initial_learning_rate_batch_frozen*(
    1-i*step_learning_rate_every_n_batch_frozen/iters_with_batch_frozen)**0.9 
                  for i in range(iters_with_batch_frozen//step_learning_rate_every_n_batch_frozen)]

def architecture(inputs, train_batch_norm_ph, phase=False, params=None):
    """Fast style fustion neural network architecture"""

    batch_decay_gen = 0.9997

    features1 = 32
    features2 = 64
    features3 = 128

    num_enhancer_blocks = 5

    with tf.variable_scope("architecture"):

        concat_axis = 3

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
                decay=batch_decay_gen,
                center=True, scale=True,
                is_training=train_batch_norm_ph,
                fused=True,
                zero_debias_moving_mean=False,
                renorm=False)
            return batch_norm

        def batch_then_activ(input):
            batch_then_activ = _batch_norm_fn(input)
            batch_then_activ = tf.nn.relu(batch_then_activ)
            return batch_then_activ

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

        def skip2_middle_block(input, features, pad_size=None):
        
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
        input_layer = tf.reshape(inputs, [-1, cropsize, cropsize, channels])

        enc = strided_conv_block(input=input_layer,
                                 filters=features1,
                                 stride=1,
                                 kernel_size = 9,
                                 pad_size=(4,4))

        enc = strided_conv_block(enc, features2, 2, 1, pad_size=(1,1))
        enc = strided_conv_block(enc, features3, 2, 1, pad_size=(1,1))

        for _ in range(num_enhancer_blocks):
            enc = skip2_middle_block(enc, features3, pad_size=(1,1))

        enc = deconv_block(enc, features2, new_size=(cropsize//2, cropsize//2), pad_size=(1,1))
        enc = deconv_block(enc, features1, new_size=(cropsize, cropsize), pad_size=(1,1))

        enc = conv_block_not_sep(enc, 1, pad_size=(4,4), kernel_size=9, batch_plus_activ=False)

        enc = 150.*tf.tanh(enc)+255./2.

        return enc


def train_fast_guided_style_fusion(img_style,
                   model_fn,
                   img_mask=None, 
                   diff2_weight=None, 
                   weight_style=2.e2,
                   rel_style=None,
                   rel_mask=None,
                   input_noise=0.1,
                   save_checkpoint_imgs_every_n=None,
                   layers_style_weights = [0.2,0.2,0.2,0.2,0.2],
                   dont_use_n_pixels=5,
                   regularization_coeff=1.e-5,
                   save_result_every_n_batches=5000,
                   path_VGG19="",#'D:/imagenet-vgg-verydeep-19.mat',
                   model_dir="",
                   data_dir=""):
    """
    Guide the training of fast neural style fusion transfer by weighting the contributions of the 
    syle images by how similar they are to the image being restyled
    img_content: Iterator for images to preserve the content of
    img_style: Image to restyle the content image with. Use list for multiple
    model_fn: Fast neural style transfer architecture. It should accept an input image and return 
    a restyled image
    img_mask: Optional mask where 1.0s indicate content to be conserved in its original.
    Use list for multiple
    form and 0.0s indicate content to restyle without this bias
    weight_mask: Weight of optional masking to loss
    weight_style: Weight of style loss
    rel_content: Relative contribution of each content image. Only needed for multiple
    rel_style: Relative contribution of each style image. Only needed for multiple
    rel_mask: Relative contribtion each mask region in case of overlap. Only needed for multiple
    input_noise: Proportion of noise to add to the content image when creating canvas
    to begin iterations from
    save_checkpoint_imgs_every_n: Optionally set to true to save images at checkpoints. Useful
    to study convergence
    model_dir: Location to save checkpoints to
    layers_style_weights: 5 element array with proportional contributions of VGG19 layers
    to style loss
    dont_use_n_pixels: Take random crops that are the size of the image minus this. This 
    improves training by reducing artifacts
    save_result_every_n_batches: Frequecy with which to save input-output image pairs. Set to None
    to disable
    """

    ## Layers
    layer_content = ['conv4_2'] 
    layers_style = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    # VGG19 mean for standardisation (RGB)
    VGG19_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

    print("Establishing dataflow...")

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
    
        print(addr)
        try:
            img = imread(addr, mode='F')
        except:
            img = np.zeros((512,512))
            print("Image read failed")

        if img.shape != (512, 512):
            img = np.zeros((512,512))
            print("Image had wrong shape")

        if resizeSize:
            img = cv2.resize(img, resizeSize, interpolation=cv2.INTER_AREA)

        return img.astype(img_type)


    def scale0to1(img):
        """Rescale image between 0 and 1"""

        min = np.min(img)
        max = np.max(img)

        if np.absolute(min-max) < 1.e-6:
            img.fill(0.5)
        else:
            img = (img-min) / (max-min)

        return img.astype(np.float32)


    def preprocess(img, shape=(cropsize, cropsize)):

        img[np.isnan(img)] = 0.5
        img[np.isinf(img)] = 0.5

        img = cv2.resize(img, shape)

        img = scale0to1(img)

        return img

    def record_parser(record):
        """Parse files and generate lower quality images from them"""

        img = flip_rotate(preprocess(load_image(record)))

        return img

    def reshaper(img1):
        img1 = tf.reshape(img1, [cropsize, cropsize, channels])
        return img1

    def input_fn(dir, subset, batch_size, num_shards):
        """Create a dataset from a list of filenames and shard batches from it"""

        with tf.device('/cpu:0'):

            dataset = tf.data.Dataset.list_files(dir+subset+"/"+"*.tif")
            dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
            dataset = dataset.repeat(num_epochs)
            dataset = dataset.map(
                lambda file: tf.py_func(record_parser, [file], [tf.float32]),
                num_parallel_calls=num_parallel_calls)
            #print(dataset.output_shapes, dataset.output_types)
            dataset = dataset.map(reshaper, num_parallel_calls=num_parallel_calls)
            #print(dataset.output_shapes, dataset.output_types)
            dataset = dataset.batch(batch_size=batch_size)
            dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

            iter = dataset.make_one_shot_iterator()
            img_batch = iter.get_next()

            if num_shards <= 1:
                return [img_batch[0]]
            else:
                raise "Code needs to be adjusted for multi-gpu"

                #image_batch = tf.unstack(img_batch, num=batch_size, axis=1)
                #feature_shards = [[] for i in range(num_shards)]
                #feature_shards_truth = [[] for i in range(num_shards)]
                #for i in range(batch_size):
                #    idx = i % num_shards
                #    tensors = tf.unstack(image_batch[i], num=2, axis=0)
                #    feature_shards[idx].append(tensors[0])
                #    feature_shards_truth[idx].append(tensors[1])
                #feature_shards = [tf.parallel_stack(x) for x in feature_shards]
                #feature_shards_truth = [tf.parallel_stack(x) for x in feature_shards_truth]

                #return feature_shards, feature_shards_truth


    img = input_fn(data_dir, 'train', batch_size=1, num_shards=num_gpus)
    img_val = input_fn(data_dir, 'val', batch_size=1, num_shards=num_gpus)

    print("Starting preprocessing...")

    #Convert inputs to lists if they are single images

    if not isinstance(img_style, (list,)):
        img_style = [img_style]

    #Normalise relative contents so that they sum to 1.
    if rel_mask:
        sum = np.sum(np.asarray(rel_mask))
        rel_mask = [x/sum for x in rel_mask]

    ### Helper functions
    def imread(path):
        return scipy.misc.imread(path).astype(np.float)   # returns RGB format

    def imsave(path, img):
        img = np.clip(img, 0, 255).astype(np.uint8)
        scipy.misc.imsave(path, img)
    
    def imgpreprocess(image):
        image = image[np.newaxis,:,:,:]
        return image - VGG19_mean

    def imgunprocess(image):
        temp = image + VGG19_mean
        return temp[0] 

    #Function to convert 2D greyscale to 3D RGB
    def to_rgb(im):
        if len(im.shape)==2 or im.shape[2] != 3:
            w, h = im.shape
            ret = np.empty((w, h, 3), dtype=np.float32)
            ret[:, :, 0] = im
            ret[:, :, 1] = im
            ret[:, :, 2] = im
            return ret
        else:
            return im

    ### Preprocessing
    img_style = [255*preprocess(img,(cropsize, cropsize)).reshape(1,cropsize,cropsize,1) 
                 for img in img_style]

    #### BUILD VGG19 MODEL
    ## with thanks to http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style

    data_dict = np.load(path_VGG19, encoding='latin1').item()

    def get_conv_filter(name):
        return tf.constant(data_dict[name][0], name="filter")

    def get_bias(name):
        return tf.constant(data_dict[name][1], name="biases")

    def get_fc_weight(name):
        return tf.constant(data_dict[name][0], name="weights")

    # help functions
    def _conv2d_relu(prev_layer, layer_name, reuse_name=False):

        filt = get_conv_filter(layer_name) if not reuse_name else tf.get_variable(reuse_name[0])
        b = get_bias(layer_name) if not reuse_name else tf.get_variable(reuse_name[1])

        conv2d = tf.nn.conv2d(prev_layer, filter=filt, strides=[1, 1, 1, 1], padding='SAME') + b    

        activated = tf.nn.relu(conv2d)

        return activated, [filt.name, b.name]

    def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer, 
                              ksize=[1, 2, 2, 1], 
                              strides=[1, 2, 2, 1], 
                              padding='SAME')

    def _maxpool(prev_layer):
        return tf.nn.max_pool(prev_layer, 
                              ksize=[1, 2, 2, 1], 
                              strides=[1, 2, 2, 1], 
                              padding='SAME')

    def _fc_layer(prev_layer, layer_name, add_relu=True, reuse_name=None):

            shape = prev_layer.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(prev_layer, [-1, dim])

            weights = get_fc_weight(layer_name) if not reuse_name else tf.get_variable(reuse_name[0])
            b = get_bias(layer_name) if not reuse_name else tf.get_variable(reuse_name[1])

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, weights), b)

            if add_relu:
                fc = tf.nn.relu(fc)

            return fc, [weights.name, b.name]

    print("Loading fast neural style fusion net...")

    # Setup trainable network
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
    with tf.control_dependencies(update_ops):

        with tf.Session() as sess:
        
            img_ph = tf.placeholder(tf.float32, 
                                    (1, cropsize, cropsize, channels), 
                                    name="img")

            train_batch_norm_ph = tf.placeholder(tf.bool, 
                                                 name="train_batch_norm")

            pred = model_fn(img_ph, train_batch_norm_ph, phase=True)

            #Prepare input for VGG-19
            if cropsize != 224:
                input = tf.image.resize_images(pred, (224, 224))
            else:
                input = pred
            input = tf.concat([input-mean for mean in VGG19_mean], axis=3)

            tower_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                             scope="architecture")

    print("Loading VGG-19...")

    #Restore VGG-19
    with tf.Session() as sess:

        def vgg19(input, scope="VGG-19", reuse_name=None):

            with tf.device('/gpu:0'):
                with tf.variable_scope(scope, reuse=bool(reuse_name)) as net_scope:

                    net = {}
                    var_names = {}
                    net['input'] = input
                    net['conv1_1'], var_names['conv1_1']  = _conv2d_relu(input, 'conv1_1', reuse_name['conv1_1'] if reuse_name else None)
                    net['conv1_2'], var_names['conv1_2']  = _conv2d_relu(net['conv1_1'], 'conv1_2', reuse_name['conv1_2'] if reuse_name else None)
                    net['avgpool1'] = _avgpool(net['conv1_2'])
                    net['conv2_1'], var_names['conv2_1']  = _conv2d_relu(net['avgpool1'], 'conv2_1', reuse_name['conv2_1'] if reuse_name else None)
                    net['conv2_2'], var_names['conv2_2']  = _conv2d_relu(net['conv2_1'], 'conv2_2', reuse_name['conv2_2'] if reuse_name else None)
                    net['avgpool2'] = _avgpool(net['conv2_2'])
                    net['conv3_1'], var_names['conv3_1']  = _conv2d_relu(net['avgpool2'], 'conv3_1', reuse_name['conv3_1'] if reuse_name else None)
                    net['conv3_2'], var_names['conv3_2']  = _conv2d_relu(net['conv3_1'], 'conv3_2', reuse_name['conv3_2'] if reuse_name else None)
                    net['conv3_3'], var_names['conv3_3']  = _conv2d_relu(net['conv3_2'], 'conv3_3', reuse_name['conv3_3'] if reuse_name else None)
                    net['conv3_4'], var_names['conv3_4']  = _conv2d_relu(net['conv3_3'], 'conv3_4', reuse_name['conv3_4'] if reuse_name else None)
                    net['avgpool3'] = _avgpool(net['conv3_4'])
                    net['conv4_1'], var_names['conv4_1']  = _conv2d_relu(net['avgpool3'], 'conv4_1', reuse_name['conv4_1'] if reuse_name else None)
                    net['conv4_2'], var_names['conv4_2']  = _conv2d_relu(net['conv4_1'], 'conv4_2', reuse_name['conv4_2'] if reuse_name else None)     
                    net['conv4_3'], var_names['conv4_3']  = _conv2d_relu(net['conv4_2'], 'conv4_3', reuse_name['conv4_3'] if reuse_name else None)
                    net['conv4_4'], var_names['conv4_4']  = _conv2d_relu(net['conv4_3'], 'conv4_4', reuse_name['conv4_4'] if reuse_name else None)
                    net['avgpool4'] = _avgpool(net['conv4_4'])
                    net['conv5_1'], var_names['conv5_1']  = _conv2d_relu(net['avgpool4'], 'conv5_1', reuse_name['conv5_1'] if reuse_name else None)
                    net['conv5_2'], var_names['conv5_2']  = _conv2d_relu(net['conv5_1'], 'conv5_2', reuse_name['conv5_2'] if reuse_name else None)
                    net['conv5_3'], var_names['conv5_3']  = _conv2d_relu(net['conv5_2'], 'conv5_3', reuse_name['conv5_3'] if reuse_name else None)
                    net['conv5_4'], var_names['conv5_4']  = _conv2d_relu(net['conv5_3'], 'conv5_4', reuse_name['conv5_4'] if reuse_name else None)
                    net['maxpool5'] = _maxpool(net['conv5_4'])
                    net['fc6'], var_names['fc6'] = _fc_layer(net['maxpool5'], 'fc6', reuse_name['fc6'] if reuse_name else None)
                    net['fc7'], var_names['fc7'] = _fc_layer(net['fc6'], 'fc7', reuse_name['fc7'] if reuse_name else None)
                    net['fc8'], var_names['fc8'] = _fc_layer(net['fc7'], 'fc8', reuse_name['fc8'] if reuse_name else None)

                    var_names = {key: var_names[key][(len("VGG-19")+1):] for key in var_names}

            return net, net_scope, var_names

        #Prepare input for VGG-19
        if cropsize != 224:
            img_224 = tf.image.resize_images(img_ph, (224, 224))
        else:
            img_224 = img_ph
        img_224 = tf.concat([img_224-mean for mean in VGG19_mean], axis=3)

        vgg_19_net, scope, var_names = vgg19(input=img_224)

        print("Established straight-through")

        #Flow the preditions of the fast neural style fusion network through VGG-19
        net, _, _ = vgg19(input=input, scope=scope, reuse_name=var_names)

        print("Estrablished through-fast-transfer")

    print("Preparing losses...")

    ### CONTENT LOSS: FUNCTION TO CALCULATE AND INSTANTIATION
    #With thanks to https://github.com/cysmith/neural-style-tf

    #Recode to be simpler: http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
    def content_layer_loss(p, x):
        _, h, w, d = [i.value for i in p.get_shape()]    # d: number of filters; h,w : height, width
        M = h * w 
        N = d
        K = 1. / (2. * N**0.5 * M**0.5)
        loss = K * tf.reduce_sum(tf.pow((x - p), 2))
        return loss

    content_loss = tf.add_n([content_layer_loss(vgg_19_net[content], net[content]) 
                             for content in layer_content]) / len(layer_content)

    ###STYLE LOSS: FUNCTION TO CALCULATE AND INSTANTIATION

    def style_layer_loss(a, x):
        _, h, w, d = [i.value for i in a.get_shape()]
        M = h * w 
        N = d 
        A = gram_matrix(a, M, N)
        G = gram_matrix(x, M, N)
        loss = (1./(4 * N**2 * M**2)) * tf.reduce_sum(tf.pow((G - A), 2))
        return loss

    def gram_matrix(x, M, N):
        F = tf.reshape(x, (M, N))                   
        G = tf.matmul(tf.transpose(F), F)
        return G

    def style_guide_weights(logits, style_logits_list):

        cos_theta = [tf.reduce_sum(tf.multiply(logits, style_logits)) / 
                     tf.sqrt( tf.reduce_sum(logits**2)*tf.reduce_sum(style_logits**2) ) 
                     for style_logits in style_logits_list]
        thetas = [tf.acos(theta) for theta in cos_theta]
        thetas = [tf.cond(angle <= np.pi, lambda: tf.abs(angle), lambda: tf.abs(angle-2.*np.pi)) 
                  for angle in thetas]
        
        #Softmax the similarities
        similarities = [tf.cond(angle <= np.pi/2, lambda: np.pi/2-angle, lambda: angle-np.pi/2) 
                        for angle in thetas]
        similarities = [tf.exp(sim) for sim in similarities]
        sum = tf.reduce_sum(tf.concat(similarities, axis=0))
        similarities = [sim/sum for sim in similarities]

        return similarities

    #Get logits for the style images
    with tf.Session() as sess:

        logits_list = []
        for style in img_style:
            logits = sess.run(vgg_19_net['fc7'], 
                              feed_dict={img_ph: style})
            logits = tf.constant(logits, dtype=tf.float32, shape=(1000,))
            logits_list.append(logits)

    #Calculate the style loss
    with tf.Session() as sess:

        #Get style guiding weights
        logits = net['fc7']
        style_tensors = [net[layer] for layer in layers_style]
        rel_style = style_guide_weights(logits, logits_list)

        style_loss = 0.
        for rel, style in zip(rel_style, img_style):
            # style loss is calculated for each style layer and summed
            loss = 0.
            for layer, weight in zip(layers_style, layers_style_weights):
                a = sess.run(vgg_19_net[layer], 
                             feed_dict={img_ph: style})
                x = net[layer]
                a = tf.convert_to_tensor(a)
                loss += style_layer_loss(a, x)

            style_loss += rel*loss

    if diff2_weight:
        def get_diff2_loss(a, x):
            #Calculated mse between marked pixels
            diff2 = diff2_weight * (a-x)**2
            sum_marked_diffs = tf.reduce_sum(diff2)
            return sum_marked_diffs

        with tf.Session() as sess:
            diff2_loss = get_diff2_loss(net['input'], img)


    def _train_op(tower_losses_ph, update_ops, learning_rate_ph, 
                  variable_strategy='GPU', **kwargs):

        tower_losses = tf.unstack(tower_losses_ph, effective_batch_size)

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

            _loss = tf.reduce_mean(tower_losses, name='loss')
            
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_ph)
            optimizer = tf.contrib.estimator.clip_gradients_by_norm(optimizer, 5.0)

        # Create single grouped train op
        train_op = [
            optimizer.apply_gradients(
                gradvars, global_step=global_step)
        ]

        train_op.extend(update_ops)
        train_op = tf.group(*train_op)

        return train_op, _loss

    os.environ['TF_SYNC_ON_FINISH'] = '0'
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

    print("Preparing training operations")

    with open(log_file, 'a') as log:
        log.flush()

        with open(val_log_file, 'a') as val_log:
            val_log.flush()

            with tf.Session() as sess:

                style_loss = weight_style * style_loss

                #Loss function
                if diff2_loss:
                    L_total  = content_loss + style_loss + weight_loss
                else:
                    L_total  = content_loss + style_loss
    
                #Add regularization loss
                if regularization_coeff:

                    regularization_loss = regularization_coeff*tf.add(
                        [tf.nn.l2_loss(v) for v in tower_params])

                    L_total += regularization_loss

                grads = tf.gradients(L_total, tower_params)
    
                init_op = tf.initialize_all_variables()
                sess.run(init_op)

                losses_ph = tf.placeholder(tf.float32, shape=(batch_size,), name='tower_losses')
                learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')

                sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                temp = set(tf.all_variables())

                mini_batch_dict = {}
                _img = sess.run(img[0])
                mini_batch_dict.update({img_ph: _img})
                gradvars_pry = sess.run(grads, feed_dict=mini_batch_dict)
                del mini_batch_dict

                tower_grads_ph = [tf.placeholder(tf.float32, shape=t.shape, name='tower_grads')
                                  for t in gradvars_pry[0]]
                del gradvars_pry
            
                train_op, _loss = _train_op(losses_ph, update_ops, learning_rate_ph, 
                                            _tower_grads=tower_grads_ph)

                sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))

                train_writer = tf.summary.FileWriter( logDir, sess.graph )

                #saver = tf.train.Saver()
                #saver.restore(sess, tf.train.latest_checkpoint("//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-13/model/"))

                print("Starting batched training!")

                counter = 0
                std_operations = [L_total, pred] + grads

                learning_rates = learning_rates_batch + learning_rates_batch_frozen
                train_batch_norms = [True for _ in learning_rates_batch] + [False for _ in learning_rates_batch_frozen]

                for learning_rate, train_batch_norm in zip(learning_rates, train_batch_norms):

                    print("Learning rate: {}, Train batch norm: {}".format(learning_rate, train_batch_norm))

                    counter0 = counter
                    while counter < counter0 + step_learning_rate_every_n_batch:
                        counter += 1

                        losses_list = []
                        preds_list = []
                        ph_dict = {}
                        j = 0
                        for incr in range(batch_size):

                            mini_batch_dict = {}
                            __img = sess.run(img[0])
                            mini_batch_dict.update({img_ph: __img})

                            mini_batch_results = sess.run(std_operations,
                                                          feed_dict=mini_batch_dict)
                            losses_list += [x for x in mini_batch_results[0]]
                            preds_list += [x for x in mini_batch_results[1]]

                            ph_dict.update({ph: val for ph, val in 
                                            zip(tower_grads_ph[j], mini_batch_results[3])})
                            j += 1

                        feed_list = np.asarray(losses_list)
                        feed_list.shape = (1,)
                        ph_dict.update({losses_ph: feed_list,
                                        learning_rate_ph: learning_rate,
                                        train_batch_norm_ph: train_batch_norm,
                                        train_batch_norm_ph: train_batch_norm,
                                        img_ph: __img})

                        del losses_list

                        if save_result_every_n_batches:
                            if counter <= 1 or not counter % save_result_every_n_batches:
                                try:
                                    save_input_loc = model_dir+"input-"+str(counter)+".tif"
                                    save_output_loc = model_dir+"output-"+str(counter)+".tif"
                                    Image.fromarray(__img.reshape(img_size, img_size).astype(np.float32)).save( save_input_loc )
                                    Image.fromarray(preds_list[-1].reshape(img_size, img_size).astype(np.float32)).save( save_output_loc )
                                except:
                                    print("Image save failed")

                        _, loss = sess.run([train_op, _loss], feed_dict=ph_dict)
                        del ph_dict

                        try:
                            log.write("Iter: {}, Loss: {:.8f}".format(counter, float(loss)))
                        except:
                            print("Failed to write to log")

                        if not counter % val_skip_n:
                            mini_batch_dict = {}
                            ___img = sess.run(img_val[0])
                            mini_batch_dict.update({img_ph: ___img})

                            mini_batch_results = sess.run(std_operations,
                                                          feed_dict=mini_batch_dict)
                            val_loss = np.mean(np.asarray([x for x in mini_batch_results[0]]))

                            try:
                                val_log.write("Iter: {}, Loss: {:.8f}".format(counter, float(val_loss)))
                            except:
                                print("Failed to write to val log")

                            print("Iter: {}, Loss: {:.6f}, Val loss: {:.6f}".format(counter, loss, val_loss))
                        else:
                            print("Iter: {}, Loss: {:.6f}".format(counter, loss))

                        #train_writer.add_summary(summary, counter)

                    #Save the model
                    saver.save(sess, save_path=model_dir+"model/", global_step=counter)


if __name__ == "__main__":

    style_imgs = [imread("E:/stills_hq-mini/train/train1.tif", mode='F')]
   
    data_dir = "E:/stills_hq-mini/"
    model_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/guided-fast-style-fusion-1/"

    train_fast_guided_style_fusion(img_style=style_imgs,
                                   model_fn=architecture,
                                   regularization_coeff=1.e-5,
                                   path_VGG19=r'\\flexo.ads.warwick.ac.uk\shared39\EOL2100\2100\Users\Jeffrey-Ede\VGG-19\vgg19.npy',
                                   model_dir=model_dir,
                                   data_dir=data_dir)