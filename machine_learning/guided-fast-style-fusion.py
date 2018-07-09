import os
import numpy as np
import scipy.misc
import scipy.io
import math
import tensorflow as tf
from sys import stderr
from functools import reduce
import time  

model_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/guided-fast-style-fusion-1/"

num_gpus = 1
img_size = 512

batch_size = 16

iters_with_batch = 40000
iters with_batch_frozen = 40000

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

def architecture(inputs, phase=False, params=None):
    """Fast style fustion neural network architecture"""

    batch_decay_gen = 0.999

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

    num_global_enhancer_blocks = 8
    num_local_enhancer_blocks = 4

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
                is_training=phase,
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
                                 stride=2,
                                 kernel_size = 9,
                                 pad_size=(4,4))
        enc = strided_conv_block(enc, gen_features1, 1, 1, pad_size=(1,1))

        enc += network_in_network(enc)

        for _ in range(num_local_enhancer_blocks):
            enc = xception_middle_block(enc, gen_features2, pad_size=(1,1))

        enc = deconv_block(enc, gen_features3, new_size=(512, 512), pad_size=(1,1))
        enc = strided_conv_block(enc, gen_features3, 1, 1, pad_size=(1,1))

        enc = conv_block_not_sep(enc, 1, pad_size=(4,4), kernel_size=9, batch_plus_activ=False)

        enc = 150.*tf.tanh(enc)+255./2.

        return enc

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


def preprocess(img):

    img[np.isnan(img)] = 0.5
    img[np.isinf(img)] = 0.5

    img = cv2.resize(img, (img_size, img_size))

    img = scale0to1(img)

    return img

def record_parser(record):
    """Parse files and generate lower quality images from them"""

    img = flip_rotate(preprocess(load_image(record)))

    return img

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


def train_fast_guided_style_fusion(img_content,
                   img_val_content,
                   img_style,
                   model_fn,
                   img_mask=None, 
                   weight_mask=None, 
                   weight_style=2.e2,
                   rel_style=None,
                   rel_mask=None,
                   input_noise=0.1,
                   save_checkpoint_imgs_every_n=None,
                   path_output='output',
                   layers_style_weights = [0.2,0.2,0.2,0.2,0.2],
                   dont_use_n_pixels=5,
                   regularization_coeff=None):
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
    path_output: Location to save checkpoint images if that option is enabled
    layers_style_weights: 5 element array with proportional contributions of VGG19 layers
    to style loss
    dont_use_n_pixels: Take random crops that are the size of the image minus this. This 
    improves training by reducing artifacts
    """

    #Convert inputs to lists if they are single images
    if not isinstance(img_content, (list,)):
        img_content = [img_content]
    if not isinstance(img_style, (list,)):
        img_style = [img_style]
    if weight_mask and not isinstance(weight_mask, (list,)):
        weight_mask = [weight_mask]

    #Normalise relative contents so that they sum to 1.
    if rel_mask:
        sum = np.sum(np.asarray(rel_mask))
        rel_mask = [x/sum for x in rel_mask]

    ## Layers
    layer_content = 'conv4_2' 
    layers_style = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    ## VGG19 model
    path_VGG19 = 'D:/imagenet-vgg-verydeep-19.mat'
    # VGG19 mean for standardisation (RGB)
    VGG19_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

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

    # function to convert 2D greyscale to 3D RGB
    def to_rgb(im):
        w, h = im.shape
        ret = np.empty((w, h, 3), dtype=np.uint8)
        ret[:, :, 0] = im
        ret[:, :, 1] = im
        ret[:, :, 2] = im
        return ret

    ### Preprocessing
    # create output directory
    if not os.path.exists(path_output):
        os.mkdir(path_output)

    # convert if greyscale
    if len(img_content.shape)==2:
        img_content = to_rgb(img_content)

    if len(img_style.shape)==2:
        img_style = to_rgb(img_style)

    # resize style image to match content
    img_style = scipy.misc.imresize(img_style, img_content.shape)

    # apply noise to create initial "canvas" using first content image
    noise = np.random.uniform(
            img_content[0].mean()-img_content[0].std(), img_content[0].mean()+img_content[0].std(),
            (img_content[0].shape)).astype('float32')
    img_initial = noise * input_noise + img_content[0] * (1-input_noise)

    # preprocess each
    img_content = [imgpreprocess(content) for content in img_content]
    img_style = [imgpreprocess(style) for style in img_style]
    img_initial = imgpreprocess(img_initial)

    if weight_mask:
        img_content_tensors = [tf.convert_to_tensor(content) for content in img_content]
        img_mask_tensors = [tf.convert_to_tensor(mask) for mask in img_mask]

    #### BUILD VGG19 MODEL
    ## with thanks to http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style

    VGG19 = scipy.io.loadmat(path_VGG19)
    VGG19_layers = VGG19['layers'][0]

    # help functions
    def _conv2d_relu(prev_layer, n_layer, layer_name):
        # get weights for this layer:
        weights = VGG19_layers[n_layer][0][0][2][0][0]
        W = tf.constant(weights)
        bias = VGG19_layers[n_layer][0][0][2][0][1]
        b = tf.constant(np.reshape(bias, (bias.size)))
        # create a conv2d layer
        conv2d = tf.nn.conv2d(prev_layer, filter=W, strides=[1, 1, 1, 1], padding='SAME') + b    
        # add a ReLU function and return
        return tf.nn.relu(conv2d)

    def _avgpool(prev_layer):
        return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def _fc_layer(prev_layer, n_layer, name, add_relu=True):

            shape = prev_layer.get_shape().as_list()
            dim = 1
            for d in shape[1:]:
                dim *= d
            x = tf.reshape(prev_layer, [-1, dim])

            weights = VGG19_layers[n_layer][0][0][2][0][0]
            W = tf.constant(weights)
            bias = VGG19_layers[n_layer][0][0][2][0][1]
            b = tf.constant(np.reshape(bias, (bias.size)))

            # Fully connected layer. Note that the '+' operation automatically
            # broadcasts the biases.
            fc = tf.nn.bias_add(tf.matmul(x, W), b)

            if add_relu:
                fc = tf.nn.relu(fc)

            return fc

    # Setup network
    with tf.Session() as sess:
        
        content = sess.run(img_content)
        input_img_ph = tf.placeholder(tf.float32, content.shape, "img")
        input_ph = tf.placeholder(tf.float32, (content.shape), "img")
        del content
        pred = model_fn(input_img_ph, phase=True)

        input = tf.concat([pred]*3, axis=3)

        net = {}
        net['input']   = input
        net['conv1_1']  = _conv2d_relu(net['input'], 0, 'conv1_1')
        net['conv1_2']  = _conv2d_relu(net['conv1_1'], 2, 'conv1_2')
        net['avgpool1'] = _avgpool(net['conv1_2'])
        net['conv2_1']  = _conv2d_relu(net['avgpool1'], 5, 'conv2_1')
        net['conv2_2']  = _conv2d_relu(net['conv2_1'], 7, 'conv2_2')
        net['avgpool2'] = _avgpool(net['conv2_2'])
        net['conv3_1']  = _conv2d_relu(net['avgpool2'], 10, 'conv3_1')
        net['conv3_2']  = _conv2d_relu(net['conv3_1'], 12, 'conv3_2')
        net['conv3_3']  = _conv2d_relu(net['conv3_2'], 14, 'conv3_3')
        net['conv3_4']  = _conv2d_relu(net['conv3_3'], 16, 'conv3_4')
        net['avgpool3'] = _avgpool(net['conv3_4'])
        net['conv4_1']  = _conv2d_relu(net['avgpool3'], 19, 'conv4_1')
        net['conv4_2']  = _conv2d_relu(net['conv4_1'], 21, 'conv4_2')     
        net['conv4_3']  = _conv2d_relu(net['conv4_2'], 23, 'conv4_3')
        net['conv4_4']  = _conv2d_relu(net['conv4_3'], 25, 'conv4_4')
        net['avgpool4'] = _avgpool(net['conv4_4'])
        net['conv5_1']  = _conv2d_relu(net['avgpool4'], 28, 'conv5_1')
        net['conv5_2']  = _conv2d_relu(net['conv5_1'], 30, 'conv5_2')
        net['conv5_3']  = _conv2d_relu(net['conv5_2'], 32, 'conv5_3')
        net['conv5_4']  = _conv2d_relu(net['conv5_3'], 34, 'conv5_4')
        net['avgpool5'] = _avgpool(net['conv5_4'])
        net['fc_1'] = _fc_layer(net['avgpool5'], 39, 'fc_1')
        net['fc_2'] = _fc_layer(net['fc1'], 41, 'fc_2')
        net['fc_3'] = _fc_layer(net['fc2'], 41, 'fc_3', add_relu=False)

    ### CONTENT LOSS: FUNCTION TO CALCULATE AND INSTANTIATION
    # with thanks to https://github.com/cysmith/neural-style-tf

    # Recode to be simpler: http://www.chioka.in/tensorflow-implementation-neural-algorithm-of-artistic-style
    def content_layer_loss(p, x):
        _, h, w, d = [i.value for i in p.get_shape()]    # d: number of filters; h,w : height, width
        M = h * w 
        N = d 
        K = 1. / (2. * N**0.5 * M**0.5)
        loss = K * tf.reduce_sum(tf.pow((x - p), 2))
        return loss

    with tf.Session() as sess:

        sess.run(net['input'].assign(pred))
        p = sess.run(net[layer_content])  #Get activation output for content layer
        x = net[layer_content]
        p = tf.convert_to_tensor(p)

        content_loss += content_layer_loss(p, x)

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
        thetas = [tf.cond(angle <= np.pi, lambda x: tf.abs(x), lambda x: tf.abs(x-2.*np.pi)) 
                  for angle in thetas]
        
        #Softmax the similarities
        similarities = [tf.cond(angle <= np.pi/2, lambda x: np.pi/2-x, lambda x: x-np.pi/2) 
                        for angle in thetas]
        similarities = [tf.exp(sim) for sim in similarities]
        sum = tf.reduce_sum(tf.concat(similarities, axis=0))
        similarities = [sim/sum for sim in similarities]

        return similarities

    #Get logits for the style images
    with tf.Session() as sess:

        logits_list = []
        for style in img_style:
            sess.run(net['input'].assign(style))
            logits = sess.run(net['fc_3'])
            logits = tf.constant(logits, dtype=tf.float32, shape=(-1,))
            logits_list.append(logits)

    #Calculate the style loss
    with tf.Session() as sess:

        #Get style guiding weights
        logits = net['fc_3']
        style_tensors = [net[layer] for layer in layers_style]
        rel_style = style_guide_weights(logits, logits_list)

        style_loss = 0.
        for rel, style in zip(rel_style, img_style):
            sess.run(net['input'].assign(style))
            # style loss is calculated for each style layer and summed
            loss = 0.
            for layer, weight in zip(layers_style, layers_style_weights):
                a = sess.run(net[layer])
                x = net[layer]
                a = tf.convert_to_tensor(a)
                loss += style_layer_loss(a, x)

            style_loss += rel*loss

    if weight_mask:
        def get_mask_loss(a, x):
            #Calculated mse between marked pixels
            diff2 = img_mask_tensor * (a-x)**2
            sum_marked_diffs = tf.reduce_sum(diff2)
            return sum_marked_diffs

        with tf.Session() as sess:
            losses = [get_mask_loss(net['input'], x) for x in img_content_tensors]
            mask_loss = tf.add_n([rel*loss for rel, loss in zip(rel_mask, losses)])

    ### Define loss function and minimise
    with tf.Session() as sess:

        style_loss = weight_style * style_loss
        weight_loss = weight_mask * mask_loss

        #Loss function
        if weight_mask:
            L_total  = content_loss + style_loss + weight_loss
        else:
            L_total  = content_loss + style_loss
    
        trainable_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 
                                                    scope="architecture")
        #Add regularization loss
        if regularization_coeff:

            regularization_loss = regularization_coeff*tf.add(
                [tf.nn.l2_loss(v) for v in trainable_variables])

            L_total += regularization_loss

        grads = tf.gradients(L_total, trainable_variables)

        #Instantiate optimiser
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_ph)
        optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
        train_op = optimizer.minimize(L_total)
    
        init_op = tf.initialize_all_variables()
        sess.run(init_op)

        train_writer = tf.summary.FileWriter( logDir, sess.graph )

        #saver = tf.train.Saver()
        #saver.restore(sess, tf.train.latest_checkpoint("//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-13/model/"))

        counter = 0
        std_operations = [L_total, pred] + grads
        for learning_rate in learning_rates_batch:

            counter0 = counter
            while counter < counter0 + iters_with_batch:

                counter += 1

                losses_list = []
                preds_list = []
                ph_dict = {}
                j = 0
                for incr in range(batch_size):
                    mini_batch_dict = {}
                    for i in range(batch_size):
                        __img = sess.run(img[i])

                        #__img, __img_truth = sess.run([img[i], img_truth[i]])
                        mini_batch_dict.update({img_ph[i]: __img})

                    mini_batch_results = sess.run(std_operations,
                                                  feed_dict=mini_batch_dict)
                    losses_list += [x for x in mini_batch_results[0]]
                    preds_list += [x for x in mini_batch_results[1]]

                    for i in range(2, 2+batch_size):
                        ph_dict.update({ph: val for ph, val in 
                                        zip(tower_grads_ph[j], 
                                            mini_batch_results[i])})
                        j += 1

                feed_list = np.asarray(losses_list)
                feed_list.shape = (1,)
                ph_dict.update({tower_losses_ph: feed_list,
                                learning_rate_ph: learning_rate,
                                img_ph[0]: __img})

                del tower_losses_list

                if counter <= 1 or not counter % save_result_every_n_batches:
                    try:
                        save_input_loc = model_dir+"input-"+str(counter)+".tif"
                        save_output_loc = model_dir+"output-"+str(counter)+".tif"
                        Image.fromarray(__img.reshape(cropsize, cropsize).astype(np.float32)).save( save_input_loc )
                        Image.fromarray(mini_batch_results[1][0].reshape(cropsize, cropsize).astype(np.float32)).save( save_output_loc )
                    except:
                        print("Image save failed")

                _, actual_loss, loss_value = sess.run([train_op, get_loss, get_loss_mse],
                                                        feed_dict=ph_dict)
                del ph_dict

                try:
                    log.write("Iter: {}, Loss: {:.8f}".format(counter, float(loss_value)))
                except:
                    print("Failed to write to log")

                if not counter % val_skip_n:
                    mini_batch_dict = {}
                    for i in range(batch_size):
                        ___img, ___img_truth = sess.run([img_val[i], img_val_truth[i]])
                        mini_batch_dict.update({img_ph[i]: ___img})
                        mini_batch_dict.update({img_truth_ph[i]: ___img_truth})

                    mini_batch_results = sess.run([_tower_losses, _tower_preds, _tower_mses]+
                                                    tower_grads,
                                                    feed_dict=mini_batch_dict)
                    val_loss = np.mean(np.asarray([x for x in mini_batch_results[2]]))

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

    style_imgs = []
   
    data_dir = "E:/ARM_scans-crops/"
    img = input_fn(data_dir, 'train', batch_size=1, num_shards=num_gpus)
    img_val = input_fn(data_dir, 'val', batch_size=1, num_shards=num_gpus)

    train_fast_guided_style_fusion(img_content=img,
                                   img_val_content=img_val,
                                   img_style=style_images,
                                   model_fn=architecture,
                                   regularization_coeff=2.e-3)





            #if logging:
            #    #Print losses
            #    stderr.write('Iteration %d/%d\n' % (i*n_iterations_checkpoint, n_checkpoints*n_iterations_checkpoint))
            #    stderr.write('  content loss: %g\n' % sess.run(content_loss))
            #    stderr.write('    style loss: %g\n' % sess.run(weight_style * style_loss))
            #    if weight_mask:
            #        stderr.write('    style loss: %g\n' % sess.run(weight_mask * mask_loss))
            #    stderr.write('    total loss: %g\n' % sess.run(L_total))

            #    #Log image
            #    img_output = sess.run(net['input'])
            #    img_output = imgunprocess(img_output)

            #    timestr = time.strftime("%Y%m%d_%H%M%S")
            #    output_file = path_output+'/'+timestr+'_'+'%s.jpg' % (i*n_iterations_checkpoint)

            #    imsave(output_file, img_output)

            #elif i == n_checkpoints:
            #    img_output = sess.run(net['input'])
            #    img_output = imgunprocess(img_output)