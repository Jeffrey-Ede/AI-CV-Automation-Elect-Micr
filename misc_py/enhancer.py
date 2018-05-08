from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import cv2
from scipy.misc import imread

slim = tf.contrib.slim #For depthwise separable strided atrous convolutions

tf.logging.set_verbosity(tf.logging.DEBUG)

features1 = 32 #Number of features to use for initial convolution
features2 = 2*features1 #Number of features after 2nd convolution
features3 = 3*features2 #Number of features after 3rd convolution
features4 = 4*features3 #Number of features after 4th convolution
aspp_filters = features4 #Number of features for atrous convolutional spatial pyramid pooling

aspp_rateSmall = 6
aspp_rateMedium = 12
aspp_rateLarge = 18

trainDir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/2100plus_dm3/"
valDir = ""
testDir = ""

model_dir = "E:/stills/"

shuffle_buffer_size = 5000
parallel_readers = 4

batch_size = 12 #Batch size to use during training
num_epochs = 1

log_every = 1 #Log every _ examples

#Dimensions of images in the dataset
height = width = 2048
channels = 1 #Greyscale input image

## Initial idea: aspp, batch norm + some PRELU, residual connection and lower feature numbers
def cnn_model_fn_enhancer(features, labels, mode):
    """Atrous convolutional encoder-decoder noise-removing network"""
    
    phase = mode == tf.estimator.ModeKeys.TRAIN #Check if the nn is training for batch normalisation updates

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
            padding="same")

        conv_block = tf.contrib.layers.batch_norm(
            conv_block, 
            center=True, scale=True, 
            is_training=phase,
            scope='bn')

        conv_block = tf.nn.leaky_relu(
            features=conv_block,
            alpha=0.2)

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
        padding="same")
        conv1x1 = tf.contrib.layers.batch_norm(
            conv1x1, 
            center=True, scale=True, 
            is_training=phase,
            scope='bn')

        conv3x3_rateSmall = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateSmall)
        conv3x3_rateSmall = tf.contrib.layers.batch_norm(
            conv3x3_rateSmall, 
            center=True, scale=True, 
            is_training=phase,
            scope='bn')

        conv3x3_rateMedium = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateMedium)
        conv3x3_rateMedium = tf.contrib.layers.batch_norm(
            conv3x3_rateMedium, 
            center=True, scale=True, 
            is_training=phase,
            scope='bn')

        conv3x3_rateLarge = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateLarge)
        conv3x3_rateLarge = tf.contrib.layers.batch_norm(
            conv3x3_rateLarge, 
            center=True, scale=True, 
            is_training=phase,
            scope='bn')

        #Image-level features
        pooling = tf.nn.pool(
            input=input,
            window_shape=(2,2),
            pooling_type="AVG",
            padding="same")
        #Use 1x1 convolutions to project into a feature space the same size as the atrous convolutions'
        pooling = tf.layers.conv2d(
            inputs=pooling,
            filters=aspp_filters,
            kernel_size=1,
            padding="same")
        pooling = tf.contrib.layers.batch_norm(
            pooling,
            center=True, scale=True,
            is_training=phase,
            scope='bn')

        #Concatenate the atrous and image-level pooling features
        concatenation = tf.concat(
        values=[conv1x1, conv3x3_rateSmall, conv3x3_rateMedium, conv3x3_rateLarge, pooling],
        axis=axis)

        #Reduce the number of channels
        reduced = tf.layers.conv2d( #Not sure if this is the correct way to reshape...
            inputs=values,
            filters=aspp_filters,
            kernel_size=1,
            padding="same")

        return reduced

    def split_separable_conv2d(
        inputs,
        filters,
        rate=1,
        stride=1,
        weight_decay=0.00004,
        depthwise_weights_initializer_stddev=0.33,
        pointwise_weights_initializer_stddev=0.06,
        scope=None):

        """
        Splits a separable conv2d into depthwise and pointwise conv2d.
        This operation differs from `tf.layers.separable_conv2d` as this operation
        applies activation function between depthwise and pointwise conv2d.
        Args:
        inputs: Input tensor with shape [batch, height, width, channels].
        filters: Number of filters in the 1x1 pointwise convolution.
        rate: Atrous convolution rate for the depthwise convolution.
        weight_decay: The weight decay to use for regularizing the model.
        depthwise_weights_initializer_stddev: The standard deviation of the
            truncated normal weight initializer for depthwise convolution.
        pointwise_weights_initializer_stddev: The standard deviation of the
            truncated normal weight initializer for pointwise convolution.
        scope: Optional scope for the operation.
        Returns:
        Computed features after split separable conv2d.
        """

        outputs = slim.separable_conv2d(
            inputs,
            None,
            3,
            strides=stride,
            depth_multiplier=1,
            rate=rate,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=depthwise_weights_initializer_stddev),
            weights_regularizer=None,
            scope=scope + '_depthwise')

        outputs = tf.contrib.layers.batch_norm(
            outputs, 
            center=True, scale=True, 
            is_training=phase,
            scope='bn')

        outputs = tf.nn.leaky_relu(
            features=outputs,
            alpha=0.2)

        outputs = slim.conv2d(
            outputs,
            filters,
            kernel_size=1,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=pointwise_weights_initializer_stddev),
            weights_regularizer=slim.l2_regularizer(weight_decay),
            scope=scope + '_pointwise')

        return outputs

    def deconv_block(input, filters, phase=phase):
        '''Transpositionally convolute a feature space to upsample it'''
        
        deconv_block = tf.layers.conv2d_transpose(
            inputs=input,
            filters=filters,
            kernel_size=3,
            strides=2,
            padding="same")

        deconv_block = tf.contrib.layers.batch_norm(
            deconv_block, 
            center=True, scale=True, 
            is_training=phase,
            scope='bn')

        deconv_block = tf.nn.leaky_relu(
            features=deconv_block,
            alpha=0.2)

        return deconv_block

    '''Model building'''
    input_layer = tf.reshape(features["x"], [-1, rows, cols, 1])

    #Encoding block 1
    cnn1_last = conv_block(
        input=input_layer, 
        filters=features1)
    cnn1_strided = split_separable_conv2d(
        inputs=cnn1_last,
        filters=features1,
        rate=2,
        stride=2)

    #Encoding block 2
    cnn2_last = conv_block(
        input=cnn1_strided,
        aspp_filters=features2)
    cnn2_strided = split_separable_conv2d(
        inputs=cnn2_last,
        filters=features2,
        rate=2,
        stride=2)

    #Encoding block 3
    cnn3 = conv_block(
        input=cnn2_strided,
        aspp_filters=features3)
    cnn3_last = conv_block(
        input=cnn3_last,
        aspp_filters=features3)
    cnn3_strided = split_separable_conv2d(
        inputs=cnn3,
        filters=features3,
        rate=2,
        stride=2)

    #Encoding block 4
    cnn4 = conv_block(
        input=cnn3_strided,
        aspp_filters=features4)
    cnn4_last = conv_block(
        input=cnn4,
        aspp_filters=features4)
    cnn4_strided = split_separable_conv2d(
        inputs=cnn4,
        filters=features4,
        rate=2,
        stride=2)

    #Atrous spatial pyramid pooling
    aspp = aspp_block(cnn4_strided)

    #Upsample the semantics by a factor of 4
    #upsampled_aspp = tf.image.resize_bilinear(
    #    images=aspp,
    #    tf.shape(aspp)[1:3],
    #    align_corners=True)

    #Decoding block 1 (deepest)
    deconv4 = conv_block(aspp)
    deconv4 = conv_block(aspp)
    deconv4 = conv_block(aspp)
    
    #Decoding block 2
    deconv4to3 = deconv_block(deconv4, filters4)
    concat3 = tf.concat(
        values=[deconv4to3, cnn3_last],
        axis=axis)
    deconv3 = conv_block(concat3, features3)
    deconv3 = conv_block(deconv3, features3)

    #Decoding block 3
    deconv3to2 = deconv_block(deconv3, filters3)
    concat2 = tf.concat(
        values=[deconv3to2, cnn2_last],
        axis=axis)
    deconv2 = conv_block(concat2, features2)
    
    #Decoding block 4
    deconv2to1 = deconv_block(deconv2, filters2)
    concat1 = tf.concat(
        values=[deconv2to1, cnn1_last],
        axis=axis)
    deconv1 = conv_block(concat1, features1)

    #Create final image with 1x1 convolutions
    deconv_final = tf.layers.conv2d_transpose(
        inputs=deconv1,
        filters=1,
        kernel_size=3,
        padding="same")

    #Residually connect the input to the output
    output = input_layer + deconv_final

    '''Evaluation'''
    predictions = { "Output": output }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(
        labels=labels,
        predictions=labels)

    '''Training'''
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    '''Evaluation'''
    eval_metric_ops = {
        "Loss": loss }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def load_image(addr, resizeSize=None, imgType=np.float32):
    """Read an image and make sure it is of the correct type. Optionally resize it"""
    
    img = imread(addr, mode='F')
    
    if resizeSize:
        img = cv2.resize(img, resizeSize, interpolation=cv2.INTER_CUBIC)
    
    img = img.astype(imgType)

    return img

def gen_lq(img, mean):
    '''Generate low quality image'''

    #Ensure that the seed is random
    np.random.seed(int(np.random.rand()*(2**32-1)))

    #Adjust the image scale so that the image has the correct average counts
    lq = np.random.poisson(mean * (img / np.mean(img)))
    
    #Rescale between 0 and 1
    min = np.min(lq)
    lq = (lq-min) / (np.max(lq)-min)

    return lq.astype(np.float32)

def preprocess(lq, img):
    
    

    return lq, img

def parser(record, mean):
    """Parse files and generate lower quality images from them"""

    img = load_image(record)
    lq = gen_lq(img, mean)

    return preprocess(lq, img)

def input_fn(dir, mean):
    """Create a dataset from a list of filenames"""

    dataset = tf.data.Dataset.list_files(dir+"*.tif")
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(lambda file: tuple(tf.py_func(parser, [file, mean], [tf.float32, tf.float32])))
    #dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat(num_epochs)
    
    iter = dataset.make_one_shot_iterator()
    #with tf.Session() as sess:
    #    next = iter.get_next()
    #    print(sess.run(next)) #Output
    #    img = sess.run(next)
        
    lq, img = iter.get_next()

    return lq, img

def main(unused_argv=None):

    mean = 64 #Average value of pixels in low quality generations

    lq_train, img_train = input_fn(trainDir, mean)

    # Create the Estimator
    estimator = tf.estimator.Estimator(
        model_fn=cnn_model_fn_enhancer, 
        model_dir=model_dir)

    ##Set up logging for predictions
    #tensors_to_log = {}
    #logging_hook = tf.train.LoggingTensorHook(
    #    tensors=tensors_to_log, every_n_iter=log_every)

    # Train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': lq_train},
        y=img_train,
        batch_size=batch_size,
        num_epochs=num_epochs,
        shuffle=True)
    estimator.train(
        input_fn=train_input_fn,
        steps=10,
        hooks=None)#[logging_hook])

    ## Evaluate the model and print results
    #eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    #    x=lq_train,
    #    y=img_train,
    #    num_epochs=1,
    #    shuffle=False)
    #eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    #print(eval_results)

    ##Train the nn
    
    #with tf.train.MonitoredTrainingSession() as sess:
    #    while not sess.should_stop():
    #        img = sess.run(lq_train)

    #        print(np.max(img))

    #        cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
    #        cv2.imshow("dfsd", img.reshape((2048,2048)))
    #        cv2.waitKey(0)

    #        #sess.run(training_op)

    return 

if __name__ == "__main__":
    tf.app.run()