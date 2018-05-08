"""
First architecture
Low capacity as only one convolution between striding
Designed to be a baseline
"""
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
    cnn0_last = conv_block(
        input=input_layer, 
        filters=features0)
    cnn0_strided = strided_conv_block(
        input=cnn0_last,
        filters=features0,
        stride=2)

    #Encoding block 1
    cnn1_last = conv_block(
        input=cnn0_strided, 
        filters=features1)
    cnn1_strided = strided_conv_block(
        input=cnn1_last,
        filters=features1,
        stride=2)

    #Encoding block 2
    cnn2_last = conv_block(
        input=cnn1_strided,
        filters=features2)
    cnn2_strided = strided_conv_block(
        input=cnn2_last,
        filters=features2,
        stride=2)

    #Encoding block 3
    #cnn3 = conv_block(
    #    input=cnn2_strided,
    #    filters=features3)
    #cnn3_last = conv_block(
    #    input=cnn3,
    #    filters=features3)
    cnn3_last = conv_block(
        input=cnn2_strided,
        filters=features3)
    cnn3_strided = strided_conv_block(
        input=cnn3_last,
        filters=features3,
        stride=2)

    #Encoding block 4
    #cnn4 = conv_block(
    #    input=cnn3_strided,
    #    filters=features4)
    #cnn4_last = conv_block(
    #    input=cnn4,
    #    filters=features4)
    cnn4_last = conv_block(
        input=cnn3_strided,
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
    #deconv4 = conv_block(deconv4, features4)
    
    #Decoding block 2
    deconv4to3 = deconv_block(deconv4, features4)
    concat3 = tf.concat(
        values=[deconv4to3, cnn3_last],
        axis=concat_axis)
    deconv3 = conv_block(concat3, features3)
    #deconv3 = conv_block(deconv3, features3)

    #Decoding block 3
    deconv3to2 = deconv_block(deconv3, features3)
    concat2 = tf.concat(
        values=[deconv3to2, cnn2_last],
        axis=concat_axis)
    deconv2 = conv_block(concat2, features2)
    
    #Decoding block 4
    deconv2to1 = deconv_block(deconv2, features2)
    concat1 = tf.concat(
        values=[deconv2to1, cnn1_last],
        axis=concat_axis)
    deconv1 = conv_block(concat1, features1)

    #Decoding block 5
    deconv1to0 = deconv_block(deconv1, features1)
    concat0 = tf.concat(
        values=[deconv1to0, cnn0_last],
        axis=concat_axis)
    deconv1 = conv_block(concat0, features0)

    #Create final image with 1x1 convolutions
    deconv_final = tf.layers.conv2d_transpose(
        inputs=deconv1,
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
        loss = cropsize*cropsize*tf.reduce_mean(tf.squared_difference(output, ground_truth))
        tf.summary.histogram("loss", loss)
    else:
        loss = -1

    return loss, output
