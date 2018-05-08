##Modified aligned xception
def architecture(lq, img=None, mode=None):
    """Atrous convolutional encoder-decoder noise-removing network"""

    #phase = mode == tf.estimator.ModeKeys.TRAIN #phase is true during training
    concat_axis = 3

    ##Reusable blocks

    def conv_block(input, filters, phase=phase):
        """
        Convolution -> batch normalisation -> leaky relu
        phase defaults to true, meaning that the network is being trained
        """

        conv_block = tf.layers.conv2d(inputs=input,
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
        conv1x1 = tf.layers.conv2d(inputs=input,
            filters=aspp_filters,
            kernel_size=1,
            padding="same",
            activation=tf.nn.relu,
            name="1x1")
        #conv1x1 = tf.contrib.layers.batch_norm(
        #    conv1x1,
        #    center=True, scale=True,
        #    is_training=phase)

        conv3x3_rateSmall = tf.layers.conv2d(inputs=input,
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

        conv3x3_rateMedium = tf.layers.conv2d(inputs=input,
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

        conv3x3_rateLarge = tf.layers.conv2d(inputs=input,
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
        pooling = tf.nn.pool(input=input,
            window_shape=(2,2),
            pooling_type="AVG",
            padding="SAME",
            strides=(2, 2))
        #Use 1x1 convolutions to project into a feature space the same size as
        #the atrous convolutions'
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
        concatenation = tf.concat(values=[conv1x1, conv3x3_rateSmall, conv3x3_rateMedium, conv3x3_rateLarge, pooling],
            axis=concat_axis)

        #Reduce the number of channels
        reduced = tf.layers.conv2d(
            inputs=concatenation,
            filters=aspp_filters,
            kernel_size=1,
            padding="SAME")

        return reduced


    def strided_conv_block(input, filters, stride, rate=1, phase=phase):
        
        return slim.separable_convolution2d(inputs=input,
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
        
        #Residual
        residual = tf.layers.conv2d(
            inputs=input,
            filters=filters,
            kernel_size=1,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)

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
            padding="SAME",
            activation=tf.nn.relu)

        return deconv_block + residual

    def xception_entry_flow(input):

        #Entry flow 0
        entry_flow = tf.layers.conv2d(
            inputs=input,
            filters=filters00,
            kernel_size=3,
            strides = 2,
            padding="SAME",
            activation=tf.nn.relu)
        entry_flow = tf.layers.conv2d(
            inputs=entry_flow,
            filters=filters01,
            kernel_size=3,
            padding="SAME",
            activation=tf.nn.relu)

        #Residual 1
        residual1 = tf.layers.conv2d(
            inputs=entry_flow,
            filters=filters1,
            kernel_size=1,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
       
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
            padding="SAME",
            activation=tf.nn.relu)
       
        #Main flow 2
        main_flow2 = strided_conv_block(
            input=main_flow1,
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
            padding="SAME",
            activation=tf.nn.relu)
       
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

        #Residual 4
        residual4 = tf.layers.conv2d(
            inputs=residual_connect3,
            filters=filters4,
            kernel_size=1,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
       
        #Main flow 4
        main_flow4 = strided_conv_block(
            input=main_flow3,
            filters=filters4,
            stride=1)
        main_flow4 = strided_conv_block(
            input=main_flow4,
            filters=filters4,
            stride=1)
        main_flow4_strided = strided_conv_block(
            input=main_flow4,
            filters=filters4,
            stride=2)

        residual_connect4 = main_flow4_strided + residual4

        return residual_connect4, entry_flow, main_flow1, main_flow2, main_flow3, main_flow4

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

        return main_flow + residual

    def xception_exit_flow(input):

        #Residual
        residual = tf.layers.conv2d(
            inputs=input,
            filters=filters5,
            kernel_size=1,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)

        #Main flow
        main_flow = main_flow = strided_conv_block(
            input=minput,
            filters=filters5,
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
            filters=filters5,
            stride=1)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters5,
            stride=1,
            rate=2)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters5,
            stride=1)

        return main_flow

    '''Model building'''
    input_layer = tf.reshape(lq, [-1, cropsize, cropsize, channels])

    #Build Xception
    main_flow, side_flow0, side_flow1, side_flow2, side_flow3, side_flow4 = xception_entry_flow(input_layer)

    for _ in range(numMiddleXception):
        main_flow = xception_middle_block(main_flow)

    main_flow = xception_exit_flow(main_flow)

    ##Atrous spatial pyramid pooling
    aspp = aspp_block(main_flow)

    decoder_flow = tf.image.resize_images(aspp, [256, 256])

    #Concatonation and transpositional convolution 1
    decoder_flow = tf.concat(values=[decoder_flow, side_flow3],
        axis=concat_axis)
    decoder_flow = deconv_block(decoder_flow)

    #Concatonation and transpositional convolution 2
    decoder_flow = tf.concat(values=[decoder_flow, side_flow2],
        axis=concat_axis)
    decoder_flow = deconv_block(decoder_flow)

    #Concatonation and transpositional convolution 3
    decoder_flow = tf.concat(values=[decoder_flow, side_flow1],
        axis=concat_axis)
    decoder_flow = deconv_block(decoder_flow)

    #Prepare the output
    decoder_flow = tf.concat(values=[decoder_flow, side_flow0],
        axis=concat_axis)
    decoder_flow = tf.layers.conv2d(
            inputs=residual_connect3,
            filters=filters01,
            kernel_size=3,
            strides=1,
            padding="SAME",
            activation=tf.nn.relu)

    #Create final image with 1x1 convolutions
    output = tf.layers.conv2d_transpose(
        inputs=decoder_flow,
        filters=1,
        kernel_size=3,
        padding="SAME",
        activation=tf.nn.relu)

    #Image values will be between 0 and 1
    output = tf.clip_by_value(output,
        clip_value_min=0,
        clip_value_max=1)

    if phase: #Calculate loss during training
        loss = cropsize * cropsize * tf.reduce_mean(tf.squared_difference(input_layer, output))
        tf.summary.histogram("loss", loss)
    else:
        loss = -1

    return loss, output                     