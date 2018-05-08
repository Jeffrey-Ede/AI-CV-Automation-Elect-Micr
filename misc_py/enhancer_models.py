##USEFUL
#To get tensorboard:
#1) python "C:/Program Files (x86)/Microsoft Visual Studio/Shared/Python36_64/Lib/site-packages/tensorboard/main.py" --logdir=c/dump/train
#2) http://DESKTOP-SA1EVJV:6006 

def main(unused_argv=None):

    temp = set(tf.all_variables())

    log = open(log_file, 'a')

    #with tf.device("/gpu:1"):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
    with tf.control_dependencies(update_ops):

        img = input_fn(trainDir)

        loss, prediction = architecture(img, tf.estimator.ModeKeys.TRAIN)
        train_op = tf.train.AdamOptimizer().minimize(loss, colocate_gradients_with_ops=True)

        config = tf.ConfigProto(allow_soft_placement=True)
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
                    _, prediction1 = sess.run([train_op, img])
                    print(prediction1)
                    print("Iter: {}, Loss: {:.6f}".format(counter, loss_value))
                    log.write("Iter: {}, Loss: {:.6f}".format(counter, loss_value))
                    #train_writer.add_summary(summary, counter)

                #Save the model
                #saver.save(sess, save_path=model_dir+"model", global_step=counter)
                tf.saved_model.simple_save(
                    session=sess,
                    export_dir=model_dir+"model-"+str(counter)+"/",
                    inputs={"img": img[0]},
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

##Modified aligned xception
def architecture(img, mode=None):
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
            input=residual_connect3,
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

        return residual_connect4

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
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)

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
            rate=2)
        main_flow = strided_conv_block(
            input=main_flow,
            filters=filters7,
            stride=1)

        return main_flow


    def autoencoding_decoder(input):

        #8x8
        #decoding = tf.reshape(input, [1, decode_size1, decode_size1, decode_channels0])
        decoding = conv_block(input, decode_channels1)
        decoding = conv_block(decoding, decode_channels1)
        decoding = conv_block(decoding, decode_channels1)

        #16x16
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decode_channels1,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, decode_channels2)
        decoding = conv_block(decoding, decode_channels2)
        decoding = conv_block(decoding, decode_channels2)

        #32x32
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decode_channels2,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, decode_channels3)
        decoding = conv_block(decoding, decode_channels3)

        #64x64
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decode_channels3,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, decode_channels4)
        decoding = conv_block(decoding, decode_channels4)

        #128x128
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decode_channels4,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, decode_channels5)
        decoding = conv_block(decoding, decode_channels5)

        #256x256
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decode_channels5,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, decode_channels6)

        #512x512
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decode_channels6,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, decode_channels7)

        #1024x1024
        decoding = tf.layers.conv2d_transpose(
            inputs=decoding,
            filters=decode_channels7,
            kernel_size=3,
            strides=2,
            padding="SAME",
            activation=tf.nn.relu)
        decoding = conv_block(decoding, decode_channels8)
        decoding = conv_block(decoding, 1)

        return decoding

    '''Model building'''
    with tf.device("/gpu:0"):
        input_layer = tf.reshape(img, [-1, cropsize, cropsize, channels])

        #Build Xception
        main_flow = xception_entry_flow(input_layer)

        for _ in range(numMiddleXception):
            main_flow = xception_middle_block(main_flow)

        main_flow = xception_exit_flow(main_flow)


    with tf.device("/gpu:1"):
        fc = tf.layers.flatten(main_flow)
        fc = tf.contrib.layers.fully_connected(
            fc,
            fc_features1)
        fc = tf.reshape(fc, (-1, 8, 8, decode_channels0))

    with tf.device("/gpu:0"):
        output = autoencoding_decoder(fc)

    #Image values will be between 0 and 1
    output = tf.clip_by_value(output,
        clip_value_min=0.0,
        clip_value_max=1.0)

    if phase: #Calculate loss during training
        #resized_input = tf.image.resize_images(
        #    input_layer,
        #    (256, 256),
        #    method=tf.image.ResizeMethod.AREA,
        #    align_corners=False)
        loss = 1.0-tf_ssim(output, input_layer)
    else:
        loss = -1

    return loss, output

## Initial model
def cnn_model_fn_initial(features, labels, mode):
    '''Atrous convolutional encoder-decoder noise-removing network'''
    
    '''Helper functions'''
    def aspp_block(input):
        '''Atrous spatial pyramid pooling'''

        #Convolutions at multiple rates
        conv1x1 = tf.layers.conv2d(
        inputs=input,
        filters=aspp_filters,
        kernel_size=1,
        padding="same")

        conv3x3_rateSmall = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateSmall)

        conv3x3_rateMedium = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateMedium)

        conv3x3_rateLarge = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateLarge)

        #Image-level features
        pooling = tf.nn.pool(
            input=input,
            window_shape=(2,2),
            pooling_type="AVG",
            padding="same")

        #Concatenate the atrous and image-level features
        concatenation = tf.concat(
        values=[conv1x1, conv3x3_rateSmall, conv3x3_rateMedium, conv3x3_rateLarge, pooling],
        axis=axis)

        #Reduce the number of channels
        reduced = tf.layers.conv2d( #Not sure if this is the correct way to reshape...
            inputs=input,
            filters=aspp_filters,
            kernel_size=1,
            padding="same")

        return reduced

    '''Model building'''
    input_layer = tf.reshape(features["x"], [-1, rows, cols, 1])

    #Encoder
    cnn1 = tf.nn.convolution(
        input=input_layer,
        filter=features1,
        padding="same",
        activation=tf.nn.relu)

    cnn1_strided = tf.layers.conv2d(
        inputs=cnn1,
        filters=features1,
        kernel_size=3,
        strides=2,
        padding='same',
        activation=tf.nn.relu)

    cnn2 = tf.nn.convolution(
        input=cnn1_strided,
        filter=features2,
        padding="same",
        activation=tf.nn.relu)

    cnn2_strided = tf.layers.conv2d(
        inputs=cnn2,
        filters=features2,
        kernel_size=3,
        strides=2,
        padding='same',
        activation=tf.nn.relu)

    cnn3 = tf.nn.convolution(
        input=cnn2_strided,
        filter=features3,
        padding="same",
        activation=tf.nn.relu)

    cnn3_strided = tf.layers.conv2d(
        inputs=cnn3,
        filters=features3,
        kernel_size=3,
        strides=2,
        padding='same',
        activation=tf.nn.relu)

    cnn4 = tf.nn.convolution(
        input=cnn3_strided,
        filter=features4,
        padding="same",
        activation=tf.nn.relu)

    cnn4_strided = tf.layers.conv2d(
        inputs=cnn4,
        filters=features3,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=tf.nn.relu)

    #Atrous spatial pyramid pooling
    aspp = aspp_block(cnn4_strided)

    #Upsample the semantics by a factor of 4
    #upsampled_aspp = tf.image.resize_bilinear(
    #    images=aspp,
    #    tf.shape(aspp)[1:3],
    #    align_corners=True)

    '''Deconvolute the semantics'''
    concat3 = tf.concat(
        values=[cnn3, aspp],
        axis=axis)

    deconv3 = tf.layers.conv2d_transpose(
        inputs=concat3,
        filters=features3,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)

    deconv3_strided = tf.layers.conv2d_transpose(
        inputs=concat2,
        filters=features2,
        kernel_size=3,
        strides=2,
        padding="same",
        activation=tf.nn.relu)

    concat2 = tf.concat(
        values=[cnn2, deconv3_strided],
        axis=axis)

    deconv2 = tf.layers.conv2d_transpose(
        inputs=concat2,
        filters=features2,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)

    deconv2_strided = tf.layers.conv2d_transpose(
        inputs=concat2,
        filters=features1,
        kernel_size=3,
        strides=2,
        padding="same",
        activation=tf.nn.relu)

    concat1 = tf.concat(
        values=[cnn1, deconv2_strided],
        axis=axis)

    deconv1 = tf.layers.conv2d_transpose(
        inputs=concat1,
        filters=features1,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)

    deconv_final = tf.layers.conv2d_transpose(
        inputs=deconv1,
        filters=1,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)

    #Residually connect the input to the output
    output = input_layer + deconv_final

    '''Evaluation'''
    predictions = {
        "output": output
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels=labels,
                                        predictions=output)

    '''Training'''
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    '''Evaluation'''
    eval_metric_ops = {
        "loss": loss }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


## Initial idea: aspp, batch norm + PRELU, no residual connection at end, lower feature numbers
def cnn_model_fn_idea1(features, labels, mode):
    '''Atrous convolutional encoder-decoder noise-removing network'''
    
    '''Helper functions'''
    def aspp_block(input):
        '''Atrous spatial pyramid pooling'''

        #Convolutions at multiple rates
        conv1x1 = tf.layers.conv2d(
        inputs=input,
        filters=aspp_filters,
        kernel_size=1,
        padding="same")

        conv3x3_rateSmall = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateSmall)

        conv3x3_rateMedium = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateMedium)

        conv3x3_rateLarge = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateLarge)

        #Image-level features
        pooling = tf.nn.pool(
            input=input,
            window_shape=(2,2),
            pooling_type="AVG",
            padding="same")

        #Concatenate the atrous and image-level features
        concatenation = tf.concat(
        values=[conv1x1, conv3x3_rateSmall, conv3x3_rateMedium, conv3x3_rateLarge, pooling],
        axis=axis)

        #Reduce the number of channels
        reduced = tf.layers.conv2d( #Not sure if this is the correct way to reshape...
            inputs=input,
            filters=aspp_filters,
            kernel_size=1,
            padding="same")

        return reduced

    '''Model building'''
    input_layer = tf.reshape(features["x"], [-1, rows, cols, 1])

    #Encoder
    cnn1 = tf.nn.convolution(
        input=input_layer,
        filter=features1,
        padding="same",
        activation=tf.nn.relu)

    cnn1_strided = tf.layers.conv2d(
        inputs=cnn1,
        filters=features1,
        kernel_size=3,
        strides=2,
        padding='same',
        activation=tf.nn.relu)

    cnn2 = tf.nn.convolution(
        input=cnn1_strided,
        filter=features2,
        padding="same",
        activation=tf.nn.relu)

    cnn2_strided = tf.layers.conv2d(
        inputs=cnn2,
        filters=features2,
        kernel_size=3,
        strides=2,
        padding='same',
        activation=tf.nn.relu)

    cnn3 = tf.nn.convolution(
        input=cnn2_strided,
        filter=features3,
        padding="same",
        activation=tf.nn.relu)

    cnn3_strided = tf.layers.conv2d(
        inputs=cnn3,
        filters=features3,
        kernel_size=3,
        strides=2,
        padding='same',
        activation=tf.nn.relu)

    cnn4 = tf.nn.convolution(
        input=cnn3_strided,
        filter=features4,
        padding="same",
        activation=tf.nn.relu)

    cnn4_strided = tf.layers.conv2d(
        inputs=cnn4,
        filters=features3,
        kernel_size=4,
        strides=2,
        padding='same',
        activation=tf.nn.relu)

    #Atrous spatial pyramid pooling
    aspp = aspp_block(cnn4_strided)

    #Upsample the semantics by a factor of 4
    #upsampled_aspp = tf.image.resize_bilinear(
    #    images=aspp,
    #    tf.shape(aspp)[1:3],
    #    align_corners=True)

    '''Deconvolute the semantics'''
    concat3 = tf.concat(
        values=[cnn3, aspp],
        axis=axis)

    deconv3 = tf.layers.conv2d_transpose(
        inputs=concat3,
        filters=features3,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)

    deconv3_strided = tf.layers.conv2d_transpose(
        inputs=concat2,
        filters=features2,
        kernel_size=3,
        strides=2,
        padding="same",
        activation=tf.nn.relu)

    concat2 = tf.concat(
        values=[cnn2, deconv3_strided],
        axis=axis)

    deconv2 = tf.layers.conv2d_transpose(
        inputs=concat2,
        filters=features2,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)

    deconv2_strided = tf.layers.conv2d_transpose(
        inputs=concat2,
        filters=features1,
        kernel_size=3,
        strides=2,
        padding="same",
        activation=tf.nn.relu)

    concat1 = tf.concat(
        values=[cnn1, deconv2_strided],
        axis=axis)

    deconv1 = tf.layers.conv2d_transpose(
        inputs=concat1,
        filters=features1,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)

    deconv_final = tf.layers.conv2d_transpose(
        inputs=deconv1,
        filters=1,
        kernel_size=3,
        padding="same",
        activation=tf.nn.relu)

    #Residually connect the input to the output
    output = input_layer + deconv_final

    '''Evaluation'''
    predictions = {
        "output": output
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    #Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.mean_squared_error(labels=labels,
                                        predictions=output)

    '''Training'''
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
    '''Evaluation'''
    eval_metric_ops = {
        "loss": loss }
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

### Initial model
#def cnn_model_fn_initialIdea(features, labels, mode):
#    '''Atrous convolutional encoder-decoder noise-removing network'''
    
#    '''Helper functions'''
#    def aspp_block(input):
#        '''Atrous spatial pyramid pooling'''

#        #Convolutions at multiple rates
#        conv1x1 = tf.layers.conv2d(
#        inputs=input,
#        filters=aspp_filters,
#        kernel_size=1,
#        padding="same")

#        conv3x3_rateSmall = tf.layers.conv2d(
#            inputs=input,
#            filters=aspp_filters,
#            kernel_size=3,
#            padding="same",
#            dilation_rate=aspp_rateSmall)

#        conv3x3_rateMedium = tf.layers.conv2d(
#            inputs=input,
#            filters=aspp_filters,
#            kernel_size=3,
#            padding="same",
#            dilation_rate=aspp_rateMedium)

#        conv3x3_rateLarge = tf.layers.conv2d(
#            inputs=input,
#            filters=aspp_filters,
#            kernel_size=3,
#            padding="same",
#            dilation_rate=aspp_rateLarge)

#        #Image-level features
#        pooling = tf.nn.pool(
#            input=input,
#            window_shape=(2,2),
#            pooling_type="AVG",
#            padding="same")

#        #Concatenate the atrous and image-level features
#        concatenation = tf.concat(
#        values=[conv1x1, conv3x3_rateSmall, conv3x3_rateMedium, conv3x3_rateLarge, pooling],
#        axis=axis)

#        #Reduce the number of channels
#        reduced = tf.layers.conv2d( #Not sure if this is the correct way to reshape...
#            inputs=input,
#            filters=aspp_filters,
#            kernel_size=1,
#            padding="same")

#        return reduced

#    '''Model building'''
#    input_layer = tf.reshape(features["x"], [-1, rows, cols, 1])

#    #Encoder
#    cnn1 = tf.nn.convolution(
#        input=input_layer,
#        filter=features1,
#        padding="same",
#        activation=tf.nn.relu)

#    cnn1_strided = tf.layers.conv2d(
#        inputs=cnn1,
#        filters=features1,
#        kernel_size=3,
#        strides=2,
#        padding='same',
#        activation=tf.nn.relu)

#    cnn2 = tf.nn.convolution(
#        input=cnn1_strided,
#        filter=features2,
#        padding="same",
#        activation=tf.nn.relu)

#    cnn2_strided = tf.layers.conv2d(
#        inputs=cnn2,
#        filters=features2,
#        kernel_size=3,
#        strides=2,
#        padding='same',
#        activation=tf.nn.relu)

#    cnn3 = tf.nn.convolution(
#        input=cnn2_strided,
#        filter=features3,
#        padding="same",
#        activation=tf.nn.relu)

#    cnn3_strided = tf.layers.conv2d(
#        inputs=cnn3,
#        filters=features3,
#        kernel_size=3,
#        strides=2,
#        padding='same',
#        activation=tf.nn.relu)

#    cnn4 = tf.nn.convolution(
#        input=cnn3_strided,
#        filter=features4,
#        padding="same",
#        activation=tf.nn.relu)

#    cnn4_strided = tf.layers.conv2d(
#        inputs=cnn4,
#        filters=features3,
#        kernel_size=4,
#        strides=2,
#        padding='same',
#        activation=tf.nn.relu)

#    #Atrous spatial pyramid pooling
#    aspp = aspp_block(cnn4_strided)

#    #Upsample the semantics by a factor of 4
#    #upsampled_aspp = tf.image.resize_bilinear(
#    #    images=aspp,
#    #    tf.shape(aspp)[1:3],
#    #    align_corners=True)

#    '''Deconvolute the semantics'''
#    concat3 = tf.concat(
#        values=[cnn3, aspp],
#        axis=axis)

#    deconv3 = tf.layers.conv2d_transpose(
#        inputs=concat3,
#        filters=features3,
#        kernel_size=3,
#        padding="same",
#        activation=tf.nn.relu)

#    deconv3_strided = tf.layers.conv2d_transpose(
#        inputs=concat2,
#        filters=features2,
#        kernel_size=3,
#        strides=2,
#        padding="same",
#        activation=tf.nn.relu)

#    concat2 = tf.concat(
#        values=[cnn2, deconv3_strided],
#        axis=axis)

#    deconv2 = tf.layers.conv2d_transpose(
#        inputs=concat2,
#        filters=features2,
#        kernel_size=3,
#        padding="same",
#        activation=tf.nn.relu)

#    deconv2_strided = tf.layers.conv2d_transpose(
#        inputs=concat2,
#        filters=features1,
#        kernel_size=3,
#        strides=2,
#        padding="same",
#        activation=tf.nn.relu)

#    concat1 = tf.concat(
#        values=[cnn1, deconv2_strided],
#        axis=axis)

#    deconv1 = tf.layers.conv2d_transpose(
#        inputs=concat1,
#        filters=features1,
#        kernel_size=3,
#        padding="same",
#        activation=tf.nn.relu)

#    deconv_final = tf.layers.conv2d_transpose(
#        inputs=deconv1,
#        filters=1,
#        kernel_size=3,
#        padding="same",
#        activation=tf.nn.relu)

#    #Residually connect the input to the output
#    output = input_layer + deconv_final

#    '''Evaluation'''
#    predictions = {
#        "output": output
#    }

#    if mode == tf.estimator.ModeKeys.PREDICT:
#        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

#    #Calculate Loss (for both TRAIN and EVAL modes)
#    loss = tf.losses.mean_squared_error(labels=labels,
#                                        predictions=output)

#    '''Training'''
#    if mode == tf.estimator.ModeKeys.TRAIN:
#        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#        train_op = optimizer.minimize(
#            loss=loss,
#            global_step=tf.train.get_global_step())
#        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
    
#    '''Evaluation'''
#    eval_metric_ops = {
#        "loss": loss }
#    return tf.estimator.EstimatorSpec(
#        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


##Write data to tfrecord

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

## 1) Training data...

#Open the TFRecords file
writer = tf.python_io.TFRecordWriter(train_filename)

for i in range(len(train_addrs)):
    #Print how many images are saved every 1000 images
    if not i % 1000:
        print('Train data: {}/{}'.format(i, len(train_addrs)))
        
    #Load the image
    img = load_image(train_addrs[i])

    #Create a feature
    feature = {'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}

    #Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    #Serialize to string and write on the file
    writer.write(example.SerializeToString())
    
writer.close()

## 2) Validation data

#Open the TFRecords file
writer = tf.python_io.TFRecordWriter(val_filename)

for i in range(len(val_addrs)):
    #Print how many images are saved every 1000 images
    if not i % 1000:
        print('Val data: {}/{}'.format(i, len(val_addrs)))
        #sys.stdout.flush()
    
    #Load the image
    img = load_image(val_addrs[i])
    
    #Create a feature
    feature = {'val/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    
    #Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    
    # Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()

## 3) Test data

#Open the TFRecords file
writer = tf.python_io.TFRecordWriter(test_filename)

for i in range(len(test_addrs)):
    #Print how many images are saved every 1000 images
    if not i % 1000:
        print('Test data: {}/{}'.format(i, len(test_addrs)))
        #sys.stdout.flush()

    #Load the image
    img = load_image(test_addrs[i])
    
    #Create a feature
    feature = {'test/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
    
    #Create an example protocol buffer
    example = tf.train.Example(features=tf.train.Features(feature=feature))

    #Serialize to string and write on the file
    writer.write(example.SerializeToString())

writer.close()

#def dataset_input_fn():
#    filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
#    dataset = tf.data.TFRecordDataset(filenames)

#    # Use `tf.parse_single_example()` to extract data from a `tf.Example`
#    # protocol buffer, and perform any additional per-record preprocessing.
#    def parser(record):
#        keys_to_features = {
#            "image_raw": tf.FixedLenFeature((), tf.string, default_value=""),
#        }
#        parsed = tf.parse_single_example(record, keys_to_features)
#        tf.decode_raw(features['image_raw'], tf.uint16)

#        image_shape = tf.pack([height, width, 1])
#        image = tf.reshape(image, image_shape)

#        # Perform additional preprocessing on the parsed data
#        return image

#    # Use `Dataset.map()` to build a pair of a feature dictionary and a label
#    # tensor for each example.
#    dataset = dataset.map(parser)
#    dataset = dataset.shuffle(buffer_size=10000)
#    dataset = dataset.batch(32)
#    dataset = dataset.repeat(num_epochs)
#    iterator = dataset.make_one_shot_iterator()

#    # `features` is a dictionary in which each value is a batch of values for
#    # that feature; `labels` is a batch of labels.
#    features, labels = iterator.get_next()

#    return features, labels

def parser(record):
    '''Parse a TFRecord and return a training example'''
    features = {
        "image": tf.FixedLenFeature((), tf.string, ""),
    }
    parsed = tf.parse_single_example(record, features)
    img = tf.decode_raw(parsed["image"], tf.float32)

    image_shape = [height, width, 1]
    img = tf.reshape(img, image_shape)

    return img

def gen_lq(img, mean):
    '''Generate low quality image'''
    print(type(img))
    print(img)
    #Ensure that the seed is random
    np.random.seed(int(np.random.rand()*(2**32-1)))

    #Adjust the image scale so that the image has the correct average counts
    lq = np.random.poisson(mean * (img / np.mean(img)))
    
    #Rescale between 0 and 1
    min = np.min(lq)
    lq = (lq-min) / (np.max(lq)-min)

    return lq, img

def generator():
    yield gen_lq()

def input_fn(filenames, mean):
    """How records will be used"""

    files = tf.data.TFRecordDataset(filenames)
    dataset = files.interleave(tf.data.TFRecordDataset, cycle_length=parallel_readers)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(map_func=parser)
    dataset = dataset.map(lambda img: tuple(tf.py_func(gen_lq, [img, mean], [tf.float32, tf.float32])))
    #dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.repeat(num_epochs)

    iter = dataset.make_one_shot_iterator()
    with tf.Session() as sess:
        next = iter.get_next()
        print(sess.run(next)) #Output

    lq, img = iter.get_next()

    return lq, img

def main(unused_argv):

    mean = 128 #Average value of pixels in low quality generated images
    
    lq_train, img_train = input_fn(tfrecords_train_filenames, mean)
    lq_val, img_val = input_fn(tfrecords_val_filenames, mean)
    lq_test, img_test = input_fn(tfrecords_test_filenames, mean)
    
    #Create the Estimator
    estimator = tf.estimator.Estimator(model_fn=cnn_model_fn_enhancer, model_dir=model_dir)

    # Set up logging for predictions. ADD METRICS LATER
    tensors_to_log = {  }#"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=50)

    #train_iter = train_data.make_one_shot_iterator()

    #next_example, next_label = iterator.get_next()

    with tf.Session() as sess:
        print(sess.run(lq_train)) # output

    #loss = loss_function(lq, img)

    #training_op = tf.train.AdagradOptimizer(...).minimize(loss)

    #with tf.train.MonitoredTrainingSession(...) as sess:
    #    while not sess.should_stop():
    #        sess.run(training_op)

    ##Batch normalisation
    ##update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    ##with tf.control_dependencies(update_ops):
    ##    #train_op = optimizer.minimize(loss)
    #with tf.Session() as sess:

    #    #Handle training data
    #    train_iter         = train_data.make_one_shot_iterator()
    #    train_iter_handle = sess.run(train_iter.string_handle())
    
    #    #Set up iteration
    #    handle = tf.placeholder(tf.string, shape=[])
    #    iterator = tf.data.Iterator.from_string_handle(
    #        handle, train_iter.output_types)
    #    lq, img = iterator.get_next()

    #    loss = loss_function(lq, img)

    #    training_op = tf.train.AdagradOptimizer(...).minimize(loss)
    #    train_loss = sess.run(loss, feed_dict={handle: train_iter_handle})

    #    print(train_loss)


    ##Generate data on the fly
    #train_data = train_data.map(lambda img: tuple(tf.py_func(
    #    gen_lq,
    #    [img, mean],
    #    [tf.float32, tf.float32])))

    ##Batch data
    #train_data.batch(batch_size)

    ##Batch normalisation
    #update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    #with tf.control_dependencies(update_ops):
    #    #train_op = optimizer.minimize(loss)
    #    with tf.Session() as sess:

    #        #Handle training data
    #        train_iter         = train_data.make_one_shot_iterator()
    #        train_iter_handle = sess.run(train_iter.string_handle())
    
    #        #Set up iteration
    #        handle = tf.placeholder(tf.string, shape=[])
    #        iterator = tf.data.Iterator.from_string_handle(
    #            handle, train_iter.output_types)
    #        next_element = iterator.get_next()
    
    #        #temp
    #        loss = next_element

    #        train_loss = sess.run(loss, feed_dict={handle: train_iter_handle})
    #        print(train_loss)

    #Classify training, validation and test data
    #train_data = tf.data.TFRecordDataset(tfrecords_train_filenames)
    #val_data = tf.data.TFRecordDataset(tfrecords_val_filenames)
    #test_data = tf.data.TFRecordDataset(tfrecords_test_filenames)

    #Process records to get training examples
    #train_data.map(map_func=parser)
    #val_data.map(map_func=parser)
    #test_data.map(map_func=parser)

    ##Generate data on the fly
    #train_data = train_data.map(lambda img: tuple(tf.py_func(
    #    gen_lq,
    #    [img, mean],
    #    [tf.float32, tf.float32])))
    #val_data = val_data.map(lambda img: tuple(tf.py_func(
    #    gen_lq,
    #    [img, mean],
    #    [tf.float32, tf.float32])))
    #test_data = test_data.map(lambda img: tuple(tf.py_func(
    #    gen_lq,
    #    [img, mean],
    #    [tf.float32, tf.float32])))

    ##Batch data
    #train_data.batch(batch_size)
    #val_data.batch(batch_size)
    #test_data.batch(batch_size)

    #train_iter         = train_data.make_initializable_iterator()
    #train_next_element = train_iter.get_next()

    #val_iter         = val_data.make_initializable_iterator()
    #val_next_element = val_iter.get_next()

    #test_iter         = test_data.make_initializable_iterator()
    #test_next_element = test_iter.get_next()

    ##Outer wrap important for batch normalisation to work
    #with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    #    with tf.Session() as sess:

    #        #Locate data
    #        feature = { 'image': tf.FixedLenFeature([], tf.string) }
    #        filename_queue = tf.train.string_input_producer(tfrecords_train_filenames, num_epochs=1)

    #        #Define a reader and read the next record
    #        reader = tf.TFRecordReader()
    #        _, serialized_example = reader.read(filename_queue)

    #        #Decode the record read by the reader
    #        features = tf.parse_single_example(serialized_example, features=feature)

    #        image = tf.decode_raw(features['image'], tf.float32)

    #        # Reshape image data into its original shape
    #        image_shape = [height, width, 1]
    #        image = tf.reshape(image, image_shape)

    #        #lq_np = gen_lq(image.eval(), mean)
    #        #lq_img = tf.constant(lq_np)

    #        # Creates batches by randomly shuffling tensors
    #        images = tf.train.shuffle_batch(
    #            [image], 
    #            batch_size=10, 
    #            capacity=64, 
    #            num_threads=1, 
    #            min_after_dequeue=10)

    #        # Initialize all global and local variables
    #        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #        sess.run(init_op)

    #        # Create a coordinator and run all QueueRunner objects
    #        coord = tf.train.Coordinator()
    #        threads = tf.train.start_queue_runners(coord=coord)

    #        #sess.run(train_iter.initializer)

    #        while True:
    #            try:
    #                img = sess.run([images])

    #                cv2.namedWindow('dfsd',cv2.WINDOW_NORMAL)
    #                cv2.imshow("dfsd", img.reshape((2048,2048)))
    #                cv2.waitKey(0)

    #                #elem = sess.run(train_next_element)
    #                print('Success')
    #            except tf.errors.OutOfRangeError:
    #                print('End of dataset.')
    #                break

    #        # Stop the threads
    #        coord.request_stop()
    
    #        # Wait for threads to stop
    #        coord.join(threads)
    #        sess.close()

    ## Initial idea: aspp, batch norm + Leaky PRELU, residual connection and lower feature numbers
def architecture(features, mode):
    """Atrous convolutional encoder-decoder noise-removing network"""

    phase = mode == tf.estimator.ModeKeys.TRAIN
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
            padding="same")

        conv_block = tf.contrib.layers.batch_norm(
            conv_block, 
            center=True, scale=True, 
            is_training=phase)

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
            padding="same",
            name="1x1")
        conv1x1 = tf.contrib.layers.batch_norm(
            conv1x1, 
            center=True, scale=True, 
            is_training=phase)

        conv3x3_rateSmall = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateSmall,
            name="lowRate")
        conv3x3_rateSmall = tf.contrib.layers.batch_norm(
            conv3x3_rateSmall, 
            center=True, scale=True, 
            is_training=phase)

        conv3x3_rateMedium = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateMedium,
            name="mediumRate")
        conv3x3_rateMedium = tf.contrib.layers.batch_norm(
            conv3x3_rateMedium, 
            center=True, scale=True, 
            is_training=phase)

        conv3x3_rateLarge = tf.layers.conv2d(
            inputs=input,
            filters=aspp_filters,
            kernel_size=3,
            padding="same",
            dilation_rate=aspp_rateLarge,
            name="highRate")
        conv3x3_rateLarge = tf.contrib.layers.batch_norm(
            conv3x3_rateLarge, 
            center=True, scale=True, 
            is_training=phase)

        #Image-level features
        pooling = tf.nn.pool(
            input=input,
            window_shape=(2,2),
            pooling_type="AVG",
            padding="SAME")
        #Use 1x1 convolutions to project into a feature space the same size as the atrous convolutions'
        pooling = tf.layers.conv2d(
            inputs=pooling,
            filters=aspp_filters,
            kernel_size=1,
            padding="SAME",
            name="imageLevel")
        pooling = tf.contrib.layers.batch_norm(
            pooling,
            center=True, scale=True,
            is_training=phase)

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

    def split_separable_conv2d(
        inputs,
        filters,
        rate=1,
        stride=1,
        weight_decay=0.00004,
        depthwise_weights_initializer_stddev=0.33,
        pointwise_weights_initializer_stddev=0.06,
        scope=''):

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
            depth_multiplier=1,
            rate=rate,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=depthwise_weights_initializer_stddev),
            weights_regularizer=None)

        outputs = tf.contrib.layers.batch_norm(
            outputs, 
            center=True, scale=True, 
            is_training=phase)

        outputs = tf.nn.leaky_relu(
            features=outputs,
            alpha=0.2)

        outputs = slim.conv2d(
            outputs,
            filters,
            kernel_size=1,
            stride=stride,
            weights_initializer=tf.truncated_normal_initializer(
                stddev=pointwise_weights_initializer_stddev),
            weights_regularizer=slim.l2_regularizer(weight_decay))

        return outputs

    def deconv_block(input, filters, phase=phase):
        '''Transpositionally convolute a feature space to upsample it'''
        
        deconv_block = tf.layers.conv2d_transpose(
            inputs=input,
            filters=filters,
            kernel_size=3,
            strides=2,
            padding="SAME")

        deconv_block = tf.contrib.layers.batch_norm(
            deconv_block, 
            center=True, scale=True, 
            is_training=phase)

        deconv_block = tf.nn.leaky_relu(
            features=deconv_block,
            alpha=0.2)

        return deconv_block

    '''Model building'''
    input_layer = tf.reshape(features, [-1, height, width, channels])

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
        filters=features2)
    cnn2_strided = split_separable_conv2d(
        inputs=cnn2_last,
        filters=features2,
        rate=2,
        stride=2)

    #Encoding block 3
    cnn3 = conv_block(
        input=cnn2_strided,
        filters=features3)
    cnn3_last = conv_block(
        input=cnn3,
        filters=features3)
    cnn3_strided = split_separable_conv2d(
        inputs=cnn3_last,
        filters=features3,
        rate=2,
        stride=2)

    #Encoding block 4
    cnn4 = conv_block(
        input=cnn3_strided,
        filters=features4)
    cnn4_last = conv_block(
        input=cnn4,
        filters=features4)
    #cnn4_strided = split_separable_conv2d(
    #    inputs=cnn4_last,
    #    filters=features4,
    #    rate=2,
    #    stride=2)

    ##Atrous spatial pyramid pooling
    #aspp = aspp_block(cnn4_strided)

    aspp = aspp_block(cnn4_last)

    #Upsample the semantics by a factor of 4
    #upsampled_aspp = tf.image.resize_bilinear(
    #    images=aspp,
    #    tf.shape(aspp)[1:3],
    #    align_corners=True)

    #Decoding block 1 (deepest)
    deconv4 = conv_block(aspp, features4)
    deconv4 = conv_block(deconv4, features4)
    
    #Decoding block 2
    deconv4to3 = deconv_block(deconv4, features4)
    concat3 = tf.concat(
        values=[deconv4to3, cnn3_last],
        axis=concat_axis)
    deconv3 = conv_block(concat3, features3)
    deconv3 = conv_block(deconv3, features3)

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

    #Create final image with 1x1 convolutions
    deconv_final = tf.layers.conv2d_transpose(
        inputs=deconv1,
        filters=1,
        kernel_size=3,
        padding="SAME")

    #Residually connect the input to the output
    output = deconv_final#+input_layer

    #Image values will be between 0 and 1
    output = tf.clip_by_value(
        output,
        clip_value_min=0,
        clip_value_max=1)

    loss = tf.reduce_mean(tf.squared_difference(input_layer, output))

    tf.summary.histogram("loss", loss)

    return loss, output
