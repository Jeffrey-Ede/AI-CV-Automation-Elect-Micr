from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

import cv2
from scipy.misc import imread

import time

slim = tf.contrib.slim #For depthwise separable strided atrous convolutions

tf.logging.set_verbosity(tf.logging.DEBUG)

modelSavePeriod = 2 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "E:/models/profiles/"

log_file = model_dir+"log.txt"

data_file = "E:/stills_all_profile/data.npy"
info_file = "E:/stills_all_profile/data_info.npy"

dataset_raw = np.load(data_file)
dataset_info = np.load(info_file)

num_inputs = len(dataset_raw[0])
num_outputs = num_inputs

layer_size1 = 2*num_inputs
layer_size2 = 2*num_inputs
layer_size3 = 2*num_inputs
layer_size_output = num_outputs

min_info = dataset_info[()]['min']
max_info = dataset_info[()]['max']
redistributors_info = dataset_info[()]['redistributors']

num_parallel_calls = 1
prefetch_buffer_size = 64
num_epochs = 100000 #Continue almost forever...


def architecture(input, ground_truth=None, finites=None, mode=None):
    """Multilayer perceptron that predicts missing values"""

    fc = tf.contrib.layers.fully_connected(
        input,
        layer_size1)

    fc = tf.contrib.layers.fully_connected(
        fc,
        layer_size2)

    fc = tf.contrib.layers.fully_connected(
        fc,
        layer_size3)

    output = tf.contrib.layers.fully_connected(
        fc,
        layer_size_output)

    if phase: #Calculate loss during training
        loss = tf.reduce_mean( tf.multiply( finites, tf.squared_difference(output, ground_truth) ) )
        loss *= num_inputs / tf.sum(finites)
    else:
        loss = -1

    return loss, output

def redistribute_params(params):
    """Redistribute parameters so that they lie of a uniform distribution in [0, 1]"""

    finites = np.ones(num_inputs)
    finites[np.isfinite(params)] = 1

    redistribution = np.zeros(num_inputs)
    for i in range(num_inputs):
        if finites[i]:
            redistribution[i] = (params[i].clip(min_info[i], max_info[i])-min_info[i]) / (max_info[i] - min_info[i])
            if redistribution[i] < redistributors_info[i][0]:
                idx = 0
            else:
                idx = next(idx for idx, val in enumerate(redistributors_info[i]) if redistribution[i] >= val)

            d = redistribution[i] / (redistributors_info[i][idx]-redistributors_info[i][idx-1])
            redistribution[i] = redistribution_info[i][idx-1] + d*(redistribution_info[i][idx]-redistribution_info[i][idx-1])

    return redistribution, finites

def parser(params):
    """Prepare inputs where some are missing and the correct outputs"""

    #Check the number of valid parameters e.g. that aren't NaNs
    inputs, finites = redistribute_params(params)

    #Mark a random number of parameters for input
    inputs_mask = np.random.randn(num_inputs)
    sorted = np.sort(inputs_mask)

    r = np.random.randint(0, num_inputs)
    inputs_mask[inputs_mask > sorted[num_inputs-1-r] ] = 1
    inputs_mask[finites == 0] = 0

    input = np.zeros(2*num_inputs)
    for i in range(num_inputs):
        if inputs_mask == 1:
            input[2*i] = inputs[i]
            input[2*i+1] = 1 #Mark as input
        else:
            input[2*i] = dataset_info['means'][i]

    print(input)
    print(params)
    print(finites)

    return input, params, finites

def input_fn(dir):
    """Create a dataset from a list of filenames"""

    dataset = tf.contrib.data.Dataset.from_tensor_slices(dataset_raw)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)
    dataset = dataset.map(
        lambda params: tuple(tf.py_func(parser, [params], [tf.float32, tf.float32])),
        num_parallel_calls=num_parallel_calls)
    #dataset = dataset.batch(batch_size=batch_size)
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    dataset = dataset.repeat(num_epochs)
    
    iter = dataset.make_one_shot_iterator()

    inputs, outputs, finites = iter.get_next()

    return inputs, outputs, finites

def main(unused_argv=None):

    temp = set(tf.all_variables())

    log = open(log_file, 'a')

    with tf.device("/cpu:0"):

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
        with tf.control_dependencies(update_ops):

            inputs, outputs = input_fn(trainDir)

            loss, prediction = architecture(inputs, outputs, tf.estimator.ModeKeys.TRAIN)
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

                        #train_writer.add_summary(summary, counter)

                    #Save the model
                    #saver.save(sess, save_path=model_dir+"model", global_step=counter)
                    tf.saved_model.simple_save(
                        session=sess,
                        export_dir=model_dir+"model-"+str(counter)+"/",
                        inputs={"lq": inputs},
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
    return 

if __name__ == "__main__":
    tf.app.run()