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

data_dir = "X:/Jeffrey-Ede/stills_all/"
#data_dir = "E:/stills_hq-mini/"

modelSavePeriod = 4 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "X:/Jeffrey-Ede/models/dedicated_kernels/"

shuffle_buffer_size = 5000
num_parallel_calls = 4
num_parallel_readers = 4
prefetch_buffer_size = 5
batch_size = 32
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
val_log_file = model_dir+"val_log.txt"
log_every = 1 #Log every _ examples

numMeans = 64 // batch_size
scaleMean = 4 #Each means array index increment corresponds to this increase in the mean
numDynamicGrad = 1 #Number of gradients to calculate for each possible mean when dynamically updating training
lossSmoothingBoxcarSize = 5

channels = 1 #Greyscale input image

#Sidelength of images to feed the neural network
cropsize = 10#171
generator_input_size = cropsize
height_crop = width_crop = cropsize

#hparams = experiment_hparams(train_batch_size=batch_size, eval_batch_size=16)

num_workers = 1

increase_batch_size_by_factor = 1
effective_batch_size = increase_batch_size_by_factor*batch_size

val_skip_n = 10

num_filters = 1#5*6

def architectures(inputs):
    """Generates fake data to try and fool the discrimator"""

    def pad(tensor, size):
        d1_pad = size[0]
        d2_pad = size[1]

        paddings = tf.constant([[0, 0], [d1_pad, d1_pad], [d2_pad, d2_pad], [0, 0]], dtype=tf.int32)
        padded = tf.pad(tensor, paddings, mode="REFLECT")
        return padded

    def make_layer(size, type):
        
        if type == 'biases':
            init = np.array([0.], dtype=np.float32)
        if type == 'weights':
            init = np.array([1./(size*size)], dtype=np.float32)

        #print("Init: {}".format(init))

        printij = False #Set to true to debug

        if printij:
            print("\nStart:")

        variables = [[None for _ in range(size)] for _ in range(size)]

        if printij:
            for i in range(3):
                for j in range(3):
                    if variables[i][j]:
                        print(i, j, variables[i][j].name)
                    else:
                        print(i,j)
            print("\n")

        offset = size//2
        for x in range(size//2+1):
            for y in range(x+1):

                with tf.variable_scope("var_x-{}_y-{}".format(x, y), reuse=False) as scope:

                    i, j = offset+x, offset+y
                    variables[i][j] = tf.get_variable('v', dtype=tf.float32, initializer=init, trainable=True)
                    if printij:
                        print(i,j,x,y,variables[i][j].name)

                        for i in range(3):
                            for j in range(3):
                                if variables[i][j]:
                                    print(i, j, variables[i][j].name)
                                else:
                                    print(i,j)
                        print("\n")

                    if x > 0:
                        if y == 0:
                            i, j = offset-x, offset
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset, offset+x
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset, offset-x
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                        elif y == x:
                            i, j = offset+x, offset-y
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset-x, offset+y
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset-x, offset-y
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                        elif y != x:
                            i, j = offset-x, offset+y
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset+x, offset-y
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset-x, offset-y
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset+y, offset+x
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset-y, offset+x
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset+y, offset-x
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                            i, j = offset-y, offset-x
                            scope.reuse_variables()
                            variables[i][j] = tf.get_variable(name='v')
                            if printij:
                                print(i,j,x,y,variables[i][j].name)

                                for i in range(3):
                                    for j in range(3):
                                        if variables[i][j]:
                                            print(i, j, variables[i][j].name)
                                        else:
                                            print(i,j)
                                print("\n")

                #print(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="depth-1_size-3/var_x-{}_y-{}".format(x, y))))

        #print(variables)

        if printij:
            for i in range(3):
                for j in range(3):
                    print(i, j, variables[i][j].name)

        concats = []
        for i in range(size):
            concats.append(tf.concat(variables[:][i], axis=0))
        kernel = tf.stack(concats, axis=1)

        kernel = tf.expand_dims(kernel, axis=0)
        kernel = tf.expand_dims(kernel, axis=3)

        #kernel = tf.reshape(kernel, [-1, size, size, 1])

        #print(kernel)

        return kernel

    depths = [2]
    widths = [3]
    #depths = [i for i in range(1, 6)]
    #widths = [3, 5, 7, 9, 13, 17]

    max_depth = 5
    max_width = 17
    filters = []
    filter_scopes = []
    filter_depths = []
    filter_widths = []
    outputs = []
    for depth in depths:
        for width in widths:

            default_scope = "depth-{}_size-{}".format(depth, width)

            #Filter creation
            def filter_fn(input):
                    with tf.variable_scope('w0'):
                        filter = make_layer(width, 'weights')*input

                    for i in range(1, depth):
                        with tf.variable_scope('b'+str(i)):
                            filter += make_layer(width, 'biases')

                        filter = tf.sigmoid(filter)
                        filter = tf.contrib.layers.fully_connected(
                            inputs=filter, 
                            num_outputs=1,
                            activation_fn=None,
                            weights_initializer=None,
                            biases_initializer=None)
                    
                        with tf.variable_scope('w'+str(i)):
                            filter = make_layer(width, 'weights')*filter

                    output = tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(filter, axis=1), axis=1), axis=1)

                    return output

            filters.append(filter_fn)
            filter_scopes.append(default_scope)
            filter_depths.append(depth)
            filter_widths.append(width)

            #Generate outputs
            padded = pad(inputs, (width//2, width//2))

            output = [[None for _ in range(cropsize)] for _ in range(cropsize)]
            for x in range(cropsize):
                for y in range(cropsize):
                    if not x+y:
                        with tf.variable_scope(default_scope) as filter_scope:
                            output[x][y] = filter_fn(padded[:, x:(x+width), y:(y+width), :])
                    else:
                        with tf.variable_scope(filter_scope, reuse=True) as filter_scope:
                            output[x][y] = filter_fn(padded[:, x:(x+width), y:(y+width), :])

                    #print(len(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="depth-1_size-3")))

            concats = []
            for i in range(cropsize):
                concats.append(tf.stack(output[i][:], axis=1))
            output = tf.stack(concats, axis=2)

            output = tf.expand_dims(output, 3)
            #print(output)

            outputs.append(output)

    return filters, filter_scopes, filter_depths, filter_widths, outputs


def experiment(img, learning_rate_ph):

    filters, filter_scopes, filter_depths, filter_widths, outputs = architectures(img)

    losses = [tf.losses.mean_squared_error(output, img) for output in outputs]

    train_ops = []
    for i in range(len(losses)):
        optimizer = tf.train.AdamOptimizer(learning_rate_ph[0])
        train_op = optimizer.minimize(losses[i])
        train_ops.append(train_op)

    return {'filters': filters, 'filter_scopes': filter_scopes, 'filter_depths': filter_depths,
            'filter_widths': filter_widths, 'outputs': outputs, 'train_ops': train_ops,
            'losses': losses}

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

def load_image(addr, resize_size=None, img_type=np.float32):
    """Read an image and make sure it is of the correct type. Optionally resize it"""

    try:
        img = imread(addr, mode='F')
        
        x = img.shape[0]-cropsize
        y = img.shape[1]-cropsize
        x = np.random.randint(0, x)
        y = np.random.randint(0, y)

        img = img[x:(x+cropsize), y:(y+cropsize)]
    except:
        img = 0.5*np.ones((cropsize,cropsize))
        print("Image read failed")

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

    img = img[:cropsize,:cropsize] #cv2.resize(img, (cropsize, cropsize))

    img = scale0to1(img)

    img /= np.mean(img)

    return img.astype(np.float32)

def record_parser(record):
    img = preprocess(flip_rotate(load_image(record)))

    if np.sum(np.isfinite(img)) != cropsize**2:
        img = np.zeros((cropsize, cropsize))

    return img

def reshaper(img):
    img = tf.reshape(img, [cropsize, cropsize, channels])
    return img


def input_fn(dir, subset, batch_size):
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

        return img_batch

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

    print("Initializing")

    tf.reset_default_graph()

    temp = set(tf.all_variables())

    with open(log_file, 'a') as log:
        log.flush()

        with open(val_log_file, 'a') as val_log:
            val_log.flush()

            # The env variable is on deprecation path, default is set to off.
            os.environ['TF_SYNC_ON_FINISH'] = '0'
            os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) #For batch normalisation windows
            with tf.control_dependencies(update_ops):

                # Session configuration.
                log_device_placement = False #Once placement is correct, this fills up too much of the cmd window...
                sess_config = tf.ConfigProto(
                    allow_soft_placement=True,
                    log_device_placement=log_device_placement,
                    intra_op_parallelism_threads=1,
                    gpu_options=tf.GPUOptions(force_gpu_compatible=True))

                config = RunConfig(session_config=sess_config, model_dir=model_dir)

                img = input_fn(data_dir, 'train', batch_size=batch_size)
                img_val = input_fn(data_dir, 'val', batch_size=batch_size)

                with tf.Session(config=sess_config) as sess:

                    sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
                    temp = set(tf.all_variables())

                    __img = sess.run(img)
                    img_ph = [tf.placeholder(tf.float32, shape=i.shape, name='img') 
                                for i in __img]
                    del __img

                    learning_rate_ph = tf.placeholder(tf.float32, name='learning_rate')

                    exp_dict = experiment(img_ph, learning_rate_ph)

                    #########################################################################################

                    sess.run( tf.initialize_variables(set(tf.all_variables()) - temp) )
                    train_writer = tf.summary.FileWriter( logDir, sess.graph )

                    #print(tf.all_variables())
                    saver = tf.train.Saver()
                    #saver.restore(sess, tf.train.latest_checkpoint(model_dir+"model/"))

                    counter = 0
                    save_counter = counter
                    counter_init = counter+1

                    print("Session started")

                    while counter < 20000:

                        counter += 1

                        lr = np.array([0.005*(1.-counter/20001)])

                        base_dict = {learning_rate_ph: lr}

                        _img = sess.run(img)

                        feed_dict = base_dict.copy()
                        feed_dict.update({ph: img for ph, img in zip(img_ph, _img)})

                        results = sess.run( exp_dict['train_ops']+exp_dict['losses'], feed_dict=feed_dict )

                        losses = results[num_filters:]

                        print("Iter: {}, Losses: {}".format(counter, losses))

                        try:
                            log.write("Iter: {}, {}".format(counter, losses))
                        except:
                            print("Write to discr pred file failed")

                        if not counter % val_skip_n:

                            _img = sess.run(img_val)

                            feed_dict = base_dict.copy()
                            feed_dict.update({ph: img for ph, img in zip(img_ph, _img)})

                            losses = sess.run( exp_dict['losses'], feed_dict=feed_dict )

                            print("Iter: {}, Val losses: {}".format(counter, losses))

                            try:
                                val_log.write("Iter: {}, {}".format(counter, losses))
                            except:
                                print("Write to val log file failed")

                        #if counter > 100:
                        #    vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, exp_dict['filter_scopes'][0])
                        #    v = sess.run(vars)
                        #    print(v)

                        #    filtered_img = sess.run(exp_dict['outputs'], feed_dict=feed_dict)
                        #    disp(_img[0])
                        #    disp(filtered_img[0][0].reshape((cropsize,cropsize)).T)

                        #    os.system("pause")

                    #Save the model
                    saver.save(sess, save_path=model_dir+"model/", global_step=counter)
    return 

if __name__ == '__main__':

    main()


