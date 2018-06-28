import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
import scipy.misc
import scipy.io
import math
import tensorflow as tf
from sys import stderr
from functools import reduce
import time

import cv2

#Helper functions
def imread(path):
    return scipy.misc.imread(path, mode='F')

def imsave(path, img):
    scipy.misc.imsave(path, img)

def scale0to1(img):
    """Rescale image between 0 and 1"""
    min = np.min(img)
    max = np.max(img)
    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)
    return img

def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return

def transfer_style(img_content,
                   img_style,
                   img_mask=None,
                   shape=None,
                   weight_content=1.,
                   weight_style=25., #Heavily weight style
                   weight_mask=None,
                   weight_tv=40.,
                   rel_content=None,
                   rel_style=None,
                   rel_mask=None,
                   input_noise=0.6,
                   n_checkpoints=10,
                   n_iterations_checkpoint=25,
                   save_checkpoint_imgs=False,
                   path_output='output',
                   layers_style_weights = [0.2,0.2,0.2,0.2,0.2], #Deeper weights towards right
                   greyscale=True,
                   logging=False):
    """
    Transfers the style of one image to the content of another. Optionally, 
    emphasise conservation of original content using a mask
    img_content: Image to preserve the content of. Use list for multiple
    img_style: Image to restyle the content image with. Use list for multiple
    img_mask: Optional mask where 1.0s indicate content to be conserved in its original
    and 0.0s indicate content to restyle without this bias. Use list for multiple
    shape: Height and width of image to synthesize. If not provided, it defaults to the shape of the 
    first content image
    weight_content: Weight of content loss
    weight_style: Weight of style loss
    weight_mask: Weight of optional masking loss
    weight_threading: Weight of total variation loss to minimize differences between adjacent 
    pixels. This helps produce smoother images that look more natural. Pass None or 0. to disable
    rel_content: Relative contribution of each content image. Only needed for multiple
    rel_style: Relative contribution of each style image. Only needed for multiple
    rel_mask: Relative contribtion each mask region in case of overlap. Only needed for multiple
    input_noise: Proportion of noise to add to the content image when creating canvas
    to begin iterations from
    n_checkpoints: Number of checkpoints
    n_iterations_checkpoint: Number of iterations for each checkpoint. The total number
    of iterations is n_iterations_checkpoint / n_checkpoints
    save_checkpoint_imgs: Optionally set to true to save images at checkpoints. Useful
    to study convergence
    path_output: Location to save checkpoint images if that option is enabled
    layers_style_weights: 5 element array with proportional contributions of VGG19 layers
    to style loss
    greyscale: Keep true to optimize for greyscale images
    logging: Enable to output images to the output director during convergence
    """

    #Convert inputs to lists if they are single images
    if not isinstance(img_content, (list,)):
        img_content = [img_content]
    if not isinstance(img_style, (list,)):
        img_style = [img_style]
    if weight_mask and not isinstance(weight_mask, (list,)):
        weight_mask = [weight_mask]

    if not shape:
        shape = img_content[0].shape

    #Normalise relative contents so that they sum to 1.
    if rel_content:
        sum = np.sum(np.asarray(rel_content))
        rel_content = [x/sum for x in rel_content]
    elif len(img_content) > 1:
        weight_content /= len(img_content)
    if rel_style:
        sum = np.sum(np.asarray(rel_style))
        rel_style = [x/sum for x in rel_style]
    elif len(img_style) > 1:
        weight_style /= len(img_style)
    if rel_mask:
        sum = np.sum(np.asarray(rel_mask))
        rel_mask = [x/sum for x in rel_mask]
    elif img_mask:
        if len(img_mask) > 1:
            weight_mask /= len(img_mask)

    if path_output[-1] != '/' and path_output[-1] != '\\':
        path_output += '/'

    ## Layers
    layer_content = 'conv4_2' 
    layers_style = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

    ## VGG19 model
    path_VGG19 = 'X:/Jeffrey-Ede/models/neural-networks/style-transfer/imagenet-vgg-verydeep-19.mat'
    # VGG19 mean for standardisation (RGB)
    VGG19_mean = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    
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

    def is_grey(img):
        if len(img.shape) == 2:
            return True
        elif img.shape[2] != 1:
            return True

    if greyscale:
        for i in range(len(img_content)):
            if len(img_content[i].shape) > 2:
                if img_content[i].shape[2] > 1:
                    img_content[i] = np.mean(img_content[i], axis=2)
        for i in range(len(img_style)):
            if len(img_style[i].shape) > 2:
                if img_style[i].shape[2] > 1:
                    img_style[i] = np.mean(img_style[i], axis=2)

    #Resize images
    img_style = [cv2.resize(style[:min(style.shape[:2]), :min(style.shape[:2])], shape) 
                 for style in img_style]
    img_content = [cv2.resize(content[:min(content.shape[:2]), :min(content.shape[:2])], shape) 
                   for content in img_content]

    #Convert to rgb if greyscale
    img_content = [to_rgb(255.999*img) if is_grey(img) else 255.999*img for img in img_content]
    img_style = [to_rgb(255.999*img) if is_grey(img) else 255.999*img for img in img_style]

    # apply noise to create initial "canvas" using first content image
    noise = np.random.uniform(
            img_content[0].mean()-img_content[0].std(), 
            img_content[0].mean()+img_content[0].std(),
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

    # Setup network
    with tf.Session() as sess:
        net = {}
        net['input']   = tf.Variable(np.zeros((1, shape[0], shape[1], 3), dtype=np.float32))
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
        content_loss = 0.
        for i, content in enumerate(img_content):
            sess.run(net['input'].assign(content))
            p = sess.run(net[layer_content])  #Get activation output for content layer
            x = net[layer_content]
            p = tf.convert_to_tensor(p)

            if rel_content:
                content_loss += rel_content[i]*content_layer_loss(p, x)
            else:
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

    with tf.Session() as sess:

        style_loss = 0.
        for i, style in enumerate(img_style):
            sess.run(net['input'].assign(style))
            # style loss is calculated for each style layer and summed
            loss = 0.
            for layer, weight in zip(layers_style, layers_style_weights):
                a = sess.run(net[layer])
                x = net[layer]
                a = tf.convert_to_tensor(a)
                loss += style_layer_loss(a, x)

            if rel_style:
                style_loss += rel_style[i]*loss
            else:
                style_loss += loss

    if weight_mask:
        def get_mask_loss(a, x):
            #Calculated mse between non-zero mask pixels
            prod = float(shape[0]*shape[1])
            if greyscale:
                diff2 = img_mask_tensor * (tf.reduce_mean(a, axis=3)-tf.reduce_mean(x, axis=3))**2 / (9.*prod)
            else:
                prod *= shape[2]
                diff2 = img_mask_tensor * (a-x)**2 / prod
            sum_marked_diffs = tf.reduce_sum(diff2)
            return sum_marked_diffs

        with tf.Session() as sess:
            losses = [get_mask_loss(net['input'], x) for x in img_content_tensors]
            if rel_mask:
                mask_loss = tf.add_n([rel*loss for rel, loss in zip(rel_mask, losses)])
            else:
                mask_loss = tf.add_n([loss for loss in losses])

    if weight_threading:
        def get_threading_loss(a, x):
            volume = float((shape[0]-1)*shape[1]+shape[0]*(shape[1]-1))
            if greyscale:
                diff2 = ((tf.reduce_mean(a, axis=3)[1:,:]-tf.reduce_mean(x, axis=3)[:(shape[1]-1),:])**2 +
                         (tf.reduce_mean(a, axis=3)[:,1:]-tf.reduce_mean(x, axis=3)[:,:(shape[0]-1)])**2 
                         ) / (9.*volume)
            else:
                volume *= shape[2]
                diff2 = ((tf.reduce_mean(a, axis=3)[:,1:,:]-tf.reduce_mean(x, axis=3)[:,:(shape[1]-1),:])**2 +
                         (tf.reduce_mean(a, axis=3)[:,:,1:]-tf.reduce_mean(x, axis=3)[:,:,:(shape[0]-1)])**2 
                         ) / (9.*volume)
            sum_diff2 = tf.reduce_sum(diff2)
            return sum_diff2

        with tf.Session() as sess:
            threading_loss = get_threading_loss(net['input'])
        
    ### Define loss function and minimise
    with tf.Session() as sess:

        # loss function
        if weight_content != 1.:
            L_total  = weight_content*content_loss + weight_style * style_loss
        else:
            L_total  = content_loss + weight_style * style_loss
        
        if weight_mask:
            L_total += weight_mask * mask_loss

        if weight_tv:
            L_total += weight_tv * threading_loss
    
        # instantiate optimiser
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
          L_total, method='L-BFGS-B',
          options={'maxiter': n_iterations_checkpoint})
    
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        sess.run(net['input'].assign(img_initial))

        if logging:
            print("Starting...")

        for i in range(1, n_checkpoints+1):

            if logging:
                print("Iter: {} of {}".format(i, n_checkpoints))

            #Run optimisation
            for j in range(n_iterations_checkpoint):
                if logging:
                    print("Sub-iter: {} of {}".format(j+1, n_iterations_checkpoint))
                optimizer.minimize(sess)
        
            if logging:
                #Print costs
                stderr.write('Content loss: %g\n' % sess.run(weight_content*content_loss if weight_content else content_loss))
                stderr.write('Style loss: %g\n' % sess.run(weight_style * style_loss))
                if weight_mask:
                    stderr.write('Style loss: %g\n' % sess.run(weight_mask * mask_loss))
                stderr.write('Total loss: %g\n' % sess.run(L_total))

                #Write image
                img_output = sess.run(net['input'])
                img_output = imgunprocess(img_output)

                output_file = path_output+'/'+'step'+'_'+'%s.jpg' % (i*n_iterations_checkpoint)

                if greyscale:
                    imsave(output_file, np.mean(img_output, 2).reshape(shape).clip(0, 255).astype(np.float32))
                else:
                    imsave(output_file, img_output.reshape(shape).clip(0, 255).astype(np.float32))

            elif i == n_checkpoints:
                img_output = sess.run(net['input'])
                img_output = imgunprocess(img_output)

    if greyscale:
        return (np.mean(img_output, 2).reshape(shape).clip(0, 255)/255.).astype(np.float32)
    else:
        return img_output.reshape(shape).clip(0, 255).astype(np.float32)

if __name__ == "__main__":

    img1_loc = r'X:\Jeffrey-Ede\models\neural-networks\style-transfer\some_images\electron-microscopy\tem_img2.tif'
    img2_loc = r'X:\Jeffrey-Ede\models\neural-networks\style-transfer\some_images\random\roadside_sketch.tif'

    img1 = scale0to1(imread(img1_loc))
    img2 = scale0to1(imread(img2_loc))

    restyled_img = transfer_style(img1,
                                  img2,
                                  shape=(400,400),
                                  n_checkpoints=10,
                                  n_iterations_checkpoint=25,
                                  path_output=r'X:\Jeffrey-Ede\logging',
                                  logging=True)
    imsave(r'X:\Jeffrey-Ede\restyled_img.tif', restyled_img)