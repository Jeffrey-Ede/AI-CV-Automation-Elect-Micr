import os
import numpy as np
import scipy.misc
import scipy.io
import math
import tensorflow as tf
from sys import stderr
from functools import reduce
import time  

def transfer_style(img_content,
                   img_style, 
                   img_mask=None, 
                   weight_mask=None, 
                   weight_style=2.e2,
                   rel_content=None,
                   rel_style=None,
                   rel_mask=None,
                   input_noise=0.1,
                   n_checkpoints=10,
                   n_iterations_checkpoint=100,
                   save_checkpoint_imgs=False,
                   path_output='output',
                   layers_style_weights = [0.2,0.2,0.2,0.2,0.2]):
    """
    Transfers the style of one image to the content of another. Optionally, 
    emphasise conservation of original content using a mask
    img_content: Image to preserve the content of. Use list for multiple
    img_style: Image to restyle the content image with. Use list for multiple
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
    n_checkpoints: Number of checkpoints
    n_iterations_checkpoint: Number of iterations for each checkpoint. The total number
    of iterations is n_iterations_checkpoint / n_checkpoints
    save_checkpoint_imgs: Optionally set to true to save images at checkpoints. Useful
    to study convergence
    path_output: Location to save checkpoint images if that option is enabled
    layers_style_weights: 5 element array with proportional contributions of VGG19 layers
    to style loss
    """

    if not isinstance(img_content, (list,)):
        img_content = [img_content]

    if not isinstance(img_style, (list,)):
        img_style = [img_style]

    if weight_mask and not isinstance(weight_mask, (list,)):
        weight_mask = [weight_mask]

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

    # Setup network
    with tf.Session() as sess:
        a, h, w, d     = img_content.shape
        net = {}
        net['input']   = tf.Variable(np.zeros((a, h, w, d), dtype=np.float32))
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
        sess.run(net['input'].assign(img_content))
        p = sess.run(net[layer_content])  #Get activation output for content layer
        x = net[layer_content]
        p = tf.convert_to_tensor(p)
        content_loss = content_layer_loss(p, x)

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
        sess.run(net['input'].assign(img_style))
        style_loss = 0.
        # style loss is calculated for each style layer and summed
        for layer, weight in zip(layers_style, layers_style_weights):
            a = sess.run(net[layer])
            x = net[layer]
            a = tf.convert_to_tensor(a)
            style_loss += style_layer_loss(a, x)

    if weight_mask:
        def get_mask_loss(a, x):
            #Calculated mse between marked pixels
            diff2 = img_mask_tensor * (a-x)**2
            sum_marked_diffs = tf.reduce_sum(diff2)
            return sum_marked_diffs

        with tf.Session() as sess:
            mask_loss = get_mask_loss(net['input'], img_content_tensor)
        
    ### Define loss function and minimise
    with tf.Session() as sess:
        # loss function
        if weight_mask:
            L_total  = content_loss + weight_style * style_loss + weight_mask * mask_loss
        else:
            L_total  = content_loss + weight_style * style_loss
    
        # instantiate optimiser
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(
          L_total, method='L-BFGS-B',
          options={'maxiter': n_iterations_checkpoint})
    
        init_op = tf.initialize_all_variables()
        sess.run(init_op)
        sess.run(net['input'].assign(img_initial))
        for i in range(1,n_checkpoints+1):
            # run optimisation
            optimizer.minimize(sess)
        
            ## print costs
            stderr.write('Iteration %d/%d\n' % (i*n_iterations_checkpoint, n_checkpoints*n_iterations_checkpoint))
            stderr.write('  content loss: %g\n' % sess.run(content_loss))
            stderr.write('    style loss: %g\n' % sess.run(weight_style * style_loss))
            if weight_mask:
                stderr.write('    style loss: %g\n' % sess.run(weight_mask * mask_loss))
            stderr.write('    total loss: %g\n' % sess.run(L_total))

            ## write image
            img_output = sess.run(net['input'])
            img_output = imgunprocess(img_output)

            timestr = time.strftime("%Y%m%d_%H%M%S")
            output_file = path_output+'/'+timestr+'_'+'%s.jpg' % (i*n_iterations_checkpoint)

    imsave(output_file, img_output)

    return img_output

if __name__ == "__main__":
    restyled_img = transfer_style(np.random.rand(3, 4),
                                  np.random.rand(3, 4))