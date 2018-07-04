from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os, random
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse

import numpy as np
import tensorflow as tf

import cv2
from scipy.misc import imread

import time

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

import Image

import arrayfire as af

slim = tf.contrib.slim #For depthwise separable strided atrous convolutions

tf.logging.set_verbosity(tf.logging.DEBUG)

modelSavePeriod = 0.2 #Train timestep in hours
modelSavePeriod *= 3600 #Convert to s
model_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/psi-art-multi-gpu-1/"

log_file = model_dir+"log.txt"

save_result_every_n_imgs = 1000

def architecture(amplitude_initial, symbols, symbol_ests, size_lims, num_imgs):
    """
    Atrous convolutional encoder-decoder noise-removing network
    phase - True during training
    """

    def super_tanh(x, scale=1.):
        if scale == 1.:
            return tf.tanh(x)
        else:
            return scale*tf.tanh(x/scale)

    amplitude_initial = tf.reshape(amplitude_estimate, [cropsize, cropsize, 1])
    amplitude = tf.get_variable("amplitude", [cropsize, cropsize], 
                                initializer=amplitude_initial)

    phase = tf.get_variable("phase", [cropsize, cropsize],
                            initializer=tf.constant(0., shape=[cropsize, cropsize]))

    #Aberration coefficients
    #Note: 'a20' is defocus and 'a40' is Cs
    all_symbols = ["a22","phi22","a20",
                   "a33","phi33","a31","phi31",
                   "a44","phi44","a42","phi42","a40",
                   "a55","phi55","a53","phi53","a51","phi51",
                   "a66","phi66","a64","phi64","a62","phi62","a60"]

    aberrations = {sym: tf.get_variable(sym, [1], 
                                    initializer=tf.constant(val, [1])) 
                                    for sym, val in zip(symbols, symbol_ests)}
    aberrations.update({sym: tf.constant(0., [1]) 
                        for sym in all_symbols if sym not in aberrations})

    defocus_offset = tf.get_variable("angle_"+str(i), [1],
                              initializer=tf.constant(0., [1]))
    defocus_offset = super_tanh(defocus_offset, defocus_offset_size)

    return amplitude, phase, aberrations, defocus_offset

def energy2wavelength(v0):

    #Wavelength in Angstroms
    m0=0.5109989461*10**3 # keV / c**2
    h=4.135667662*10**(-15)*10**(-3) # eV * s
    c=2.99792458*10**8 # m / s

    return h*c/np.sqrt(v0*(2*m0+v0))*1.e10

def spatial_frequencies(shape ,sampling, return_polar=False, 
                        return_nyquist=False, wavelength=None):
    
    dkx=1./(shape[0]*sampling[0])
    dky=1./(shape[1]*sampling[1])

    if shape[0]%2==0:
        kx = np.fft.fftshift(dkx*np.arange(-shape[0]/2,shape[0]/2,1))
    else:
        kx = np.fft.fftshift(dkx*np.arange(-shape[0]/2-.5,shape[0]/2-.5,1))

    if shape[1]%2==0:
        ky = np.fft.fftshift(dky*np.arange(-shape[1]/2,shape[1]/2,1))
    else:
        ky = np.fft.fftshift(dky*np.arange(-shape[1]/2-.5,shape[1]/2-.5,1))

    ky,kx = np.meshgrid(ky,kx)

    k2 = kx**2+ky**2

    ret = (kx,ky,k2)
    if return_nyquist:
        knx = 1/(2*sampling[0])
        kny = 1/(2*sampling[1])
        Kx=kx/knx
        Ky=ky/knx
        K2 = Kx**2+Ky**2
        ret += (Kx,Ky,K2)
    if return_polar:
        theta = np.sqrt(k2*wavelength**2)
        phi = np.arctan2(ky,kx)
        ret += (theta,phi)

    return ret

def get_spatial_envelope(theta, phi, wavelength, aberrations, convergence_angle):

    a = aberrations
    if convergence_angle > 0.:

        dchi_dq=2*np.pi/self.wavelength*(\
                    (a["a22"]*tf.cos(2.*(phi-a["phi22"]))+a["a20"])*theta +\
                    (a["a33"]*tf.cos(3.*(phi-a["phi33"]))+\
                    a["a31"]*tf.cos(1.*(phi-a["phi31"])))*theta**2+\
                    (a["a44"]*tf.cos(4.*(phi-a["phi44"]))+\
                    a["a42"]*tf.cos(2.*(phi-a["phi42"]))+a["a40"])*theta**3+\
                    (a["a55"]*tf.cos(5.*(phi-a["phi55"]))+\
                    a["a53"]*tf.cos(3.*(phi-a["phi53"]))+\
                    a["a51"]*tf.cos(1.*(phi-a["phi51"])))*theta**4+\
                    (a["a66"]*tf.cos(6.*(phi-a["phi66"]))+\
                    a["a64"]*tf.cos(4.*(phi-a["phi64"]))+\
                    a["a62"]*tf.cos(2.*(phi-a["phi62"]))+a["a60"])*theta**5)

        dchi_dphi=-2*tf.pi/wavelength*(\
            1/2.*(2.*a["a22"]*tf.sin(2.*(phi-a["phi22"])))*theta +\
            1/3.*(3.*a["a33"]*tf.sin(3.*(phi-a["phi33"]))+\
                    1.*a["a31"]*tf.sin(1.*(phi-a["phi31"])))*theta**2+\
            1/4.*(4.*a["a44"]*tf.sin(4.*(phi-a["phi44"]))+\
                    2.*a["a42"]*tf.sin(2.*(phi-a["phi42"])))*theta**3+\
            1/5.*(5.*a["a55"]*tf.sin(5.*(phi-a["phi55"]))+\
                    3.*a["a53"]*tf.sin(3.*(phi-a["phi53"]))+\
                    1.*a["a51"]*tf.sin(1.*(phi-a["phi51"])))*theta**4+\
            1/6.*(6.*a["a66"]*tf.sin(6.*(phi-a["phi66"]))+\
                    4.*a["a64"]*tf.sin(4.*(phi-a["phi64"]))+\
                    2.*a["a62"]*tf.sin(2.*(phi-a["phi62"])))*theta**5)

        spatial=tf.exp(-np.sign(convergence_angle)*(
            convergence_angle/2)**2*(dchi_dq**2+dchi_dphi**2))
    else:
        spatial=None

    return spatial

def get_temporal_envelope(theta, wavelength, focal_spread):

    temporal = tf.exp(-np.sign(focal_spread)*(
        .5*np.pi/wavelength*focal_spread*theta**2)**2)

    return temporal

def get_aperture_envelope(theta, aperture): #Not in use at the moment

    if np.isfinite(aperture):
        aperture = np.ones_like(theta)
        aperture[theta > self.aperture + self.aperture_edge] = 0.
        ind=(theta > self.aperture)&(theta < self.aperture_edge + self.aperture)
        aperture[ind]*= .5*(1+np.cos(np.pi*(theta[ind]-self.aperture)/self.aperture_edge))
    else:
        aperture=None

    return aperture

def get_chi(theta, phi, wavelength, aberrations):

    a = aberrations
    chi=1/2.*(a["a22"]*tf.cos(2.*(phi-a["phi22"]))+a["a20"])*theta**2 +\
        1/3.*(a["a33"]*tf.cos(3.*(phi-a["phi33"]))+\
                a["a31"]*tf.cos(1.*(phi-a["phi31"])))*theta**3 +\
        1/4.*(a["a44"]*tf.cos(4.*(phi-a["phi44"]))+\
                a["a42"]*tf.cos(2.*(phi-a["phi42"]))+a["a40"])*theta**4+\
        1/5.*(a["a55"]*tf.cos(5.*(phi-a["phi55"]))+\
                a["a53"]*tf.cos(3.*(phi-a["phi53"]))+\
                a["a51"]*tf.cos(1.*(phi-a["phi51"])))*(theta**5) +\
        1/6.*(a["a66"]*tf.cos(6.*(phi-a["phi66"]))+\
                a["a64"]*tf.cos(4.*(phi-a["phi64"]))+\
                a["a62"]*tf.cos(2.*(phi-a["phi62"]))+a["a60"])*theta**6
    chi *= 2.*np.pi/wavelength

    return chi

def get_ctf(theta, phi, wavelength, aberrations, focal_spread):

    ctf = tf.exp(-1.j*get_chi(theta, phi, wavelength, aberrations))

    #aperture = self.get_aperture_envelope(theta)
    #if aperture is not None:
    #    ctf*=aperture

    temporal = get_temporal_envelope(theta, wavelength, aberrations, focal_spread)
    if temporal is not None:
        ctf *= temporal

    spatial = get_spatial_envelope(theta, phi, wavelength, aberrations)
    if spatial is not None:
        ctf *= spatial

    return ctf

def save_returned_tensor(img, loc, size):
    Image.fromarray(img.reshape(size[0], size[1]).astype(np.float32)).save( loc )
    return 

def get_affine_transform(init=None, scope=None):

    with tf.variable_scope(variable_scope, reuse=False):
        if init:
            init = init[:,:2]
        else:
            init = [[1., 0.], [0., 1.], [0., 0.]]

        tril_var = tf.variable(init, tf.float32)
        tril_const = tf.constant([[0.], [0.], [1.]], tf.float32)
        tril = tf.concat([tril_var, tril_const], axis=1)

        scale = tf.linalg.LinearOperatorLowerTriangular(tril)
        affine = tf.contrib.distributions.bijectors.AffineLinearOperator(shift, scale)

        return affine

def experiment(stack, symbols, symbol_ests, middle_focus_img, 
               size_lims, cropsize, energy=200, focal_spread_est=None, 
               boot_size0=3, central_fraction=0.6):

    stack_means = [tf.convert_to_tensor(np.mean(img), dtype=tf.float32) for img in stack]
    root_stack = [tf.convert_to_tensor(np.sqrt(img), dtype=tf.float32) for img in stack]

    for i in range(middle_focus_img-1):
        affine = get_affine_transform(scope="affine"+str(i))
        img = affine.forward(root_stack[i])
        img = tf.image.central_crop(image, central_fraction)
        root_stack[i] = img

    root_stack[middle_focus_img-1] = tf.image.central_crop(root_stack[middle_focus_img-1], 
                                                           central_fraction)

    for i in range(middle_focus_img, num):
        affine = get_affine_transform(scope="affine"+str(i))
        img = affine.forward(root_stack[i])
        img = tf.image.central_crop(image, central_fraction)
        root_stack[i] = img

    if not focal_spread:
        focal_spread = tf.constant(0., [1])
    else:
        focal_spread = tf.get_variable("focal_spread", 
                                       [1], 
                                       initializer=tf.constant(focal_spread_est, [1]))

    #Get wavefunction and variables
    num_sym = len(symbols)
    num_imgs = len(symbols)
    amplitude, phase, aberrations, defocus_offset = architecture(
        amplitude_initial, size_lims, num_imgs, symbols=None, symbol_ests=None)
    
    #Propagate to image planes
    wavelength = energy2wavelength(energy)
    kx, ky, k2, theta, phi = spatial_frequencies(
        shape, sampling, wavelength=wavelength, return_polar=True)
    kx, ky, k2, theta, phi = [tf.convert_to_tensor(arg, dtype=tf.float32)
                              for arg in [kx, ky, k2, theta, phi]]

    wavefunction = tf.complex(real=amplitude*tf.cos(phase), 
                              imag=amplitude*tf.sin(phase))

    backpropagations = []
    #Get ctf at various defocuses
    defocus_ramp = [aberrations['a20']*np.abs(i)*3+defocus_offset 
                    for i in range(-middle_focus_img, num_imgs-middle_focus_img)]
    for i, defocus in enumerate(defocus_ramp):
        aberrations['a20'] = defocus
        ctf = get_ctf(theta, phi, wavelength, aberrations, focal_spread)
        backpropagation = tf.ifft2d(tf.fft2d(wavefunction)*ctf)
        backpropagations.append(stack_means[i]*backpropagation/tf.reduce_mean(backpropagation))

    #See https://github.com/jcjohnson/neural-style/wiki/Fine-Tuning-The-Adam-Optimizer
    #about the choice of ADAM parameters
    learning_rate_ph = tf.placeholder(tf.float32, name="learning_rate")
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_ph, 
                                      beta1=0.99, 
                                      epsilon=0.1)

    #Calculate losses
    mse_losses = []
    for i in range(num_imgs):
        mse_losses.append(
            tf.reduce_mean(
                tf.losses.mean_squared_error(root_stack[i], backpropagations[i])))

    #saver = tf.train.Saver()
    #saver.restore(sess, tf.train.latest_checkpoint("//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/denoiser-multi-gpu-13/model/"))
    #del saver

    with open(log_file, 'a') as log:
        log.flush()

        with tf.Session() as sess:

            sess.run(tf.initialize_all_variables())

            #Modes
            STOP = 0
            ALIGN = 1
            DISTIL = 2

            learning_rate = 0.001
            counter = 0
            lb = ub = middle_focus_img-1
            for boot_size in range(boot_size0, num_imgs):

                #Only use a portion of the stack to calculate distillation losses

                lb = middle_focus_img - boot_size//2 - boot_size%2
                ub = middle_focus_img + boot_size//2

                if lb < 0:
                    ub += np.absolute(lb)
                    lb = 0
                if ub >= num_imgs:
                    lb -= ub - (num_imgs-1)
                    ub = num_imgs-1

                mse = tf.add_n(mse_losses[lb:ub]) / (boot_size+1)

                train_op = optimizer.minimize(loss=mse, 
                                              var_list=var_to_train,
                                              global_step=tf.train.get_global_step())

                #Use prior aberration coefficients to align the next image
                mode == ALIGN
                while mode == ALIGN:

                    try:
                        with open(model_dir+"learning_rate.txt", "r") as lrf:
                            learning_rate_list = [float(line) for line in lrf]
                            learning_rate = np.float32(learning_rate_list[0])
                            print("Using learning rate: {}".format(learning_rate))
                    except:
                        pass

                    while time.time()-time0 < modelSavePeriod:
                        counter += 1

                        if counter <= 1 or not counter % save_result_every_n_batches:

                            try:
                                save_amplitude_loc = model_dir+"amplitude-"+str(counter)+".tif"
                                save_phase_loc = model_dir+"amplitude-"+str(counter)+".tif"
                                save_stack_loc = [model_dir+"output-"+str(i)+".tif" 
                                                  for i in range(num_imgs)]

                                results = sess.run([train_op, mse, amplitude, phase] + backpropagations,
                                                   feed_dict={learning_rate_ph: learning_rate})
                                mse = results[1]
                                _amplitude = results[2]
                                _phase = results[3]
                                _backpropagations = results[4:(4+num_imgs)]

                                save_returned_tensor(_amplitude, save_amplitude_loc, stack.shape)
                                save_returned_tensor(_phase, save_phase_loc, stack.shape)
                                for img, loc in zip(_backpropagations, save_stack_loc):
                                    save_returned_tensor(img, loc, stack.shape)

                                log.write("Iter: {}, Loss: {:.8f}".format(counter, mse))
                            except:
                                 print("Image save failed")
                        else:
                            _, mse = sess.run([train_op, mse])
                            log.write("Iter: {}, Loss: {:.8f}".format(counter, mse))

                        print("Iter: {}, Loss: {:.8f}".format(counter, mse))

                    saver.save(sess, save_path=model_dir+"model/", global_step=counter)

                #Refine the contrast transfer function
                while mode == DISTIL:

                    try:
                        with open(model_dir+"learning_rate.txt", "r") as lrf:
                            learning_rate_list = [float(line) for line in lrf]
                            learning_rate = np.float32(learning_rate_list[0])
                            print("Using learning rate: {}".format(learning_rate))
                    except:
                        pass

                    while time.time()-time0 < modelSavePeriod:
                        counter += 1

                        if counter <= 1 or not counter % save_result_every_n_batches:

                            try:
                                save_amplitude_loc = model_dir+"amplitude-"+str(counter)+".tif"
                                save_phase_loc = model_dir+"amplitude-"+str(counter)+".tif"
                                save_stack_loc = [model_dir+"output-"+str(i)+".tif" 
                                                  for i in range(num_imgs)]

                                results = sess.run([train_op, mse, amplitude, phase] + backpropagations,
                                                   feed_dict={learning_rate_ph: learning_rate})
                                mse = results[1]
                                _amplitude = results[2]
                                _phase = results[3]
                                _backpropagations = results[4:(4+num_imgs)]

                                save_returned_tensor(_amplitude, save_amplitude_loc, stack.shape)
                                save_returned_tensor(_phase, save_phase_loc, stack.shape)
                                for img, loc in zip(_backpropagations, save_stack_loc):
                                    save_returned_tensor(img, loc, stack.shape)

                                log.write("Iter: {}, Loss: {:.8f}".format(counter, mse))
                            except:
                                 print("Image save failed")
                        else:
                            _, mse = sess.run([train_op, mse])
                            log.write("Iter: {}, Loss: {:.8f}".format(counter, mse))

                        print("Iter: {}, Loss: {:.8f}".format(counter, mse))

                    saver.save(sess, save_path=model_dir+"model/", global_step=counter)

    return

def load_image(addr, resizeSize=None, imgType=np.float32):
    """Read an image and make sure it is of the correct type. Optionally resize it"""
    
    try:
        img = imread(addr, mode='F')
    except:
        print("Image read failed")

    if resizeSize:
        img = cv2.resize(img, resizeSize, interpolation=cv2.INTER_AREA)

    return img.astype(imgType)

def load_stack(dir):
    return [imread(dir+file, mode='F') 
            for file in os.listdir(dir) if '.tif' in file]

def scale0to1(img):
    """Rescale image between 0 and 1"""

    min = np.min(img)
    max = np.max(img)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def disp(img):

    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)

    return

def main(dir):

    stack = load_stack(dir)

    num_imgs = len(stack)

    stack_min = np.min([np.min(img) for img in stack])
    stack_max = np.max([np.max(img) for img in stack])
    stack = [(img-stack_min) / (stack_max-stack_min) for img in stack]

    symbols = ['a20', 'a40']
    symbol_ests = [50., 0.]
    middle_focus_img = 8
    energy = 200 #keV
    focal_spread = 0.

    experiment(stack, symbols, symbol_ests, middle_focus_img, 
               size_lims, cropsize, energy, focal_spread)

    return 

if __name__ == '__main__':

    dir = r'\\flexo.ads.warwick.ac.uk\shared39\EOL2100\2100\Users\Jeffrey-Ede\series72\stack1'
    if dir[-1] != ['\\']:
        dir += '\\'

    main(dir)



