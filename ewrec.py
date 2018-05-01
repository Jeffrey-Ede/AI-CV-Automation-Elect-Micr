import numpy as np
import glob
import cv2

import arrayfire as af

from skimage.measure import compare_ssim as ssim
from scipy.misc import imread

from numba import cuda

#a = np.random.random((1,2))
#a = af.Array(a.ctypes.data, a.shape, a.dtype.char)
##af.display(b)

#for ii in af.ParallelRange(2):
#    a[:,ii] = ii*ii
#af.display(a)

#r = af.range(10,5)
#s = af.range(10,5)
#t = r + 2*complex(0, 1)*s
##a = af.Array(seq1)
#af.display(t)

#side = 10
#rec_square_dist = af.range(side, side, dim=0)*af.range(side, side, dim=0) + \
#    af.range(side, side, dim=1)*af.range(side, side, dim=1)
#af.display(rec_square_dist)

def mult_amp_phase(arr1, arr2):

    amp_mult = af.abs(arr1)*af.abs(arr2)
    phase_mult = af.atan(af.imag(arr1) / af.real(arr1)) * af.atan(af.imag(arr2) / af.real(arr2))

    return (amp_mult*af.cos(phase_mult) + complex(0, 1)*amp_mult*af.sin(phase_mult)).as_type(af.Dtype.c32)

def np_to_af(np_arr):
    return af.Array(np_arr.ctypes.data, np_arr.shape, np_arr.dtype.char)

def scale0to1(img):
    """Rescale image between 0 and 1"""

    min = np.min(img)
    max = np.max(img)

    if min == max:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def disp_fft_amp(fft, log_of_fft=True):

    amp = np.log(np.absolute(fft)) if log_of_fft else np.absolute(fft)

    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(amp))
    cv2.waitKey(0)

    return

def disp_fft_phase(fft):

    phase = np.angle(fft)

    #Wrap values from [0,2pi] to [-pi,pi]
    f = lambda x: x if x < np.pi else np.pi-x
    vecf = np.vectorize(f)
    phase = vecf(phase)

    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(phase))
    cv2.waitKey(0)

    return

def save_wavefunction(wavefunction):

    return

def save_results(wavefunction, params):
    #Will call save_wavefunction, among other things...



    return

def preprocess_stack(stack):

    return stack

def get_stack_from_tifs(dir):

    if dir[-1] != "/":
        dir += "/"

    files = glob.glob(dir+"*.tif")

    stack = []
    for file in files:
        img = imread(file, mode='F')

        stack.append(img)

    return preprocess_stack(stack)

def crop_from_stack(stack, rois):
    """Crop parts of images from the stack to compate to the reconstructed wavefunction"""

    for roi in rois:
        #Crop the roi from the img
        a = 1

    return cropped_stack

def stack_from_wavefunction():

    return reconstructed_stack

def stack_ssims(cropped_stack, reconstructed_stack):
    """ssim between the stacks"""

    stack_size = len(cropped_stack)
    ssims = []
    for i in range(stack_size):
        ssims.append(ssim(cropped_stack[i], reconstructed_stack[i]))

    return ssims

def crop_center(img, cropx, cropy):
    """Crop from center of numpy array"""

    y , x = img.shape
    startx = x//2 - (cropx//2)
    starty = y//2 - (cropy//2)    

    return img[starty:starty+cropy,startx:startx+cropx]

def crop_stack(stack, side, params):

    cropped_stack = []
    for i, image in enumerate(stack):
        cropped_stack.append( crop_center(image, side, side) )
    
    return cropped_stack

def params_from_cross_correlation(stack):
    """Initial estimate of parameters from phase correlation"""

    #Align the images using cross correlations
    rel_pos = []
    for i in range(len(stack)-1):
        rel_pos = cv2.phaseCorrelate(stack[i], stack[i+1])

    #Make relative positions relative to the center image to minimise error
    rel_pos_center = []
    rel0 = (0.0)
    for rel in rel_pos:
        rel = 1

    #Find the defocusses that best recreate the stack
    

    return stack

def calc_transfer_func(side, wavelength, defocus_change, px_dim = 1.0):
    
    ctf_coeff = np.pi * wavelength * defocus_change
    
    rec_px_width = 1.0 / (side*px_dim)
    rec_origin = -1.0 / (2.0*px_dim)

    rec_x_dist = rec_origin + rec_px_width * af.range(side, side, dim=0)
    rec_y_dist = rec_origin + rec_px_width * af.range(side, side, dim=1)

    ctf_phase = ctf_coeff*(rec_x_dist*rec_x_dist + rec_y_dist*rec_y_dist)
    ctf = af.cos(ctf_phase) + complex(0, 1) * af.sin(ctf_phase)

    return ctf.as_type(af.Dtype.c32)

def fft_to_diff(fft, side):

    diff = af.constant(0.0, side, side)
    for x in af.ParallelRange(side):
        for y in af.ParallelRange(side):
            diff[x, y] = disp_fft_amp[(x+side//2) % dim, (y+side//2) % dim]

    return diff

def diff_to_fft(diff, side):
    return fft_to_diff(disp_fft_amp, side)

def propagate_wave(img, ctf):

    fft = af.fft(img)
    ctf = diff_to_fft(ctf, fft.dims()[0])

    amp_phase_mult = mult_amp_phase(fft, ctf)
    propagation = af.ifft(amp_phase_mult)

    return propagation

def propagate_to_focus(img, defocus, params):

    ctf = calc_transfer_func(
        side=img.dims()[0],
        wavelength=params["wavelength"],
        defocus_change=-defocus)
    print("ctf calculated")
    return propagate_wave(img, ctf)

def propagate_back_to_defocus(img, defocus, params):
    
    ctf = calc_transfer_func(
        side=params["side"],
        wavelength=params["wavelength"],
        defocus_change=defocus)

    return propagate_wave(img, ctf)

def reconstruct(stack, params, num_iter = 50):
    """GPU accelerate wavefunction reconstruction and mse calculation"""

    width = stack[0].shape[0]+1
    height = stack[0].shape[1]+1
    stack_gpu = [np_to_af(img) for img in stack]

    exit_wave = af.constant(0, width, height)
    for i in range(num_iter):

        print("Iteration {0} of {1}".format(i, num_iter))

        for img, idx in zip(stack_gpu, range(len(stack))):

            print("Iteration {0} of {1}".format(idx, len(stack)))

            img = propagate_to_focus(img, params["defocus"][idx], params)

            exit_wave += img

        exit_wave /= len(stack)

        for img, idx in zip(stack_gpu, range(len(stack))):
            stack[idx] = propagate_back_to_defocus(exit_wave, params["defocus"][idx])
            stack[idx] = (af.abs(img) / af.abs(stack[idx])) * stack[idx]

    return exit_wave

def deconstruct(exit_wave, params):

    backpropagation = propagate_back_to_defocus()

    return deconstruction

def reconstruct_wavefunction(stack, side_to_use = None, params=None):
    """
    Reconstruct the wavefunction by finding the best parameters to do it
    stack - the stack to calculate the best wavefunction for
    side - length of sides of crop of images in stack to use
    """

    if not params["rel_pos"]:
        params["rel_pos"] = params_from_cross_correlation(stack)

    cropped_stack = crop_stack(stack, side_to_use, params)

    #Get reconstruction loss for stack as is
    #loss = reconstruction_loss(cropped_stack, params)

    reconstruction = reconstruct(cropped_stack, params)
    #deconstruction = deconstruct(reconstruction, params)

    return reconstruction, params

def minify_stack(stack, side):

    mini_stack = []
    for img in stack:
        mini_stack.append(cv2.resize(img, (side, side)))

    return mini_stack

if __name__ == "__main__":

    dir = "E:/dump/stack1/"
    mini_side = 256
    prop_side_to_use = 0.9

    stack = get_stack_from_tifs(dir)
    stack = stack[0:2]

    params = { "rel_pos": [0.0 for _ in stack],
               "defocus": [1, 2, 3, 4],
               "wavelength": 2.07e-15}

    mini_stack = minify_stack(stack, mini_side)
    mini_reconstruct, mini_params = reconstruct_wavefunction(
        stack=mini_stack, 
        side_to_use=mini_side,
        params=params)

    side = int(np.log2(stack[0].shape[0]))
    side **= 2
    unminification_factor = side / mini_side

    start_params = params
    start_params["rel_pos"] = [unminification_factor*rel_pos for rel_pos in params["params"]]

    reconstuct, params = reconstruct_wavefunction(
        stack=stack, 
        side_to_use=int(prop_side_to_use*side), 
        start_params=start_params)

