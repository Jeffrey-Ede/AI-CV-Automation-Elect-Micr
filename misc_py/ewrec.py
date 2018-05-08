import numpy as np
import glob
import cv2

import arrayfire as af

from skimage.measure import compare_ssim as ssim
from scipy.misc import imread

def np_to_af(np_arr, dtype=af.Dtype.f32):
    return af.Array(np_arr.ctypes.data, np_arr.shape, np_arr.dtype.char).as_type(dtype)

def fft_shift(fft):
    return af.shift(fft, fft.dims()[0]//2 + fft.dims()[0]%2, fft.dims()[1]//2 + fft.dims()[1]%2)

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

def disp_af(arr):

    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(arr.__array__()))
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

    y, x = img.shape
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

    rel_pos = []
    for i in range(len(stack)-1):
        rel_pos = cv2.phaseCorrelate(stack[i], stack[i+1])

    #Make relative positions relative to the center image to minimise error
    rel_pos_center = []
    rel0 = (0.0)
    for rel in rel_pos:
        rel = 1

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

def fft_to_diff(fft, side=None):
    return fft_shift(fft)

def diff_to_fft(diff, side=None):
    return fft_to_diff(diff, side)

def propagate_wave(img, ctf):

    fft = af.fft2(img)

    ctf = diff_to_fft(ctf, fft.dims()[0])

    amp_phase_mult = fft*ctf
    propagation = af.ifft2(amp_phase_mult)

    return propagation

def propagate_to_focus(img, defocus, params):

    ctf = calc_transfer_func(
        side=img.dims()[0],
        wavelength=params["wavelength"],
        defocus_change=-defocus)

    return propagate_wave(img, ctf)

def propagate_back_to_defocus(img, defocus, params):
    
    ctf = calc_transfer_func(
        side=img.dims()[0],
        wavelength=params["wavelength"],
        defocus_change=defocus)

    return propagate_wave(img, ctf)

def reconstruct(stack, params, num_iter = 50, stack_on_gpu=False):
    """GPU accelerate wavefunction reconstruction and mse calculation"""

    width = stack[0].shape[0]
    height = stack[0].shape[1]

    stack_gpu = stack if stack_on_gpu else [np_to_af(img) for img in stack]

    exit_wave = af.constant(0, width, height)
    for i in range(num_iter):

        #print("Iteration {0} of {1}".format(i+1, num_iter))
        exit_wave = 0
        for img, idx in zip(stack_gpu, range(len(stack_gpu))):

            #print("Propagation {0} of {1}".format(idx+1, len(stack)))
            exit_wave += propagate_to_focus(img, params["defocus"][idx], params)

        exit_wave /= len(stack)

        for idx in range(len(stack)):
            amp = af.abs(stack_gpu[idx])
            stack_gpu[idx] = propagate_back_to_defocus(exit_wave, params["defocus"][idx], params)
            stack_gpu[idx] = (amp / af.abs(stack_gpu[idx])) * stack_gpu[idx]

    return exit_wave

#def deconstruct(exit_wave, stack, params):
    
#    deconstruction = []
#    for i, defocus in enumerate(params["defocus"]):
#        abs2 = af.abs(propagate_back_to_defocus(exit_wave, defocus, params))**2
#        backpropagation = af.mean(stack[i])*abs2 / af.mean(abs2)
#        deconstruction.append( backpropagation )

#    return deconstruction


def get_radial_freq_hist(img, mean=1.0):

    abs_shifted_fft = np.abs(np.fft.fftshift(np.fft.fft2(img)))

    rows = cols = int(abs_shifted_fft.shape[0])
    mid_row = mid_col = int(np.ceil(abs_shifted_fft.shape[0]/2))
    max_rad = int(np.ceil(np.sqrt((mid_row)**2 + (mid_col)**2)))+1
    radial_profile = np.zeros(max_rad)

    for col in range(cols):
        for row in range(rows):
            radius = np.sqrt((row-mid_row+1)**2 + (col-mid_col+1)**2)
            idx = int(np.ceil(radius))

            radial_profile[idx] += abs_shifted_fft[col][row]

    return radial_profile

#Function currently unused. Needs more robust method. Maybe variance of laplacian?
def get_defocus_seq_type(stack, prop_top_freq_to_use=0.05): 
    """
    Assume that the higher frequency images have a larger high frequency component to
    determine the sequence type
    """

    top_prop_freq_sums = []
    for img in stack:
        radial_freq = get_radial_freq_hist(img)
        top_prop_freq_sums.append(np.sum(radial_freq[int((1.0-prop_top_freq_to_use)*len(radial_freq)):])/np.sum(radial_freq))

    for s in top_prop_freq_sums:
        print(s)

    #left middle right
    return "middle"

def defocus_initial_estimate(stack, params):
    #Try various defocuses until one is found that matches the expected pattern

    if params["defocus_seq_type"] == "linear":
        gen = lambda x: x
    elif params["defocus_seq_type"] == "quadratic":
        gen = lambda x: x**2
    elif params["defocus_seq_type"] == "cubic":
        gen = lambda x: x**3

    mid = params["defocus_seq_start"]
    defocus_dir = 1.0 if params["forward_seq"] else -1.0

    side = stack[0].shape[0]
    stack_gpu = [np_to_af(img, af.Dtype.c32) for img in stack]
    
    losses = []
    for incr in [int(1e13)]:
        params["defocus"] = [defocus_dir*np.sign(x-mid)*(incr)*gen(x-mid) for x in range(len(stack_gpu))]

        ##Split stack in 2 and compare reconstructions
        #params1 = dict(params)
        #params2 = dict(params)
        #params1["defocus"] = [x for i, x in enumerate(params["defocus"]) if i%2]
        #params2["defocus"] = [x for i, x in enumerate(params["defocus"]) if not i%2]

        #stack_gpu1 = [x for i, x in enumerate(stack_gpu) if i%2]
        #stack_gpu2 = [x for i, x in enumerate(stack_gpu) if not i%2]

        #rec1 = reconstruct(stack_gpu1.copy(), params1, stack_on_gpu=True)
        #rec2 = reconstruct(stack_gpu2.copy(), params2, stack_on_gpu=True)

        #abs1 = af.abs(rec1)**2
        #abs2 = af.abs(rec2)**2
        #abs2 *= af.mean(abs1) / af.mean(abs2)
        #loss = af.mean((abs1-abs2)**2)

        #print("{:.2E}".format(loss))
        
        #losses.append(loss)

        #disp_af(af.abs(rec2))
        #disp_fft_phase(rec1)
        #disp_fft_phase(rec2)

        Image.fromarray
        disp_fft_phase(reconstruct(stack_gpu.copy(), params, stack_on_gpu=True))

    print(losses)

    return defocuses

def reconstruct_wavefunction(stack, side_to_use = None, params=None):
    """
    Reconstruct the wavefunction by finding the best parameters to do it
    stack - the stack to calculate the best wavefunction for
    side - length of sides of crop of images in stack to use
    """

    if not params["rel_pos"]:
        params["rel_pos"] = params_from_cross_correlation(stack)
    #if not params["defocus_seq_type"]:
    #    params["defocus_seq_type"] = get_defocus_seq_type(stack)

    cropped_stack = crop_stack(stack, side_to_use, params)

    if not params["defocus"]:
        params["defocus"] = defocus_initial_estimate(cropped_stack, params)

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
    mini_side = 512
    prop_side_to_use = 0.9

    stack = get_stack_from_tifs(dir)
    print("Stack loaded")
    #stack = stack[0:1]

    params = { "rel_pos": [0.0 for _ in stack],
               "defocus_seq_start": 6, #index at the lowest defocus
               "defocus_seq_type": "quadratic", #"linear", "quadratic" or "cubic" 
               "forward_seq": True,
               "defocus": None,
               "wavelength": 2.51e-12}

    mini_stack = minify_stack(stack, mini_side)
    mini_reconstruct, mini_params = reconstruct_wavefunction(
        stack=mini_stack, 
        side_to_use=mini_side,
        params=params)

    side = int(np.log2(stack[0].shape[0]))**2
    unminification_factor = side / mini_side

    start_params = params
    start_params["rel_pos"] = [unminification_factor*rel_pos for rel_pos in params["rel_pos"]]

    reconstuct, params = reconstruct_wavefunction(
        stack=stack,
        side_to_use=int(prop_side_to_use*side), 
        start_params=start_params)
