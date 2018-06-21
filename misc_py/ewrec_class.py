import numpy as np
import glob
import cv2

import arrayfire as af

from skimage.measure import compare_ssim as ssim
from scipy.misc import imread
from scipy.optimize import minimize

import re
import pprint

class Utility(object):

    def __init__(self):
        pass

    @staticmethod
    def np_to_af(np_arr, dtype=af.Dtype.f32):
        return af.Array(np_arr.ctypes.data, np_arr.shape, np_arr.dtype.char).as_type(dtype)

    @staticmethod
    def fft_shift(fft, dtype=af.Dtype.c32):
        return af.shift(fft, fft.dims()[0]//2 + fft.dims()[0]%2, 
                        fft.dims()[1]//2 + fft.dims()[1]%2).as_type(dtype)

    @staticmethod
    def scale0to1(img):
        """Rescale image between 0 and 1"""
        min = np.min(img)
        max = np.max(img)

        if min == max:
            img.fill(0.5)
        else:
            img = (img-min) / (max-min)

        return img.astype(np.float32)

    def disp(self, arr):
        cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
        cv2.imshow('CV_Window', self.scale0to1(arr))
        cv2.waitKey(0)
        return

    def disp_af(self, arr):
        cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
        cv2.imshow('CV_Window', self.scale0to1(arr.__array__()))
        cv2.waitKey(0)
        return

    @staticmethod
    def af_phase(img):
        phase = np.angle(img)
        f = lambda x: x if x < np.pi else np.pi-x
        vecf = np.vectorize(f)
        return vecf(phase)

    def disp_af_complex_amp(self, fft, log_of_fft=True):
        amp = np.log(np.absolute(fft)) if log_of_fft else np.absolute(fft)

        cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
        cv2.imshow('CV_Window', self.scale0to1(amp))
        cv2.waitKey(0)

        return

    def disp_af_complex_phase(self, fft):
        
        cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
        cv2.imshow('CV_Window', self.scale0to1(self.af_phase(fft.__array__())))
        cv2.waitKey(0)

        return

    @staticmethod
    def save_af_as_npy(arr, filename, save_loc=""):

        if filename[-4:-1] != ".npy":
            filename += ".npy"
        if save_loc[-1] != "/":
            save_loc += "/"

        np.save(save_loc+filename+".npy", arr.__array__())

        return

    @staticmethod
    def get_radial_freq_hist(img, mean=1.0): #Currently unused
        
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

    @staticmethod
    def af_padded_fft2(img, pad_val=0., pad_periods=1):
        side = img.dims()[0]
        pad_side = int((1+pad_periods)*side)
        padded_img = af.constant(0., pad_side, pad_side, dtype=af.Dtype.c32)
        padded_img[:side, :side] = img
        return af.fft2(padded_img)

    @staticmethod
    def af_unpadded_ifft2(fft, pad_periods=1):
        side = fft.dims()[0] // (1+pad_periods)
        return af.ifft2(fft)[:side, :side]

    def af_phase_corr(self, img1, img2, gauss_blur_size=3, small_const=1.e-6):
        prod = af.fft2(img1) * af.conjg(af.fft2(img2))
        phase_corr = self.fft_shift(af.abs(
            af.ifft2( prod / (af.abs(prod)+small_const) )), af.Dtype.f32)
        
        if gauss_blur_size:
            phase_corr = af.convolve(phase_corr, 
                                     af.gaussian_kernel(gauss_blur_size, gauss_blur_size))
        return phase_corr

    @staticmethod
    def get_centroid(img):
        moments_x = moments_y = sum = 0.
        for i in range(img.shape[1]):
            for j in range(img.shape[0]):
                moments_x += i*img[j, i]
                moments_y += j*img[j, i]
                sum += img[j, i]
        return moments_x/sum, moments_y/sum

    @staticmethod
    def transparent_overlay(img1, img2):
        return cv2.addWeighted(img1, 0.5, img2, 0.5, 0)

    def align_by_eye(self, img1, img2): #This is not practical
        img1 = np.dstack((self.scale0to1(img1), np.zeros(img1.shape), np.zeros(img1.shape)))
        img2 = np.dstack((np.zeros(img2.shape), np.zeros(img2.shape), self.scale0to1(img2)))

        shift = [0, 0]
        while True:
            self.disp(self.transparent_overlay(img1, img2))
            io = input()
            if io is "b":
                break
            
            if io is "q": #shift left
                shift[0] -= 1
            if io is "w": #shift up
                shift[1] += 1
            if io is "e": #shift right
                shift[0] += 1
            if io is "s": #shift down
                shift[1] -= 1

        return shift

###########################################################################################

class EWREC(Utility):
    
    def __init__(self, 
                 stack_dir, 
                 wavelength, 
                 rel_pos=None,
                 rel_pos_method="ccoeff",
                 max_rel_pos_prop=0.05, #Maximum proportion of image length that image can be translated from its neighbours
                 series_type = "cubic",
                 series_alternating=True, #Focuses alternating about in-focus position
                 series_middle=None, #Middle focus index only needed if series alternates about centre. If not
                                       #provided, the halfway index will be chosen
                 series_increasing=True,
                 defocuses=None,
                 px_dim=1.,
                 defocus_search_range=[45.e-9, 50.e-9],
                 reconstruction_side=None,
                 num_iter=50,
                 defocus_sweep_num=10, #Number of defocuses to try initial sweep to find the defocus
                 defocus_search_criteria=["gradient_plateau"],
                 preprocess_fn=None,
                 param_refinement=False, #Time-consuming refinement of relative positions and defocus values
                 nn_alignment=False, #TODO
                 nn_refinement=False, #TODO
                 report_progress=True,
                 pad_periods=1, #Number of periods of signal to pad it by for fft
                 pad_val=0.,
                 refinement_max_iter=20,
                 refinement_tol=1.e-3):

        self.init_logging = report_progress
        self.display_iter_nums = report_progress
        self.updates_during_refinement = report_progress
        self.log_defocus_initial_scan_completion = report_progress
        self.log_reconstruction_loss = report_progress

        self.num_iter=num_iter
        self.stack_dir = stack_dir
            
        self.stack = self.get_stack_from_tifs(self.stack_dir)
        if self.init_logging: print("Got stack from tifs")
        if preprocess_fn:
            self.stack = preprocess_fn(self.stack)
        self.stack_side = self.stack[0].shape[0]

        #self.stack = self.stack[2:7]
        #series_middle -= 2


        self.pad_periods = pad_periods
        self.pad_value = pad_val

        #Focal series meta
        self.wavelength = wavelength
        self.series_type = series_type
        self.series_alternating = series_alternating
        self.series_mid = series_middle
        self.series_increasing = series_increasing
        self.px_dim = px_dim

        self.reconstruction_side = reconstruction_side

        self.refinement_side = 5
        if not rel_pos:
            if rel_pos_method:
                self.max_rel_pos_prop = max_rel_pos_prop
                self.rel_pos = rel_pos if rel_pos else self.rel_pos_estimate(rel_pos_method, 
                                                                             rel_to_top_left=True)        
                if self.init_logging: print("Got relative positions")
                for rel in self.rel_pos: 
                    print(rel)

                self.stack = self.crop_stack(self.rel_pos)
            else:
                self.stack = self.correct_cropped_stack_size(self.stack)

        print("Starting defocus calculation")
        if defocuses:
            self.defocuses = defocuses
        else:
            self.defocus_initial_sweep_num = defocus_sweep_num
            self.defocus_search_criteria = defocus_search_criteria
            self.defocus_search_range = defocus_search_range
            self.defocuses = self.defocus_initial_estimate()
            if self.init_logging: 
                print("Got initial defocus estimate:")
                pprint.pprint(self.defocuses)

        self.refinement_max_iter = refinement_max_iter
        self.refinement_tol = refinement_tol
        if param_refinement:
            self.defocuses = self.refine_params(self.refinement_max_iter, self.refinement_tol)
            if self.init_logging:
                print("Refined parameters")

        print(self.stack[0].shape)
        self.exit_wave = self.reconstruct(self.stack,
                                          self.defocuses,
                                          num_iter=self.num_iter)

        print(self.af_phase(self.exit_wave))
        self.disp_af_complex_phase(self.exit_wave)

    @staticmethod
    def get_stack_from_tifs(dir):

        if dir[-1] != "/":
            dir += "/"

        files = glob.glob(dir+"*.tif")

        #Get image numbers from filenames
        nums = [[int(float(s)) for s in re.findall(r'-?\d+\.?\d*', file)][-1] for file in files]

        #Sort files into the correct order
        files = [file for _, file in sorted(zip(nums, files))] 

        stack = [imread(file, mode='F') for file in files]

        return stack

    def crop_stack(self, centres, side=None, resize_crop_for_reconstruction=True):
        """Crop parts of images from a stack"""
        if not side:
            #Calculate largest possible crop side
            min_dist = 2**32-1 #A large number
            stack_side = self.stack_side
            for centre in centres:
                min_from_side = np.min([centre[0], centre[1], 
                                        stack_side-centre[0],
                                        stack_side-centre[1]])
                if min_from_side < min_dist:
                    min_dist = min_from_side
            side = int(2*min_dist)

        side_pow2 = int(np.log2(side))**2
        
        crops = []
        for img in self.stack:
            left = int(centres[0][0]-side/2)
            bottom = int(centres[0][1]-side/2)

            horiz_over = centres[0][0]-int(centres[0][0])
            vert_over = centres[0][1]-int(centres[0][1])
            prop_tl = horiz_over*vert_over
            prop_tr = (1.-horiz_over)*vert_over
            prop_bl = horiz_over*(1.-vert_over)
            prop_br = (1.-horiz_over)*(1.-vert_over)
            
            crop = np.zeros((side, side))
            crop = (prop_tl*img[(bottom+1):(bottom+side+1), left:(left+side)]+
                    prop_tr*img[(bottom+1):(bottom+side+1), (left+1):(left+side+1)]+
                    prop_bl*img[bottom:(bottom+side), left:(left+side)]+
                    prop_br*img[bottom:(bottom+side), (left+1):(left+side+1)])

            crops.append(crop)

        if resize_crop_for_reconstruction:
            return self.correct_cropped_stack_size(crops)
        else:
            return crops

    def correct_cropped_stack_size(self, stack):
        """Resize cropped stack so that its side lengths are a power of 2"""
        crop_side = min(2**int(np.log2(stack[0].shape[0])), 
                        2**int(np.log2(self.reconstruction_side)))
        cropped_stack = self.resize_stack(stack, crop_side)
        return cropped_stack

    @staticmethod
    def resize_stack(stack, side):
        return [cv2.resize(img, (side, side)) for img in stack]

    def rel_pos_estimate(self, method="phase_corr", stack=None, rel_to_top_left=True):
        if not stack:
            stack = self.stack

        side = stack[0].shape[0]
        side_pow2 = 2**int(np.log2(side))
        scaling_ratio = side / side_pow2 
        stack_gpu = [self.np_to_af(img) for img in self.resize_stack(stack, side_pow2)]

        rel_pos = [(0.,0.)]

        if method == "phase_corr":
            half_roi_side = int(self.max_rel_pos_prop * side_pow2)
            roi_side = 2*half_roi_side
            min_idx = side//2 - half_roi_side
            max_idx = side//2 + half_roi_side

            for i in range(1, len(self.stack)):
                phase_corr = self.af_phase_corr(stack_gpu[i-1], stack_gpu[i], 3)[
                    min_idx:max_idx,min_idx:max_idx].__array__()

                #Refine the maximum to sub-px accuracy
                max_pos = (np.argmax(phase_corr)%roi_side, np.argmax(phase_corr)//roi_side)
                search_size = min(max_pos[0], max_pos[1], roi_side-max_pos[0],
                                  roi_side-max_pos[1], self.refinement_side//2)
      
                centroid = self.get_centroid(phase_corr[(max_pos[1]-search_size):(max_pos[1]+search_size+1),
                                                        (max_pos[0]-search_size):(max_pos[0]+search_size+1)])
                x = scaling_ratio*(max_pos[0]+centroid[0]-search_size//2)
                y = scaling_ratio*(max_pos[1]+centroid[1]-search_size//2)
                rel_pos.append((x, y))

        if method == "ccoeff":
            half_roi_side = int(side_pow2 // 20)
            roi_side = 2*half_roi_side
            min_idx = side//2 - half_roi_side
            max_idx = side//2 + half_roi_side

            half_kernel_side = side_pow2 // 4
            half_side_pow2 = side_pow2 // 2
            for i in range(1, len(self.stack)):
                kernel = stack[i-1][(half_side_pow2-half_kernel_side):(half_side_pow2+half_kernel_side),
                                        (half_side_pow2-half_kernel_side):(half_side_pow2+half_kernel_side)]
                match = cv2.matchTemplate(stack[i], kernel, cv2.TM_CCOEFF)
                max_loc = np.argmax(match[min_idx:max_idx,min_idx:max_idx])

                max_pos = (max_loc%roi_side, max_loc//roi_side)
                #search_size = min(max_pos[0], max_pos[1], roi_side-max_pos[0],
                #                  roi_side-max_pos[1], self.refinement_side//2)
      
                #centroid = self.get_centroid(match[(max_pos[1]-search_size):(max_pos[1]+search_size+1),
                #                                        (max_pos[0]-search_size):(max_pos[0]+search_size+1)])
                #x = scaling_ratio*(max_pos[0]+centroid[0]-search_size//2)
                #y = scaling_ratio*(max_pos[1]+centroid[1]-search_size//2)
                rel_pos.append(max_pos)

                #self.disp(match)

        if rel_to_top_left:
            #chain relative positions from the centermost and find the position closest to the mean
            pos = np.zeros((len(rel_pos), 2))
            for i in range(1, len(rel_pos)):
                pos[i][0] = pos[i-1][0]+rel_pos[i][0]
                pos[i][1] = pos[i-1][1]+rel_pos[i][1]
            
            mean = [0., 0.]
            for i in range(len(pos)):
                mean[0] += pos[i][0]
                mean[1] += pos[i][1]
            mean[0] /= len(pos)
            mean[1] /= len(pos)

            dists = [(x-mean[0])**2+(y-mean[1])**2 for x, y in pos]
            idx = dists.index(min(dists))
            
            half_side = self.stack_side/2
            
            return [(half_side+mean[0]-x, half_side+mean[1]-y) for x, y in pos]
        else:
            return rel_pos

    @staticmethod
    def calc_transfer_func(side, wavelength, defocus_change, pad_periods=0, 
                            px_dim=1., spher_aber_coeff=None, aperture_mask=None):
        size = int((1.+pad_periods)*side)
        px_dim = (1.+pad_periods)*px_dim

        ctf_coeff = np.pi * wavelength * defocus_change
    
        rec_px_width = 1.0 / (side*px_dim)
        rec_origin = -1.0 / (2.0*px_dim)

        rec_x_dist = rec_origin + rec_px_width * af.range(size, size, dim=0)
        rec_y_dist = rec_origin + rec_px_width * af.range(size, size, dim=1)
        rec_dist2 = rec_x_dist*rec_x_dist + rec_y_dist*rec_y_dist

        ctf_phase = ctf_coeff*rec_dist2

        if spher_aber_coeff:
            ctf_phase += 0.5 * np.pi * wavelength**3 * spher_aber_coeff * rec_dist2**2

        ctf = af.cos(ctf_phase) + complex(0, 1)*af.sin(ctf_phase)

        if aperture_mask:
            ctf *= aperture_mask

        return ctf.as_type(af.Dtype.c32)

    def fft_to_diff(self, fft):
        return self.fft_shift(fft)

    def diff_to_fft(self, diff):
        return self.fft_to_diff(diff)

    def propagate_wave(self, img, ctf):
        fft = self.af_padded_fft2(img, self.pad_value, self.pad_periods)
        ctf = self.diff_to_fft(ctf)
        propagation = self.af_unpadded_ifft2(fft*ctf, self.pad_periods)
        return propagation

    def propagate_to_focus(self, img, defocus, wavelength, 
                           pad_periods=0, px_dim=1.):
        ctf = self.calc_transfer_func(
            side=int(img.dims()[0]),
            wavelength=wavelength,
            defocus_change=-defocus,
            pad_periods=pad_periods,
            px_dim=px_dim)
        return self.propagate_wave(img, ctf)

    def propagate_back_to_defocus(self, exit_wave, defocus, wavelength, 
                                  pad_periods=0, px_dim=1.):
        ctf = self.calc_transfer_func(
            side=int(exit_wave.dims()[0]),
            wavelength=wavelength,
            defocus_change=defocus,
            pad_periods=pad_periods)
        return self.propagate_wave(exit_wave, ctf)

    def reconstruct(self, stack, defocuses=None, num_iter = 50, stack_on_gpu=False):
        """GPU accelerate wavefunction reconstruction and mse calculation"""

        stack_gpu = stack if stack_on_gpu else [self.np_to_af(img, af.Dtype.c32) for img in stack]
        defocuses = defocuses if defocuses else self.defocuses

        width = stack[0].shape[0]
        height = stack[0].shape[1]
        exit_wave = af.constant(0, width, height)
        for i in range(num_iter):

            if self.display_iter_nums:
                print("Iteration {0} of {1}".format(i+1, num_iter))

            exit_wave = 0
            for img, idx in zip(stack_gpu, range(len(stack_gpu))):

                #print("Propagation {0} of {1}".format(idx+1, len(stack)))
                exit_wave += self.propagate_to_focus(img, defocuses[idx], 
                                                     self.wavelength, 
                                                     self.pad_periods,
                                                     self.px_dim)
            exit_wave /= len(stack)

            for idx in range(len(stack)):
                amp = af.abs(stack_gpu[idx])
                stack_gpu[idx] = self.propagate_back_to_defocus(exit_wave, defocuses[idx], 
                                                                self.wavelength, 
                                                                self.pad_periods,
                                                                self.px_dim)
                stack_gpu[idx] = (amp / af.abs(stack_gpu[idx])) * stack_gpu[idx]

        return exit_wave

    def reconstruction_loss(self, stack_gpu, defocus_incr, defocus_ramp):
        defocuses = [defocus_incr*ramp for ramp in defocus_ramp]
        reconstruction = self.reconstruct(stack_gpu.copy(), 
                                          defocuses,
                                          num_iter=self.num_iter,
                                          stack_on_gpu=True)

        #Use the wavefunction to recreate the original images
        deconstruction = []
        for i, defocus in enumerate(defocuses):
            print(defocus)
            _deconstruction = af.abs(self.propagate_back_to_defocus(
                                                            reconstruction, 
                                                            defocus, 
                                                            self.wavelength, 
                                                            self.pad_periods,
                                                            self.px_dim))
            _deconstruction = (af.mean(af.abs(stack_gpu[i])) / 
                               af.mean(af.abs(_deconstruction))) * _deconstruction
            deconstruction.append(_deconstruction)
        losses = [0.]*len(stack_gpu)

        for i in range(len(losses)):
            losses[i] = np.mean((np.sqrt(self.stack[i])-af.abs(deconstruction[i]).__array__())**2)
            collapse = af.abs(deconstruction[i])
            #print(self.af_phase(deconstruction[i]))
            #collapse /= af.mean(collapse)
            #losses[i] = np.sqrt(af.mean(
            #    (af.abs(stack_gpu[i])/af.mean(af.abs(stack_gpu[i])) - collapse)**2))
            #self.disp_af(collapse)
        loss = np.mean(losses)

        if self.log_reconstruction_loss:
            print("Reconstruction loss: {}".format(loss))

        return loss

    def defocus_initial_estimate(self):
        #Try various defocuses until one is found that matches the expected pattern

        stack = self.stack

        if self.series_type == "linear":
            gen = lambda x: x
        elif self.series_type == "quadratic":
            gen = lambda x: x**2
        elif self.series_type == "cubic":
            gen = lambda x: x**3

        mid = (self.series_mid if self.series_mid else len(stack) // 2) if self.series_alternating else 0
        defocus_dir = 1.0 if self.series_increasing else -1.0

        side = stack[0].shape[0]
        stack_gpu = [self.np_to_af(img, af.Dtype.c32) for img in stack]
    
        search_ramp = [((2**x-1) / 2**(self.defocus_initial_sweep_num-1)) for x in range(self.defocus_initial_sweep_num)]
        m = self.defocus_search_range[1]-self.defocus_search_range[0]
        c = self.defocus_search_range[0]
        defocus_incr = [m*x+c for x in search_ramp]
        defocus_ramp = [defocus_dir*np.sign(x-mid)*gen(x-mid) for x in range(len(stack_gpu))]
        losses = [self.reconstruction_loss(stack_gpu, incr, defocus_ramp) for incr in defocus_incr]

        #Get the highest loss neigbouring the highest and refine using bilinear interpolation
        idx = losses.index(max(losses))
        if idx == 0:
            idx1, idx2 = idx, 0
        elif idx == self.defocus_initial_sweep_num-1:
            idx1, idx2 = 0, idx
        else:
            idx1, idx2 = (idx, idx+1) if losses[idx-1] < losses[idx+1] else (idx-1, idx)

        losses = [losses[idx]]
        incr1 = defocus_incr[idx1]
        incr2 = defocus_incr[idx2]

        if self.log_defocus_initial_scan_completion:
            print("Initial defocus scan completed")

        condition_fn=None
        def condition(losses):
            if len(losses) == 1:
                return True
            else:
                return losses[-1] < losses[-2]
        if self.defocus_search_criteria[0] == "gradient_plateau":
            condition_fn = condition

        while True:
            incr = 0.5*(incr1+incr2)
            losses.append(self.reconstruction_loss(stack_gpu.copy(), incr, defocus_ramp))

            if condition_fn(losses):
                incr1, incr2 = incr2, incr
            else:
                return [incr2*defocus for defocus in defocus_ramp]
        
    def reconstruction_loss_arbitrary_params(self, centres, defocuses):

        stack = self.crop_stack(centres)
        stack_gpu = [self.np_to_af(img, af.Dtype.c32) for img in stack]
        reconstruction = reconstruct(stack_gpu.copy(),
                                     defocuses, 
                                     num_iter=self.num_iter, 
                                     stack_on_gpu=True)

        losses = [0.]*len(stack_gpu)
        for i in range(len(losses)):
            collapse = af.abs(deconstruction[i])**2
            collapse *= af.mean(stack_gpu[i]) / af.mean(collapse)
            
            losses[i] = af.mean((stack_gpu[i]-collapse)**2)

        return np.max(losses)

    def refine_params(self, max_iter=20, tol=1.e-3):

        if self.updates_during_refinement:
            update = 1

        def loss(x):
            if self.updates_during_refinement:
                print("Refinement {}".format(update))
                i += update
            return self.reconstruction_loss_arbitrary_params(self.rel_pos, x)

        x0 = self.defocuses
        refinement = minimize(
            loss,
            x0,
            method='trust-krylov',
            tol=tol,
            iter=max_iter)
        
        x = refinement.x
        return x

if __name__ == "__main__":

    ewrec = EWREC(
        stack_dir=r'\\flexo.ads.warwick.ac.uk\Shared39\EOL2100\2100\Users\Jeffrey-Ede\series72\stack1',
        wavelength=2.51e-12,
        series_type="quadratic",
        rel_pos_method=None,
        series_middle=8,
        num_iter=50,
        defocus_search_range=[100.e-9, 110.e-9],
        defocus_sweep_num=2, #Number of sweeps at start
        px_dim=1.e-5,
        series_increasing=True,
        reconstruction_side=512,
        param_refinement=False,
        pad_periods=0)