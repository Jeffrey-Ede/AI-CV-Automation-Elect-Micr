import numpy as np
import cv2

from scipy.misc import imread

def random_polar_transform(x):
    return

def polar2cart(r, theta, center):
    x = r * np.cos(theta) + center[0]
    y = r * np.sin(theta) + center[1]
    return x, y

def img2polar(img, center=None, initial_radius=None, final_radius=None, phase_width=None):

    if center is None:
        center = (img.shape[0]/2, img.shape[1]/2)

    if initial_radius is None:
        initial_radius = 0

    if final_radius is None:
        final_radius = int((min(img.shape)-1)/2)

    if phase_width is None:
        phase_width = min(img.shape)

    theta , R = np.meshgrid(np.linspace(0, 2*np.pi, phase_width), 
                            np.sqrt(np.linspace(initial_radius**2, final_radius**2, phase_width)))

    Xcart, Ycart = polar2cart(R, theta, center)

    Xcart = Xcart.astype(int)
    Ycart = Ycart.astype(int)

    polar_img = img[Ycart,Xcart]
    print(polar_img.shape)

    return polar_img

def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def disp(img):
    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(img))
    cv2.waitKey(0)
    return

if __name__ == "__main__":

    loc = "Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-1/truth-4000.tif"
    img = imread(loc, 'F')

    polar_img = img2polar(img)
    print(polar_img.shape)
    disp(polar_img)