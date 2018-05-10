import PIL
import numpy as np
import cv2
from io import BytesIO

test_dir = "D:/data/stack1/"
test_img = test_dir + "img1.tif"
save_img = "D:/dump/test.jpeg"

def encode_jpeg(img_np, quality):

    img = PIL.Image.fromarray(img_np).convert('RGB')
    destination = BytesIO()
    try:
        img.save(destination, format='jpeg', quality=quality, optimize=False, progressive=True)
    except IOError:
        PIL.ImageFile.MAXBLOCK = img.size[0] * img.size[1]
        img.save(destination, 'jpeg', quality=quality, optimize=True, progressive=True)

    destination.seek(0)
    return destination

def decode_jpeg(encoding):
    return np.array(PIL.Image.open(encoding))

def jpeg(img, quality=95):
    return decode_jpeg(encode_jpeg(img, quality))

if __name__ == "__main__":

    from scipy.misc import imread

    def scale0to1(img):
        """Rescale image between 0 and 1"""
        min = np.min(img)
        max = np.max(img)

        if min == max:
            img.fill(0.5)
        else:
            img = (img-min) / (max-min)

        return img.astype(np.float32)

    img = imread(test_img, mode='F')

    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(jpeg(img)))
    cv2.waitKey(0)
