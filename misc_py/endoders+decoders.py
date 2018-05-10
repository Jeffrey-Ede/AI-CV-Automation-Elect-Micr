import PIL
import numpy as np
import cv2
from io import StringIO

def encode_jpeg(img_np, quality):

    img = PIL.Image.fromarray(img_np).convert('RGB')
    destination = StringIO()
    try:
        img.save(destination, format="JPEG", quality=quality, optimize=False, progressive=True)
    except IOError:
        PIL.ImageFile.MAXBLOCK = img.size[0] * img.size[1]
        img.save(destination, "JPEG", quality=quality, optimize=True, progressive=True)

    encoding = destination.getvalue()
    destination.close()
    return encoding

def decode_jpeg(encoding):
    return np.array(PIL.Image.open(encoding))

def jpeg(img, quality=0.95):
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

    img = imread("C:/dump/noise1.tif", mode='F')

    cv2.namedWindow('CV_Window', cv2.WINDOW_NORMAL)
    cv2.imshow('CV_Window', scale0to1(jpeg(img)))
    cv2.waitKey(0)