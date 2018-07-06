import numpy as np

def denoise(img, preprocess=True, postprocess=True, overlap=80):

    if preprocess:
        img = self.preprocess(img)

    denoised = np.zeros(img.shape)
    contributions = np.zeros(img.shape)
    
    num0 = img.shape[0]//(512-overlap)+1
    num1 = img.shape[1]//(512-overlap)+1
    len0 = img.shape[0]/num0
    len1 = img.shape[1]/num1

    for i in range(num0):
        x = np.round(i*len0)
        for j in range(num1):
            y = np.round(j*len1)

            denoised[x:(x+512), y:(y+512)] = self.denoise_crop(
                img[x:(x+512), y:(y+512)], 
                preprocess=False, 
                postprocess=False).reshape((512,512))
            contributions[x:(x+512), y:(y+512)] += 1

    denoised /= contributions

    if postprocess:
        return denoised.clip(0., 1.)
    else:
        return denoised
