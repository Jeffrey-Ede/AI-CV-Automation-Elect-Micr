import numpy as np
from pycroscopy.io.translators.df_utils.dm_utils import read_dm3 #There is also read_dm4

"""
This short script shows how to read dm3 or dm4 files with pycroscopy.
Pycroscopy can be installed with pip e.g. pip install pycroscopy.
Author: Jeffrey Ede (j.m.ede@warwick.ac.uk)
"""

#Replace this with your dm3 or dm4 file
my_dmx_file = "C:/dump/img1.dm3"

#Use pycroscopy to load digital micrograph file
dmx_img, dmx_metadata = read_dm3(my_dmx_file)
print("Digital Micrograph image:\n", dmx_img) #Image is numpy array
print("Metadata:\n", dmx_metadata) #Metadata is dictionary

#The numpy array can now be processed as usual e.g.
processed_img = np.log(1.+dmx_img**2)
print("Example processing:\n", processed_img)


#Tip: if you don't intend to use an object returned by a function, indicate that
#with an underscore. For instance, if you didn't intend to use the metadata, you
#could have written
#dmx_img, _ = read_dm3(my_dmx_file)