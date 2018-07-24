from PIL import Image
import numpy as np

save_loc = r'Z:\Jeffrey-Ede\gan-compression\\'

np.random.seed(1)
arr = np.random.rand(512, 512)
select = arr < 0.01
Image.fromarray(select.astype(np.float32)).save(save_loc+"mask-100.tif")
select = arr < 0.05
Image.fromarray(select.astype(np.float32)).save(save_loc+"mask-20.tif")