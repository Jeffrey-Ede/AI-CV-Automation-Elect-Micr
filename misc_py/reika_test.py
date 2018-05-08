import numpy as np
#from reikna.fft import FFT
#from reikna.core import Type
import pyculib

data = np.array([4, 5, 6, 7], dtype=np.complex64)
b = pyculib.fft.fft(data)
print(b)

#arr_t = Type(np.complex64, shape=data.shape)

#fft = FFT(arr_t).compile()

#r = fft(data)
#print(r)