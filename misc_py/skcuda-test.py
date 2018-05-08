import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import numpy as np
#from skcuda.fft import fft, Plan
import skcuda.cufft

N = 128
x = np.asarray(np.random.rand(N), np.float32)
xf = np.fft.fft(x)
x_gpu = gpuarray.to_gpu(x)
xf_gpu = gpuarray.empty(N/2+1, np.complex64)
plan = Plan(x.shape, np.float32, np.complex64)
fft(x_gpu, xf_gpu, plan)
np.allclose(xf[0:N/2+1], xf_gpu.get(), atol=1e-6)
