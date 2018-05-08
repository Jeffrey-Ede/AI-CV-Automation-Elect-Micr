import re
import matplotlib.pyplot as plt
import numpy as np

log_file = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/autoencoder_xception-multi-gpu-1/log.txt"

losses = []
with open(log_file, "r") as f:
    for line in f:
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)

        for i in range(1, len(numbers), 2):
            losses.append(float(numbers[i]))

def moving_average(a, n=100):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

losses = moving_average(np.array(losses[1:25000]))

plt.plot(losses)
plt.show()