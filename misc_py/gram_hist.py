import matplotlib.pyplot as plt
import numpy as np

file = "E:/stills_all/diffs_of_gram_matrices.npy"
data = np.load(file)

n, bins, patches = plt.hist(data, 50, density=True, facecolor='g', alpha=0.75)
plt.show()