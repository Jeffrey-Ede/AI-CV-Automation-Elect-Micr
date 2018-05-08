import numpy as np

paramsLoc = "E:/stills_all_profile/data.txt"
saveFile = "E:/stills_all_profile/data.npy"

dataset = []
with open(paramsLoc, "r") as f:
    k = 0
    for line in f:
        if k:
            data = [float(s) for s in line.split(",")]
            dataset.append(np.array(data))
        k += 1

np.save(saveFile, np.asarray(dataset))