import numpy as np

data_file = "E:/stills_all_profile/data.npy"
save_file = "E:/stills_all_profile/data_info.npy"

redistributor_size = 100

dataset = np.load(data_file)

min = np.zeros(dataset.shape[1])
max = np.zeros(dataset.shape[1])
redistributors = []
means = np.zeros(dataset.shape[1])
for i in range(dataset.shape[1]):
    #Extrema
    data = dataset[:, i:(i+1)]
    data = data[np.isfinite(data)]
    min[i] = np.min(data)
    max[i] = np.max(data)
    means[i] = np.mean(data)

    #Redistribution
    sorted = np.sort(data.clip(min[i], max[i]))
    sorted = (sorted-min[i]) / (max[i]-min[i])
    sum = np.sum(sorted)
    cumsum = np.cumsum(sorted)

    redistributor = np.zeros(redistributor_size)
    for j in range(redistributor_size-1):
        threshold = (j+1)*sum/redistributor_size
        redistributor[j] = next(value for value in cumsum if value >= threshold)
    redistributor[redistributor_size-1] = cumsum[len(cumsum)-1]
    redistributor = redistributor / np.max(redistributor)

    redistributors.append(np.array(redistributor))


redistributors = np.asarray(redistributors)
dataset_info = { 'min': min, 'max': max, 'redistributors': redistributors, 'mean': means }

np.save(save_file, dataset_info)