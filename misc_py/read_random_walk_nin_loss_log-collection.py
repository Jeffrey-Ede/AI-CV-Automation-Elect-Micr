import numpy as np
import re

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
fontsize = 9
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['axes.titlepad'] = 7
mpl.rcParams['savefig.dpi'] = 600
plt.rcParams["figure.figsize"] = [4,3]

take_ln = True
moving_avg = True
save = True
save_val = True
window_size = 1#2500
dataset_num = 8
mean_from_last = 20000
remove_repeats = True #Caused by starting from the same counter multiple times

scale = 1.0
ratio = 1.3 # 1.618
width = scale * 3.3
height = (width / 1.618)
num_data_to_use = 20000
num_hist_bins = 200
mse_x_to = 0.012

f = plt.figure()

labels_sets = [["Path Info, Extra Residuals",
          "Path Info, No Extra Residuals",
          "No Path Info, No Extra Residuals"],
          ["Small Kernels",
           "Large Kernels"],
          [r"75k-50k-175k, LR 0.0002, No Decay",
           r"200k-50k-150k, LR 0.0002, No Decay",
           r"270k-60k-120k, LR 0.0002, Decay",
           r"270k-60k-120k, LR 0.0004, Decay"],
          [r"Leaky ReLU, Output in [$-$1,1]",
           r"Leaky ReLU, Output in [0,1]"]
          ]
sets = [[1, 2, 3], [4, 5], [1, 4, 6, 7], [7, 8]]

losses_sets = []
iters_sets = []
for i, (data_nums, labels) in enumerate(zip(sets, labels_sets)):
    for j, dataset_num in enumerate(data_nums):
        log_loc = ("Z:/Jeffrey-Ede/models/stem-random-walk-nin-20-"+str(dataset_num)+"/")
        log_file = log_loc+"log.txt"
        val_file = log_loc+"val_log.txt"

        notes_file = log_loc+"notes.txt"
        with open(notes_file, "r") as f:
            for l in f: print(l)

        switch = False
        losses = []
        vals = []
        losses_iters = []
        with open(log_file, "r") as f:
            for line in f:
                numbers = re.findall(r"\[([-+]?\d*\.\d+|\d+)\]", line)
        
                val_marks = re.findall(r"(\d+)NiN", line)
                val_marks = [x is '1' for x in val_marks]

                losses = [float(x) for x in numbers]
                vals = [x for x, m in zip(losses, val_marks) if m]
                losses = [x for x, m in zip(losses, val_marks) if not m]
                losses_iters = [i for i in range(1, len(losses)+1)]
        try:
            switch = False
            val_losses = []
            val_iters = []
            with open(val_file, "r") as f:
                for line in f:
                    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", line)

                    for i in range(1, len(numbers), 2):

                        val_losses.append(float(numbers[i]))
                        val_iters.append(float(numbers[i-1]))
        except:
            print("No val log {}".format(val_file))

        def moving_average(a, n=window_size):
            ret = np.cumsum(np.insert(a,0,0), dtype=float)
            return (ret[n:] - ret[:-n]) / float(n)

        losses = moving_average(np.array(losses[:])) if moving_avg else np.array(losses[:])
        losses_iters = moving_average(np.array(losses_iters[:])) if moving_avg else np.array(losses[:])
        val_losses = moving_average(np.array(val_losses[:])) if moving_avg else np.array(val_losses[:])
        val_iters = moving_average(np.array(val_iters[:])) if moving_avg else np.array(val_iters[:])

        print(np.mean((losses[(len(losses)-mean_from_last):])[np.isfinite(losses[(len(losses)-mean_from_last):])]))

        #if take_ln:
        #    import math
        #    losses = math.log(losses)

        losses = losses[10000:]
        losses_sets.append(losses)

        losses_iters = losses_iters[10000:]
        losses_iters = [i/1000 for i in losses_iters]
        iters_sets.append(losses_iters)
        print(j, len(labels))
        plt.plot(losses_iters, losses if take_ln else losses, label=labels[j], linewidth=0.8)
        #plt.plot(val_iters, np.log(val_losses) if take_ln else val_losses)

    plt.axvline(x=10, color='black', linestyle='--', linewidth=0.8)
    plt.ylabel('Mean Squared Error-Based Loss')
    plt.xlabel('Training Iterations (x10$^3$)')
    plt.legend(loc='upper right', frameon=False)

    #plt.show()

    #f.set_size_inches(width, height)

    save_loc =  "Z:/Jeffrey-Ede/models/stem-random-walk-nin-figures/" + str(i+1) + ".png"
    plt.savefig( save_loc, bbox_inches='tight', )

    plt.gcf().clear()
