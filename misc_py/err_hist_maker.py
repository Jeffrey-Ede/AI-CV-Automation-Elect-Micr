import numpy as np
import matplotlib as mpl
#mpl.use('pdf')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
fontsize = 10
mpl.rcParams['axes.labelsize'] = fontsize
mpl.rcParams['xtick.labelsize'] = fontsize
mpl.rcParams['ytick.labelsize'] = fontsize
mpl.rcParams['legend.fontsize'] = fontsize

import matplotlib.mlab as mlab

import scipy.stats as stats

# width as measured in inkscape
scale = 1.0
width = scale * 2.2 * 3.487
height = (width / 1.618) / 2.2
num_data_to_use = 20000
num_hist_bins = 200
mse_x_to = 0.012 

codes = [(7, 2, x+1) for x in range(14)]
labels = ["Unfiltered", "Gaussian", "Bilateral", "Median", "Wiener", "Wavelet", "Chambolle"]
data = np.load('//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/train-losses.npy')

datasets = []
means = []
for comp_idx in range(2):
    for metric_idx in range(7):
        dataset = data[:num_data_to_use,metric_idx,comp_idx]
        
        mean = np.mean(dataset[np.isfinite(dataset)])
        dataset[np.logical_not(np.isfinite(dataset))] = mean

        if comp_idx == 0:
            dataset[dataset > mse_x_to] = mse_x_to
        elif comp_idx == 1:
            dataset = dataset.clip(0.,1.)

        means.append(mean)
        datasets.append(dataset)

f = plt.figure(1)

print(means)

def subplot_creator(loc, data):
    plt.subplot(loc[0], loc[1], loc[2])

    # the histogram of the data
    n, bins, patches = plt.hist(data, 30, normed=1, facecolor='grey', edgecolor='black', alpha=0.75, linewidth=1)

    # add a 'best fit' line
    #y = mlab.normpdf( bins, mu, sigma)
    #l = plt.plot(bins, y, 'r--', linewidth=1)

    #plt.xlabel('Smarts')
    #plt.ylabel('Probability')
    #plt.title(r'$\mathrm{Histogram\ of\ IQ:}\ \mu=100,\ \sigma=15$')
    #plt.axis([40, 160, 0, 0.03])
    #plt.grid(True)

    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=False)
    plt.rc('xtick', labelsize=8)
    plt.rc('ytick', labelsize=8)
    plt.rc('axes', labelsize=8)

bins_set = []
density_set = []
for i in range(len(datasets)):
    density_set.append(stats.gaussian_kde(datasets[i]))
    n, bins, patches = plt.hist(np.asarray(datasets[i]).T, num_hist_bins, normed=1, histtype='step')
    bins_set.append(bins)

plt.clf()

integs = []
maxs = [0., 0.]
for i in range(7):
    dens = density_set[i](bins_set[i])
    
    dens = [den for _, den in sorted(zip(bins_set[i], dens))]
    bins = sorted(bins_set[i])
    integ = np.trapz(dens, bins)

    max = np.max(dens/integ)
    if max > maxs[0]:
        maxs[0] = max

    integs.append(integ)

for i in range(7, 14):
    dens = density_set[i](bins_set[i])
    
    dens = [den for _, den in sorted(zip(bins_set[i], dens))]
    bins = sorted(bins_set[i])
    integ = np.trapz(dens, bins)

    max = np.max(dens/integ)
    if max > maxs[1]:
        maxs[1] = max

    integs.append(integ)

ax = f.add_subplot(1,2,1)
for i in range(7):
    dens = density_set[i](bins_set[i])
    dens /= integs[i]
    print(np.sum(dens))
    print( 0.012 / maxs[0])
    dens /= maxs[0]

    #bins_to_use = bins_set[i] < 0.006
    #bins_not_to_use = np.logical_not(bins_to_use)
    #bins = np.append(bins_set[i][bins_to_use], 0.008)
    #dens = np.append(dens[bins_to_use], np.sum(dens[bins_not_to_use]))

    plt.plot(bins_set[i], dens, linewidth=1., label=labels[i])
plt.xlabel('Mean Squared Error')
plt.ylabel('Relative PDF')
plt.minorticks_on()
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.set_xscale('log')
#ax.grid()
#plt.rc('font', family='serif', serif=['Times'])
#plt.rc('text', usetex=False)
#plt.rc('xtick', labelsize=8)
#plt.rc('ytick', labelsize=8)
#plt.rc('axes', labelsize=8)

plt.legend(loc='upper right', frameon=False)

ax = f.add_subplot(1,2,2)
for i in range(7, 14):
    dens = density_set[i](bins_set[i])
    dens /= integs[i]
    print(np.sum(dens))
    print(1. / maxs[1])
    dens /= maxs[1]
    plt.plot(bins_set[i], dens, linewidth=1.)
plt.xlabel('Structural Similarity Index')
plt.minorticks_on()
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
#ax.grid()
plt.tick_params()
##plt.rc('font', family='serif', serif=['Times'])
#plt.rc('text', usetex=False)
#plt.rc('xtick', labelsize=8)
#plt.rc('ytick', labelsize=8)
#plt.rc('axes', labelsize=8)

#plt.show()

#for code, data in zip(codes, datasets):
#    subplot_creator(code, data)

f.subplots_adjust(wspace=0.18, hspace=0.18)
f.subplots_adjust(left=.00, bottom=.00, right=1., top=1.)

#ax.set_ylabel('Some Metric (in unit)')
#ax.set_xlabel('Something (in unit)')
#ax.set_xlim(0, 3*np.pi)

f.set_size_inches(width, height)

plt.show()

f.savefig('plot.pdf', bbox_inches='tight')
