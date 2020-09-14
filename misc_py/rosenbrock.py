#Demo gradient descent methods with Rosenbrock function

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
#mpl.rcParams['title.fontsize'] = fontsize
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['axes.titlepad'] = 7
mpl.rcParams['savefig.dpi'] = 300
plt.rcParams["figure.figsize"] = [4, 3]

import tensorflow as tf

b = 1
num_steps = 61
start = [-0.5, 2.5]
lr = 0.05

f = lambda x,y: (x-1)**2 + b*(y-x**2)**2


# Evaluate function
X = np.arange(-1.1, 2.1, 0.01)
Y = np.arange(-1, 3.1, 0.01)
X, Y = np.meshgrid(X, Y)
Z = f(X,Y)

fig = plt.figure(figsize=(4, 3))
ax = fig.add_subplot(111)

ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(labeltop=False, labelright=False)
ax.minorticks_on()

plt.contour(X,Y,Z,100, linewidth=0.8)



#plt.plot([x0[0]],[x0[1]],marker='o',markersize=15, color ='r')

initial_value = tf.convert_to_tensor(np.array(start), dtype=tf.float32)
w = tf.Variable(start, dtype=tf.float32)

loss = f(w[0], w[1])

optimizers = {
    "SGD": tf.train.GradientDescentOptimizer(lr).minimize(loss),
    "Momentum": tf.train.MomentumOptimizer(lr, 0.9).minimize(loss),
    "ADAM": tf.train.AdamOptimizer(lr).minimize(loss),
    #"Momentum": tf.train.MomentumOptimizer(lr, 0.9).minimize(loss),
    #"Nesterov Momentum": tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True).minimize(loss),
    }

locs = {}
label_iters = {}

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    for k in optimizers:
        opt = optimizers[k]

        sess.run(w.assign(initial_value))
        xs = []
        ys = []
        label_xs = []
        label_ys = []
        for i in range(num_steps):
            loc = sess.run(w)
            xs.append(loc[0])
            ys.append(loc[1])

            loss_val, _ = sess.run([loss, opt])
            print(loss_val)

            if not i % (num_steps//4):
                label_xs.append(loc[0])
                label_ys.append(loc[1])

        locs[k] = [xs, ys]
        label_iters[k] = [label_xs, label_ys]


colors = ["b", "r", "g"]
for c, k in zip(colors, locs):
    xs, ys = locs[k]
    line, = ax.plot(xs, ys, label=k, linewidth=0.8)

    x, y = label_iters[k]
    plt.plot(x,y, marker='o',markersize=3, color=line.get_color(), linewidth=0)

ax.legend(loc='upper right', frameon=True, fontsize=7, framealpha=1)

ax.set_ylabel('$x_2$')
ax.set_xlabel('$x_1$')

#plt.show()

save_loc =  "Z:/Jeffrey-Ede/models/stem-random-walk-nin-figures/rosenbrock_example.png"
plt.savefig( save_loc, bbox_inches='tight')

