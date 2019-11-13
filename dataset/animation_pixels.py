# Usually we use `%matplotlib inline`. However we need `notebook` for the anim to render in the notebook.

import random
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import matplotlib.animation as animation

import cv2
import os.path as path
import glob
import sys

rectangle_size = np.int(sys.argv[1])
pixels = np.load('textures.npy')
paras = np.load('paras.npy')

snapshots = [ mat.reshape(rectangle_size,rectangle_size)for mat in pixels ]

# First set up the figure, the axis, and the plot element we want to animate
fig = plt.figure( figsize=(8,8) )

a = snapshots[0]
im = plt.imshow(a, cmap='gray', interpolation='none', aspect='auto', vmin=0, vmax=255)

def animate_func(i):

    im.set_array(snapshots[i])
    para = np.round([float(item) for item in paras[i,2:]],2)
    para = [float(item) for item in para]
    plt.title('ID:{0} \n {1}'.format(i, para))
    return [im]

anim = animation.FuncAnimation(
                               fig,
                               animate_func,
                               frames = len(pixels),
                               interval = 2000, # in ms
                               )

anim.save('textures_anim.mp4', fps=10, extra_args=['-vcodec', 'libx264'])

