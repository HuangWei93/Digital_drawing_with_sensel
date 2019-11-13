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


file_names = glob.glob(path.join('pixels/', 'pixel*.npy'))
file_names = sorted(file_names)
rectangle_size = np.int(sys.argv[1])

for i,  file_name in enumerate(file_names):
    pixels = np.load(file_name)

    snapshots = [ mat.reshape(rectangle_size,rectangle_size)for mat in pixels ]

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure( figsize=(4,4) )

    a = snapshots[0]
    im = plt.imshow(a, cmap='gray', interpolation='none', aspect='auto', vmin=0, vmax=255)

    def animate_func(i):

        im.set_array(snapshots[i])
        plt.title('{}'.format(i))
        return [im]

    anim = animation.FuncAnimation(
                                   fig,
                                   animate_func,
                                   frames = len(pixels),
                                   interval = 2000, # in ms
                                   )

    anim.save('animation_of_pixels/pixels_anim{}.mp4'.format(i+1), fps=10, extra_args=['-vcodec', 'libx264'])
    
print('Done!')
