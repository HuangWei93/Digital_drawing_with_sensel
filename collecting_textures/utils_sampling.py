import numpy as np
import cv2
import os.path as path
import glob
from scipy import ndimage
from scipy import interpolate
import sys

def compute_arc_length(out):
    dist = 0
    x = out[0]
    y = out[1]
    for i in range(len(x) - 1):
        dist = dist + np.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
    return dist

def reparametrize(tck, u):
    new_u = []
    new_u.append(0)
    for i in range(len(u) - 1):
        dense_samples = np.linspace(u[i],u[i+1], 100)
        out = interpolate.splev(dense_samples , tck, der=0)
        arc_length = compute_arc_length(out)
        new_u.append(arc_length + new_u[-1])
    return np.array(new_u)

def get_shift(dist, scale):
    if scale == 1.0:
        if dist<1.4350:
            shift_term = 0.8
        elif dist<1.5580:
                shift_term = 0.6
        elif dist<1.5820:
            shift_term = 0.5
        elif dist<1.5900:
                shift_term = 0.4
        elif dist<1.5960:
            shift_term = 0.3
        elif dist<1.5990:
            shift_term = 0.15
        else:
            shift_term = 0
    elif scale == 0.8:
        if dist<1.1950:
            shift_term = 0.4
        elif dist<1.2600:
            shift_term = 0.2
        elif dist<1.2800:
            shift_term = 0.1
        else:
            shift_term = 0.05
    return shift_term
