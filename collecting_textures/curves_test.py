import numpy as np
import cv2
import os.path as path
import glob
from scipy import ndimage
from scipy import interpolate
import sys
from utils_sampling import *



rectangle_size = np.int(sys.argv[1])

height = 3300
width = 5700
img = np.zeros(shape=[height, width, 3], dtype=np.uint8)
img.fill(255)
radii = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
shift = 5
for i in range(len(radii)):
    theta = np.linspace(0,np.pi,100)
    radius = radii[i]
    x_sample= radius * np.cos(theta) + shift + 2*radius
    shift = shift + 2*radius + 2
    y_sample= radius * np.sin(theta) + 10
    t_span = len(x_sample)
    u = np.arange(t_span)
    u = u.astype('float32')
    tck, u = interpolate.splprep([x_sample, y_sample], u=u, k=3, s=0)
    new_u = reparametrize(tck, u)
    tck_cubic, _ = interpolate.splprep([x_sample, y_sample], u=new_u, k=3, s=0)
    scale = 1.0 - np.float(sys.argv[2])
    eq_sample = np.arange(0, new_u[-1] + scale*rectangle_size/30.0, scale*rectangle_size/30.0)
    out_cubic = interpolate.splev(eq_sample, tck_cubic)
    new_x_sample = out_cubic[0]
    new_y_sample = out_cubic[1]
    dout_cubic = interpolate.splev(eq_sample, tck_cubic, der=1)
    ddout_cubic = interpolate.splev(eq_sample, tck_cubic, der=2)
    new_dx_sample = dout_cubic[0]
    new_dy_sample = dout_cubic[1]
    dists = np.sqrt((new_x_sample[1:] - new_x_sample[:-1])**2 + (new_y_sample[1:] - new_y_sample[:-1])**2)
    print(np.mean(dists))
    for id in range(len(eq_sample)):
        x = new_x_sample[id]*width/190.0
        y = new_y_sample[id]*height/110.0
        cnt = np.array([[x-rectangle_size//2, y-rectangle_size//2], [x+rectangle_size//2, y-rectangle_size//2], [x+rectangle_size//2, y+rectangle_size//2], [x-rectangle_size//2, y+rectangle_size//2]])
        bias = np.array([[x], [y]])
        cnt = cnt.T - bias
        cos_angle = new_dx_sample[id]/np.sqrt(new_dx_sample[id]**2 + new_dy_sample[id]**2)
        sin_angle = new_dy_sample[id]/np.sqrt(new_dx_sample[id]**2 + new_dy_sample[id]**2)
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        cnt = np.dot(rotation_matrix, cnt) + bias
        cnt = cnt.T
        cnt = cnt[np.newaxis, :]
        cnt = cnt.astype(int)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(0,0,255),2)

#straight line
shift = 5
x_sample= np.arange(80) + shift
y_sample= 25 * np.ones(80)
t_span = len(x_sample)
u = np.arange(t_span)
u = u.astype('float32')
tck, u = interpolate.splprep([x_sample, y_sample], u=u, k=3, s=0)
new_u = reparametrize(tck, u)
tck_cubic, _ = interpolate.splprep([x_sample, y_sample], u=new_u, k=3, s=0)
eq_sample = np.arange(0, new_u[-1] + scale*rectangle_size/30.0, scale*rectangle_size/30.0)
out_cubic = interpolate.splev(eq_sample, tck_cubic)
new_x_sample = out_cubic[0]
new_y_sample = out_cubic[1]
dout_cubic = interpolate.splev(eq_sample, tck_cubic, der=1)
ddout_cubic = interpolate.splev(eq_sample, tck_cubic, der=2)
new_dx_sample = dout_cubic[0]
new_dy_sample = dout_cubic[1]
dists = np.sqrt((new_x_sample[1:] - new_x_sample[:-1])**2 + (new_y_sample[1:] - new_y_sample[:-1])**2)
for id in range(len(eq_sample)):
    x = new_x_sample[id]*width/190.0
    y = new_y_sample[id]*height/110.0
    cnt = np.array([[x-rectangle_size//2, y-rectangle_size//2], [x+rectangle_size//2, y-rectangle_size//2], [x+rectangle_size//2, y+rectangle_size//2], [x-rectangle_size//2, y+rectangle_size//2]])
    bias = np.array([[x], [y]])
    cnt = cnt.T - bias
    cos_angle = new_dx_sample[id]/np.sqrt(new_dx_sample[id]**2 + new_dy_sample[id]**2)
    sin_angle = new_dy_sample[id]/np.sqrt(new_dx_sample[id]**2 + new_dy_sample[id]**2)
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    cnt = np.dot(rotation_matrix, cnt) + bias
    cnt = cnt.T
    cnt = cnt[np.newaxis, :]
    cnt = cnt.astype(int)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(0,0,255),2)


shift = 5
for i in range(len(radii)):
    theta = np.linspace(0,np.pi,100)
    radius = radii[i]
    x_sample= radius * np.cos(theta) + shift + 2*radius
    shift = shift + 2*radius + 2
    y_sample= radius * np.sin(theta) + 30
    t_span = len(x_sample)
    u = np.arange(t_span)
    u = u.astype('float32')
    tck, u = interpolate.splprep([x_sample, y_sample], u=u, k=3, s=0)
    new_u = reparametrize(tck, u)
    tck_cubic, _ = interpolate.splprep([x_sample, y_sample], u=new_u, k=3, s=0)
    sample_t = []
    current_t = 0
    while current_t <=new_u[-1]:
        sample_t.append(current_t)
        out_cubic = interpolate.splev([current_t, current_t + scale*rectangle_size/30.0], tck_cubic)
        x = out_cubic[0]
        y = out_cubic[1]
        dist =  np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
        shift_term = get_shift(dist, scale)
        
        current_t = current_t + scale*rectangle_size/30.0 - shift_term
        
    out_cubic = interpolate.splev(sample_t, tck_cubic)
    new_x_sample = out_cubic[0]
    new_y_sample = out_cubic[1]
    dout_cubic = interpolate.splev(sample_t, tck_cubic, der=1)
    new_dx_sample = dout_cubic[0]
    new_dy_sample = dout_cubic[1]
    for id in range(len(sample_t)):
        x = new_x_sample[id]*width/190.0
        y = new_y_sample[id]*height/110.0
        cnt = np.array([[x-rectangle_size//2, y-rectangle_size//2], [x+rectangle_size//2, y-rectangle_size//2], [x+rectangle_size//2, y+rectangle_size//2], [x-rectangle_size//2, y+rectangle_size//2]])
        bias = np.array([[x], [y]])
        cnt = cnt.T - bias
        cos_angle = new_dx_sample[id]/np.sqrt(new_dx_sample[id]**2 + new_dy_sample[id]**2)
        sin_angle = new_dy_sample[id]/np.sqrt(new_dx_sample[id]**2 + new_dy_sample[id]**2)
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        cnt = np.dot(rotation_matrix, cnt) + bias
        cnt = cnt.T
        cnt = cnt[np.newaxis, :]
        cnt = cnt.astype(int)
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img,[box],0,(255,0,0),2)

#straight line
shift = 5
x_sample= np.arange(80) + shift
y_sample= 45 * np.ones(80)
t_span = len(x_sample)
u = np.arange(t_span)
u = u.astype('float32')
tck, u = interpolate.splprep([x_sample, y_sample], u=u, k=3, s=0)
new_u = reparametrize(tck, u)
tck_cubic, _ = interpolate.splprep([x_sample, y_sample], u=new_u, k=3, s=0)
sample_t = []
current_t = 0
while current_t <=new_u[-1]:
    sample_t.append(current_t)
    out_cubic = interpolate.splev([current_t, current_t + scale*rectangle_size/30.0], tck_cubic)
    x = out_cubic[0]
    y = out_cubic[1]
    dist =  np.sqrt((x[1] - x[0])**2 + (y[1] - y[0])**2)
    shift_term = get_shift(dist, scale)
    
    current_t = current_t + scale*rectangle_size/30.0 - shift_term

out_cubic = interpolate.splev(sample_t, tck_cubic)
new_x_sample = out_cubic[0]
new_y_sample = out_cubic[1]
dout_cubic = interpolate.splev(sample_t, tck_cubic, der=1)
new_dx_sample = dout_cubic[0]
new_dy_sample = dout_cubic[1]
for id in range(len(sample_t)):
    x = new_x_sample[id]*width/190.0
    y = new_y_sample[id]*height/110.0
    cnt = np.array([[x-rectangle_size//2, y-rectangle_size//2], [x+rectangle_size//2, y-rectangle_size//2], [x+rectangle_size//2, y+rectangle_size//2], [x-rectangle_size//2, y+rectangle_size//2]])
    bias = np.array([[x], [y]])
    cnt = cnt.T - bias
    cos_angle = new_dx_sample[id]/np.sqrt(new_dx_sample[id]**2 + new_dy_sample[id]**2)
    sin_angle = new_dy_sample[id]/np.sqrt(new_dx_sample[id]**2 + new_dy_sample[id]**2)
    rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
    cnt = np.dot(rotation_matrix, cnt) + bias
    cnt = cnt.T
    cnt = cnt[np.newaxis, :]
    cnt = cnt.astype(int)
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    cv2.drawContours(img,[box],0,(255,0,0),2)

cv2.imwrite('./curves/semi-circles.jpg', img)


