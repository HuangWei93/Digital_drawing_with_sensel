import numpy as np
import cv2
import os.path as path
import glob
from scipy import ndimage
from scipy import interpolate
import sys
from utils_sampling import *




start_ids_list = [[5, 53, 97, 277, 371]]
end_ids_list = [[45, 86, 225, 350, 540]]

modified_inputs = glob.glob(path.join('../generate_paras/modified_inputs', 'modified_input*.npy'))
modified_inputs = sorted(modified_inputs)
rectangle_size = np.int(sys.argv[1])

for i, modified_input in enumerate(modified_inputs):
    raw_inputs = np.load('../generate_paras/modified_inputs/modified_input{}.npy'.format(i+1))
    curves = []
    start_ids = start_ids_list[i]
    end_ids = end_ids_list[i]
    for k in range(len(start_ids)):
        curves.append(raw_inputs[start_ids[k]:end_ids[k],:])

    height = 3300
    width = 5700
    img = np.zeros(shape=[height, width, 3], dtype=np.uint8)
    img.fill(255)
    for k in range(len(start_ids)):
        x_sample=curves[k][:,0]
        y_sample=curves[k][:,1]
        vx = curves[k][:,2]
        vy = curves[k][:,3]
        force = curves[k][:,4]
        theta = curves[k][:,6]
        phi = curves[k][:,7]
        t_span = len(x_sample)
        u = np.arange(t_span)
        u = u.astype('float32')
        tck, u = interpolate.splprep([x_sample, y_sample, vx, vy, force, theta, phi], u=u, s=0)
        new_u = reparametrize(tck, u)
        tck_cubic, _ = interpolate.splprep([x_sample, y_sample, vx, vy, force, theta, phi], u=new_u, s=0)
        scale = 1.0 - np.float(sys.argv[2])
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
        new_dx_sample = out_cubic[2]
        new_dy_sample = out_cubic[3]
        new_force = out_cubic[4]
        new_theta = out_cubic[5]
        new_phi = out_cubic[6]
        dout_cubic = interpolate.splev(sample_t, tck_cubic, der=1)
        ddout_cubic = interpolate.splev(sample_t, tck_cubic, der=2)
        curvature = (dout_cubic[0]*ddout_cubic[1] - dout_cubic[0]*ddout_cubic[0])/((dout_cubic[0]**2 + dout_cubic[1]**2)**1.5)
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
            cv2.drawContours(img,[box],0,(0,0,255),2)

    cv2.imwrite('./images_with_resampled_patches/resampled_patches{}.jpg'.format(1+i), img)


