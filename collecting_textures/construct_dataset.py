import numpy as np
import cv2
import os.path as path
import glob
from scipy import ndimage
from scipy import interpolate
import sys
from utils_sampling import *

def main():
    #Start points and end points for each separate curve in each image.
    start_ids_list = [[5, 53, 97, 277, 371]]
    end_ids_list = [[45, 86, 225, 350, 540]]
    modified_inputs = glob.glob(path.join('../generate_paras/modified_inputs', 'modified_input*.npy'))
    modified_inputs = sorted(modified_inputs)
    #Patch size: 48
    rectangle_size = np.int(sys.argv[1])
    #Extract patches from each image
    textures = [] #containing textures
    paras = [] #containing paras
    for i, modified_input in enumerate(modified_inputs):
        #Import raw parameters <coordinates-2, velocity-2, force, original_phi, theta, modified_phi>
        raw_paras = np.load('../generate_paras/modified_inputs/modified_input{}.npy'.format(i+1))
        #Curves list contains all independent curves in one image
        curves = []
        start_ids = start_ids_list[i]
        end_ids = end_ids_list[i]
        for k in range(len(start_ids)):
            curves.append(raw_paras[start_ids[k]:end_ids[k],:])
        #Import corresponding processed images
        img = cv2.imread('processed_images/processed_image{}.JPG'.format(1+i))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #gray image
        height, width = gray.shape
        eq_global = cv2.equalizeHist(gray) #To equalize histograms of images
        _, th = cv2.threshold(eq_global, thresh=10, maxval=255, type=cv2.THRESH_BINARY) #pay attention to  thresh
        th = ~th
        kernel = np.ones((12, 12), np.uint8)
        closing = cv2.morphologyEx(th.astype(np.uint8), cv2.MORPH_DILATE, kernel, iterations = 1)#dilate images
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
            #cubic spline interpolation
            tck, u = interpolate.splprep([x_sample, y_sample, vx, vy, force, theta, phi], u=u, s=0)
            #reparameterize to real arc-lenght unit: mm
            new_u = reparametrize(tck, u)
            tck_cubic, _ = interpolate.splprep([x_sample, y_sample, vx, vy, force, theta, phi], u=new_u, s=0)
            #control overlapping when constructing dataset we use 0.2
            scale = 1.0 - np.float(sys.argv[2])
            #ratio: 1/30 mm/pixel
            ratio = 1/30.0
            eq_sample = np.arange(0, new_u[-1] + scale*rectangle_size*ratio, scale*rectangle_size*ratio)
            
            out_cubic = interpolate.splev(eq_sample, tck_cubic)
            new_x_sample = out_cubic[0]
            new_y_sample = out_cubic[1]
            new_dx_sample = out_cubic[2]
            new_dy_sample = out_cubic[3]
            new_force = out_cubic[4]
            new_theta = out_cubic[5]
            new_phi = out_cubic[6]
            for id in range(len(eq_sample)):
                x = new_x_sample[id]*width/190.0
                y = new_y_sample[id]*height/110.0
                #Moving to centroid by interative algorithm
                texture = closing[int(y)-rectangle_size: int(y)+rectangle_size, int(x)-rectangle_size: int(x)+rectangle_size]
                if np.sum(texture) != 0:
                    for iter in range(2):
                        pixel_y, pixel_x = ndimage.measurements.center_of_mass(texture)
                        y = int(y)-rectangle_size + pixel_y
                        x = int(x)-rectangle_size + pixel_x
                        texture = closing[int(y)-rectangle_size: int(y)+rectangle_size, int(x)-rectangle_size: int(x)+rectangle_size]
                else:
                    pass
                #top-left, top-right, down-right, down-left
                cnt = np.array([[x-rectangle_size//2, y-rectangle_size//2], [x+rectangle_size//2, y-rectangle_size//2], [x+rectangle_size//2, y+rectangle_size//2], [x-rectangle_size//2, y+rectangle_size//2]])
                bias = np.array([[x], [y]])
                cnt = cnt.T - bias
                cos_angle = new_dx_sample[id]/np.sqrt(new_dx_sample[id]**2 + new_dy_sample[id]**2)
                sin_angle = new_dy_sample[id]/np.sqrt(new_dx_sample[id]**2 + new_dy_sample[id]**2)
                rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]]) #rotation matrix
                cnt = np.dot(rotation_matrix, cnt) + bias
                cnt = cnt.T
                src_pts = cnt.astype("float32")
                #top-left, top-right, down-right, down-left
                dst_pts = np.array([[0, 0], [rectangle_size, 0], [rectangle_size, rectangle_size], [0, rectangle_size]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                texture = cv2.warpPerspective(gray, M, (rectangle_size, rectangle_size))
                texture = texture.flatten()
                textures.append(texture)
                para = np.array([new_x_sample[id], new_y_sample[id], new_dx_sample[id], new_dy_sample[id], new_force[id], new_theta[id], new_phi[id]])
                paras.append(para)
                #visualize extracted patches
                if id % 1 == 0:
                    cnt = cnt[np.newaxis, :]
                    cnt = cnt.astype(int)
                    rect = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    cv2.drawContours(img,[box],0,(0,0,255),2)
        cv2.imwrite('../dataset/image_with_patches/image_with_patches{}.jpg'.format(1+i), img)
    textures = np.vstack(textures)
    paras = np.vstack(paras)
    np.save('../dataset/textures.npy', textures)
    np.save('../dataset/paras.npy', paras)

if __name__== "__main__":
    main()


