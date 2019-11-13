import numpy as np
import cv2
import os.path as path
import glob
from scipy import ndimage
from scipy import interpolate
import sys
from sklearn.cluster import KMeans
import random
from utils_sampling import *

def main():
    #load paras and textures
    dataset_textures = np.load('../dataset/textures.npy')
    dataset_paras = np.load('../dataset/paras.npy')
    dataset_paras = dataset_paras[:,(2,3,4,5,6)]
    rectangle_size = np.int(sys.argv[1])
    means = np.mean(dataset_paras, axis = 0)
    stds = np.std(dataset_paras, axis = 0)
    dataset_paras = (dataset_paras - means)/stds
    ##Kmeans Part##
    n_clusters = 20
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(dataset_paras)
    centers = kmeans.cluster_centers_

    def find_candidates(para):
        global kmeans
        global means
        global stds
        para = para[np.newaxis, :]
        cluster = kmeans.predict((para-means)/stds)
        labels = np.where(kmeans.labels_ == cluster[0])
        labels = labels[0]
        return labels





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
        #Create one blanket white image
        height = 3300
        width = 5700
        gray = np.zeros(shape=[height, width], dtype=np.uint8)
        gray.fill(255)
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
            #Adapt sampling in order to avoid gap
            sample_t = [] # adapted sample points: sample_t
            current_t = 0
            while current_t <= new_u[-1]:
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
            #Compute curvature for each sample points
            dout_cubic = interpolate.splev(sample_t, tck_cubic, der=1)
            ddout_cubic = interpolate.splev(sample_t, tck_cubic, der=2)
            curvature = (dout_cubic[0]*ddout_cubic[1] - dout_cubic[0]*ddout_cubic[0])/((dout_cubic[0]**2 + dout_cubic[1]**2)**1.5)
            pre_mask = np.zeros(shape=[height, width], dtype=np.uint8)
            for id in range(len(sample_t)):
                x = new_x_sample[id]*width/190.0
                y = new_y_sample[id]*height/110.0
                cnt = np.array([[x-rectangle_size//2, y-rectangle_size//2], [x+rectangle_size//2-1, y-rectangle_size//2], [x+rectangle_size//2-1, y+rectangle_size//2-1], [x-rectangle_size//2, y+rectangle_size//2-1]])
                bias = np.array([[x], [y]])
                cnt = cnt.T - bias
                cos_angle = new_dx_sample[id]/np.sqrt(new_dx_sample[id]**2 + new_dy_sample[id]**2)
                sin_angle = new_dy_sample[id]/np.sqrt(new_dx_sample[id]**2 + new_dy_sample[id]**2)
                rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
                cnt = np.dot(rotation_matrix, cnt) + bias
                cnt = cnt.T
                dst_pts = cnt.astype("float32")
                src_pts = np.array([[0, 0], [rectangle_size-1, 0], [rectangle_size-1, rectangle_size - 1], [0, rectangle_size - 1]], dtype="float32")
                M = cv2.getPerspectiveTransform(src_pts, dst_pts)
                mask = np.ones(shape=[rectangle_size, rectangle_size], dtype=np.uint8)
                paras = np.array([new_dx_sample[id], new_dy_sample[id], new_force[id], new_theta[id], new_phi[id]])
                candidates = find_candidates(paras)
                mask = cv2.warpPerspective(mask, M, (width, height))
                min_diff = float("inf")
                best_choice = 0
                for j in range(len(candidates)):
                    texture =dataset_textures[candidates[j],:]
                    texture = texture.reshape(rectangle_size,rectangle_size)
                    patch = cv2.warpPerspective(texture, M, (width, height))
                    overlap = mask & pre_mask
                    diff = np.sum(abs((gray - patch) * overlap))
                    if diff < min_diff:
                        best_choice = candidates[j]
                        min_diff = diff
                texture = dataset_textures[best_choice,:]
                texture = texture.reshape(rectangle_size,rectangle_size)
                patch = cv2.warpPerspective(texture, M, (width, height))
                gray = (1-mask)*gray + mask * patch
                pre_mask = mask
                print(id)
        cv2.imwrite('reconstruct_stroke_with_eqdist_by_clustering_paras/reconstruct_stroke_with_eqdist_by_clustering_paras{}.jpg'.format(i+1), gray)

if __name__== "__main__":
    main()
