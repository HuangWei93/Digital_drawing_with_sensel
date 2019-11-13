import numpy as np
import cv2
import os.path as path
import glob
from scipy import ndimage
import sys

modified_inputs = glob.glob(path.join('../generate_paras/modified_inputs', 'modified_input*.npy'))
modified_inputs = sorted(modified_inputs)
rectangle_size = np.int(sys.argv[1])
for i, modified_input in enumerate(modified_inputs):
    img = cv2.imread('processed_images/processed_image{}.JPG'.format(i+1))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    eq_global = cv2.equalizeHist(gray)
    _, th = cv2.threshold(eq_global, thresh=10, maxval=255, type=cv2.THRESH_BINARY)
    th = ~th
    kernel = np.ones((12, 12), np.uint8)
    closing = cv2.morphologyEx(th.astype(np.uint8), cv2.MORPH_DILATE, kernel, iterations = 1)
    coord = np.load(modified_input)
    pixels = []
    centroids = []
    for id, point in enumerate(coord):
        x = point[0]*width/190.0
        y = point[1]*height/110.0
        pixel = closing[int(y)-32: int(y)+32, int(x)-32: int(x)+32]
        if np.sum(pixel) != 0:
            for k in range(2):
                pixel_y, pixel_x = ndimage.measurements.center_of_mass(pixel)
                y = int(y)-32 + pixel_y
                x = int(x)-32 + pixel_x
                pixel = closing[int(y)-32: int(y)+32, int(x)-32: int(x)+32]
        else:
            pass
        ## Rotaion and get pixels here
        cnt = np.array([[x-rectangle_size//2, y-rectangle_size//2], [x+rectangle_size//2, y-rectangle_size//2], [x+rectangle_size//2, y+rectangle_size//2], [x-rectangle_size//2, y+rectangle_size//2]])
        bias = np.array([[x], [y]])
        cnt = cnt.T - bias
        cos_angle = point[2]/np.sqrt(point[2]**2 + point[3]**2)
        sin_angle = point[3]/np.sqrt(point[2]**2 + point[3]**2)
        rotation_matrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])
        cnt = np.dot(rotation_matrix, cnt) + bias
        cnt = cnt.T
        src_pts = cnt.astype("float32")
        dst_pts = np.array([[0, 0], [rectangle_size, 0], [rectangle_size, rectangle_size], [0, rectangle_size]], dtype="float32")
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        pixel = cv2.warpPerspective(gray, M, (rectangle_size, rectangle_size))
        ##
        pixel = pixel.flatten()
        pixels.append(pixel)
        centroids.append(np.array([x,y]))
        if id % 3 == 0:
            cnt = cnt[np.newaxis, :]
            cnt = cnt.astype(int)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img,[box],0,(0,0,255),2)
    pixels = np.vstack(pixels)
    centroids = np.vstack(centroids)
    np.save('pixels/pixel{}.npy'.format(i+1), pixels)
    np.save('centroids/centroid{}.npy'.format(i+1), centroids)
    cv2.imwrite('images_with_centroids/image_with_centroids{}.jpg'.format(i+1), img)
