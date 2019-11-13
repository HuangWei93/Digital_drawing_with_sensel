import cv2
import numpy as np
from sklearn.cluster import KMeans
import sys
import os
import glob
import os.path as path

def main():
    #Load paras
    X = np.load('paras.npy')
    #Only use <velocity-2, force, theta, modified_phi>
    X = X[:,2:]
    X.astype('float32')
    print('Means of parameters:')
    output_means = np.mean(X, axis = 0)
    output_means = np.round([float(item) for item in output_means],4)
    output_means = [float(item) for item in output_means]
    print(output_means)
    print('Stds of parameters:')
    output_stds = np.std(X, axis = 0)
    output_stds = np.round([float(item) for item in output_stds],4)
    output_stds = [float(item) for item in output_stds]
    print(output_stds)
    
    filename = './classification/paras/mean_and_std.npy'
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    np.save(filename, np.concatenate((np.mean(X, axis = 0), np.std(X, axis = 0))))
    #Normalize parameters
    means = np.mean(X, axis = 0)
    stds = np.std(X, axis = 0)
    X = (X - means)/stds
    textures = np.load('textures.npy')
    n_clusters = np.int(sys.argv[2]) #number of classes
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    centers = kmeans.cluster_centers_
    rectangle_size = np.int(sys.argv[1])
    filename = './classification/paras/centers.npy'
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    np.save(filename, centers)
    print('Centers:')
    real_centers = centers * stds + means
    print(real_centers)

    for k in range(n_clusters):
        labels = np.where(kmeans.labels_ == k)
        for i, id in enumerate(labels[0]):
            im = textures[id,:]
            im = im.reshape(rectangle_size,rectangle_size)
            filename = './classification/paras/class{0}/image{1}.jpg'.format(k,i)
            os.makedirs(os.path.dirname(filename), exist_ok = True)
            cv2.imwrite(filename, im)
    print(kmeans.labels_)

    #Concatenate all images in one class folder
    print('Merging images in one class folder')

    for k in range(n_clusters):
        class_id = k
        images = glob.glob(path.join('./classification/paras/class{}/'.format(int(class_id)), 'image*.jpg'))
        size = int(np.sqrt(len(images)))
        images = images[:size**2]
        ls =[]
        for i, texture_input in enumerate(images):
            if i%size == 0:
                if i != 0:
                    ls.append(np.concatenate(row, 1))
                row = []
            img = cv2.imread(texture_input)
            row.append(img)
        img = np.concatenate(ls, 0)
        filename = './classification/merged_images/paras/class{}.jpg'.format(class_id)
        os.makedirs(os.path.dirname(filename), exist_ok = True)
        cv2.imwrite(filename, img)

if __name__== "__main__":
    main()
