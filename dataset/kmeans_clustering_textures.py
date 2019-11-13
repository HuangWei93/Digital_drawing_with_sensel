import cv2
import numpy as np
from sklearn.cluster import KMeans
import sys
import os
import glob
import os.path as path

def main():
    
    textures = np.load('textures.npy')
    n_clusters = np.int(sys.argv[2]) #number of classes
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(textures)
    centers = kmeans.cluster_centers_
    rectangle_size = np.int(sys.argv[1])
    filename = './classification/textures/centers.npy'
    os.makedirs(os.path.dirname(filename), exist_ok = True)
    np.save(filename, centers)

    for k in range(n_clusters):
        labels = np.where(kmeans.labels_ == k)
        for i, id in enumerate(labels[0]):
            im = textures[id,:]
            im = im.reshape(rectangle_size,rectangle_size)
            filename = './classification/textures/class{0}/image{1}.jpg'.format(k,i)
            os.makedirs(os.path.dirname(filename), exist_ok = True)
            cv2.imwrite(filename, im)
    print(kmeans.labels_)

    #Concatenate all images in one class folder
    print('Merging images in one class folder')

    for k in range(n_clusters):
        class_id = k
        images = glob.glob(path.join('./classification/textures/class{}/'.format(int(class_id)), 'image*.jpg'))
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
        filename = './classification/merged_images/textures/class{}.jpg'.format(class_id)
        os.makedirs(os.path.dirname(filename), exist_ok = True)
        cv2.imwrite(filename, img)

if __name__== "__main__":
    main()
