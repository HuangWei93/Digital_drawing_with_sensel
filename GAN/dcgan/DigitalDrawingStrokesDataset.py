from __future__ import print_function, division
import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class DigitalDrawingStrokesDataset(Dataset):
    """Digital Drawing Strokes Dataset."""

    def __init__(self, features_npy, images_npy, transform=None):
        self.images = [Image.fromarray(raw.reshape(64, 64)) for raw in np.load(images_npy)]
        self.features = np.load(features_npy)
        self.transform = transform
        self.ranges = self._set_ranges()
        assert len(self.images) == len(self.features), print("Different number of elements ({} images and {} features rows)".format(len(self.images), len(self.features)))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if self.transform is not None:
            img = self.transform(img)

        return img, self.features[idx]

    def _set_ranges(self):
        maxs = np.amax(self.features, axis=0)
        mins = np.amin(self.features, axis=0)
        assert len(maxs) == len(mins), print("Maxs and mins should have same dimension ({}, {})".format(len(maxs), len(mins)))
        return np.array(list(zip(mins, maxs)))

    def get_random_conditioning_data(self, n_items):
        # return np.array([[np.random.uniform(r[0], r[1]) for r in self.ranges] for i in range(n_items)])
        return np.array([[((np.random.uniform(r[0], r[1]) - r[0])/(r[1] - r[0])) for r in self.ranges] for i in range(n_items)])

if __name__ == '__main__':
    features_npy = './features.npy'
    images_npy = './reshaped_images3.npy'

    dataset = DigitalDrawingStrokesDataset(
        features_npy=features_npy,
        images_npy=images_npy,
        transform=transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    )

    ranges = dataset._set_ranges()

    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    #
    # for i, (imgs, labels) in enumerate(dataloader):
    #
    #     print(labels)


    # Coord_X (to be removed), Coord_Y (to be removed), Velocity_X, Velocity_Y, Pressure, Theta, Phi (to be removed), Phi_modified

    conditioning_data = np.array([np.random.uniform(r[0], r[1]) for r in ranges])
    print(ranges)
    print(conditioning_data)

    tmp = dataset.get_random_conditioning_data(100)
    print(tmp.shape)
    print(tmp)




