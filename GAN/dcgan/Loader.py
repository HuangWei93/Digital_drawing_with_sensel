import numpy as np
import cv2
from typing import List
import torch
import torch.utils.data as utils
import torchvision.transforms as transforms
import torchvision
import matplotlib.pyplot as plt

# Coord_X (to be removed), Coord_Y (to be removed), Velocity_X, Velocity_Y, Pressure, Theta, Phi (to be removed), Phi_modified
path = './reshaped_images3.npy'
# labels = np.load('modified_input1.npy')
raw_images = np.load('pixel1.npy')


# print(labels[0])
# print(images)
# images = np.load('reshaped_images.npy')


def visualize_image(img_array: np.ndarray):
    cv2.imshow('image', np.squeeze(img_array, axis=0))
    cv2.waitKey(0)


def reshape_images(images: List) -> np.ndarray:
    return np.array([img.reshape(1, 64, 64) for img in images])


def save_raw_images(images: List, path='./images/main/'):
    for i, img in enumerate(images):
        cv2.imwrite(path+'{}.png'.format(i), img.reshape(64, 64, 1))


def grayscale_loader(path):
    sample = torch.from_numpy(np.load(path))
    print(sample.shape)
    return sample

images = reshape_images(raw_images)
images = np.delete(images, [260, 261, 262, 263, 264], axis=0)
np.save(path, images)
# visualize_image(images[np.random.randint(low=0, high=images.shape[0])])

images = np.load(path)
save_raw_images(images)


dataset = torchvision.datasets.ImageFolder(
    root='./images/',
    transform=transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, ))
    ])
)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=10, shuffle=True)

for i, data in enumerate(dataloader, 0):
    print(i)

# https://discuss.pytorch.org/t/loading-npy-files-using-torchvision/28481


# img = torch.stack([torch.Tensor(img) for img in np.load(path)])
# dataset = utils.TensorDataset(img)
# dataloader = utils.DataLoader(dataset, batch_size=5, shuffle=True)
#



# def npy_loader(path):
#     sample = torch.from_numpy(np.load(path))
#     print(sample.shape)
#     return sample


# dataset = torchvision.datasets.DatasetFolder(
#     root='./images/',
#     loader=npy_loader,
#     extensions=('.npy')
# )



# for i, data in enumerate(dataloader, 0):
#     img = data[0][0]
#     plt.imshow(img.permute(1, 2, 0))
#     plt.show()
# break
# print(i)
# def npy_loader(path: str) -> torch.Tensor:
#     sample = torch.from_numpy(np.load(path))
#     return sample
#
#
# dataset = datasets.DatasetFolder(
#     root='./reshaped_images/',
#     loader=npy_loader,
#     extensions=['.npy']
# )

# print()
# https://discuss.pytorch.org/t/dimensions-of-an-input-image/19439
