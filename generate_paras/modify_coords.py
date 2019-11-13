import numpy as np
import glob
import os.path as path
import math

# utility functions


def getAngle(a, b):
    angles = []
    for i in range(len(a)):
        ang = b[i] - math.degrees(math.atan2(a[i, 1], a[i, 0]))
        angles.append(ang)
    angles = np.array(angles)
    angles = angles[:, np.newaxis]
    return angles


def normalize(v):
    norm = np.linalg.norm(v, ord=2)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    return v/norm

def main():
    trails = glob.glob(path.join('inputs', 'input*.txt'))
    trails = sorted(trails)
    frames = np.load('frames.npy')

    # Obtain new coordinates and velocity with respect to new axis and
    # add new angle between pen and drawing direction(modified_phi)
    # line: x, y, vx, vy, presure, phi, theta
    # target output: new_x, new_y, new_vx, new_vy, pressure, phi, theta, new_phi
    for i, trail in enumerate(trails):
        with open(trail) as f:
            lines = f.readlines()
            coord = [[float(row.split(' ')[0]), float(row.split(' ')[1])] for row in lines]
            coord = np.array(coord)
            velocity = [[float(row.split(' ')[2]), float(row.split(' ')[3])] for row in lines]
            velocity = np.array(velocity)
            pressure_angles = [[float(row.split(' ')[4]), float(row.split(' ')[5]), float(row.split(' ')[6])] for row in lines]
            pressure_angles = np.array(pressure_angles)
            # adapt coordinate and velocity to the new coordinates and add new angle
            frame = frames[i, :, :]
            axis = [normalize(frame[1, :] - frame[0, :]),
            normalize(frame[2, :] - frame[0, :])]
            axis = np.array(axis)
            axis = axis.T
            coord = np.dot(coord - frame[0:1, :], axis)
            angles = getAngle(velocity, pressure_angles[:, -2])
            velocity = np.dot(velocity, axis)
            modified_input = np.concatenate((coord, velocity, pressure_angles, angles), axis=1)
        np.save('modified_inputs/modified_input{}.npy'.format(i+1), modified_input)

if __name__== "__main__":
    main()
