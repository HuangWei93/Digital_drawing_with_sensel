#!/usr/bin/env python

##########################################################################
# MIT License
#
# Copyright (c) 2013-2017 Sensel, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the "Software"), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons
# to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
##########################################################################
import sys
sys.path.append('../../../sensel-lib-wrappers/sensel-lib-python')
import sensel
import binascii
from ctypes import *
import numpy as np
import time
import serial

#Gyro part:
def cal_readline():
        try:
            return ser.readline().decode('UTF-8').rstrip()
        except:
            return None

def readline(initial_phi):
    try:
        line = ser.readline().decode('UTF-8').rstrip()
        if line and line.startswith('yprg'):
            parts = line.split('\t')
            if len(parts) == 5:
                phi, _, _, theta = tuple([float(x) for x in parts[1:]])
                phi = phi - initial_phi
                if phi > 180:
                    phi = phi - 360
                elif phi < -180:
                    phi = phi + 360
                else:
                    pass
                
        return (phi, theta)
    except:
        return None

   
#Sensel API part:
def openSensel():
    handle = None
    (error, device_list) = sensel.getDeviceList()
    if device_list.num_devices != 0:
        (error, handle) = sensel.openDeviceByID(device_list.devices[0].idx)
        error = sensel.setDynamicBaselineEnabled(handle, 0)
    return handle

def initFrame():
    error = sensel.setFrameContent(handle, sensel.FRAME_CONTENT_CONTACTS_MASK)
    error = sensel.setContactsMask(handle, sensel.CONTACT_MASK_ELLIPSE|sensel.CONTACT_MASK_DELTAS|sensel.CONTACT_MASK_PEAK)
    (error, frame) = sensel.allocateFrameData(handle)
    error = sensel.startScanning(handle)
    return frame

def scanFrames(frame, info):
    error = sensel.readSensor(handle)
    (error, num_frames) = sensel.getNumAvailableFrames(handle)
    for i in range(num_frames):
        error = sensel.getFrame(handle, frame)
        printFrame(frame,info)

def printFrame(frame, info):
    line = readline(initial_phi)
    if frame.n_contacts > 0:
        print('Contacts:\n')
        for n in range(frame.n_contacts):
            c = frame.contacts[n]
            if True:
                msg1 = "%.2f %.2f %.2f %.2f %.2f" % (c.x_pos + 3.5, c.y_pos + 3.0, c.delta_x, c.delta_y, c.peak_force) 
                if line != None:
                    msg2 = "%.2f %.2f" % (line[0], line[1])
                else:
                    msg2 = " "
                msg = msg1 + " " + msg2 + '\n'
                print(msg)
def closeSensel(frame):
    error = sensel.freeFrameData(handle, frame)
    error = sensel.stopScanning(handle)
    error = sensel.close(handle)

if __name__ == "__main__":
    #Preparing reading data from Gyro
    connected = False
    locations = ['/dev/cu.wchusbserial1430', '/dev/tty.wchusbserial1430']
    ser: serial.Serial = None
    for device in locations:
        try:
            print("Trying...", device)
            ser = serial.Serial(device, 115200)
            break
        except:
            print("Failed to connect on", device)

    if not ser:
        raise Exception('Serial not found')
    initial_phi = 0.0
    for i in range(5000):
        line = cal_readline()
        if line and line.startswith('yprg'):
            parts = line.split('\t')
            if len(parts) == 5:
                phi, _, _, _ = tuple([float(x) for x in parts[1:]])
                initial_phi = initial_phi*0.1 + phi *0.9
                print('Calibration, please do not move......')           
    print('Gyro calibration Done')
    print('Detecting Sensel Morph')
    handle = openSensel() 
    if handle != None:
        (error, info) = sensel.getSensorInfo(handle)
        frame = initFrame()
        while(True):
            scanFrames(frame, info)
        closeSensel(frame)
    
