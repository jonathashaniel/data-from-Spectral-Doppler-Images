# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 11:29:09 2019
@author: Jonathas
get data .csv from a  spectral doppler ultrasound image with a red average aproximation
"""



PSV = 75.20 #value of first peak
PERIOD = False # inform the correct period 



""""""""""""""""""""""""""""""""""""""""""""""""""""""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import find_peaks
plt.close('all')

img = cv2.imread('14-22-43.png')
plt.figure()
imgplot = plt.imshow(img)

def find_zero(img):
    zer=np.zeros(img.shape[0])
    for x in range(img.shape[1]):
        for y in range(img.shape[0]):
            if img[y,x,0] == 200 and img[y,x,1] == 200 and img[y,x,2] == 200:
                zer[y] = zer[y]+ 1;
    zer = np.where(zer == np.amax(zer))
    return zer[0]

def create_signal(img, crop_x = 600): # to crop image in x
    signal=np.zeros(crop_x)
    zero = find_zero(img)
    crop_y = img.shape[0]
    for y in range(crop_y):
        for x in range(crop_x):
            if img[y,x,0] == 0 and img[y,x,1] == 0 and img[y,x,2] == 255:
                signal[x] = zero-y
    return signal

def evaluate(signal,max_value):
    peaks , _ = find_peaks(signal, height = 50)
    signal = max_value * signal/signal[peaks[0]]

    return signal
    
def crop_signal(signal):
    peaks , _ = find_peaks(signal, height = 50)
    signal = signal[peaks[0]:peaks[-1]]
    peaks , _ = find_peaks(signal, height = 1)
    peaks = np.insert(peaks,0,0)
    peaks = np.append(peaks, signal.size - 1)
    return signal, peaks

signal = create_signal(img)
signal= evaluate(signal,PSV)
signal , peaks= crop_signal(signal)



np.savetxt("signal.csv", signal, delimiter=",")
np.savetxt("peaks.csv", [peaks, signal[peaks]], delimiter=",")
            
plt.figure()
plt.plot(signal)
plt.plot(peaks, signal[peaks], "x")


