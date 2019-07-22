#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:57:37 2019

@author: preston
"""
# Purpose: successfully read ultrasound scan in dicom with python's pydicom library
 
import pydicom as dicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import platform

def readFrame(frameNumber, data, dims = (600, 800), frames = 119, constBorder = 100, process = True):
    frame = np.zeros((dims[0], dims[1], 3), np.uint8)
    if frameNumber > frames:
        raise Exception ('{} exceeded the maximum number of frames'.format(frameNumber))
    else:
        # group 3 data entries into 1 pixel
        # each frame will take 600 * 800 * 3 = 1440000 data points)
        frameData = data[dims[0] * dims[1] * (frameNumber - 1) * 3 : dims[0] * dims[1] * frameNumber * 3]
        
        frameData = [i for i in frameData]
    
        row = 0
        col = 0
        
        for i in np.arange(0, len(frameData), 3):
            if col >= 800:
                row += 1
                col = 0
            frame[row, col, :] = frameData[i : i + 3]
            col += 1
    frame = frame[300:, :600]                                                   # crop to see only the loop
    frame = cv2.copyMakeBorder(frame, constBorder, constBorder, constBorder, constBorder, cv2.BORDER_CONSTANT)
    
    if process: 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                         # convert to grayscale (1-channel)
#        frame = cv2.GaussianBlur(frame, (3, 3), 1, 1)
        ret, mask = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)            # apply a mask to increase image contrast
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel, iterations = 3) # closing - removes false negative
        return closed
    else: 
        return frame

def translate(src, toRight, toBottom):
    M = np.float32([[1, 0, toRight], [0, 1, toBottom]])
    dst = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))
    return dst

def rotate(src, angle):
    rows, cols = src.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)                 # rotate about the center of the image
    rotated = cv2.warpAffine(src, M, (cols, rows))
    return rotated

def transform(src, toRight, toBottom, angle):                               # includes translation and rotation
	return rotate(translate(src, toRight, toBottom), angle)

# Pixel-by-pixel comparison to determine similarity of two images
# (obsolete - unable to account for standard deviation)
def pixelCompare(imgA, imgB):
    if len(imgA.shape) == 3 or len(imgB.shape) == 3:
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    if imgA.shape[:2] != imgB.shape[:2]:
        raise Exception('image sizes are not the same. Please resize them before finding the correlation')
    corr = np.sum(np.multiply(imgA, imgB))
    return corr

# Similarity of two images based on the Pearson Correlation Coefficient
def pearsonCompare(imgA, imgB):
    if len(imgA.shape) == 3 or len(imgB.shape) == 3:
        imgA = cv2.cvtColor(imgA, cv2.COLOR_BGR2GRAY)
        imgB = cv2.cvtColor(imgB, cv2.COLOR_BGR2GRAY)
    if imgA.shape[:2] != imgB.shape[:2]:
        raise Exception('image sizes are not the same. Please resize them before finding the correlation')
    
    meanA, meanB = np.mean(imgA), np.mean(imgB)
    stdA, stdB = np.std(imgA), np.std(imgB)
    corv = np.sum(np.multiply((imgA - meanA), (imgB - meanB)))
    
    return corv / (stdA * stdB)

# ref: reference image, src: source
# This function shifts and rotates the source image within the give range of parameter
# until it matches the reference image

def shiftCompare(ref, src, horRange = [-5, 5], verRange = [-5, 5], rotRange = [-10, 0], seeAlignment = False):
    
    def checkValues(hor, ver, rot, hl, hh, vl, vh, rl, rh):
        result = np.empty([1, 3], dtype = int)
        result = result[0]
        if hor == (hh - 1):
            result[0] = 1
        elif hor == hl:
            result[0] = -1
        else:
            result[0] = 0
        
        if ver == (vh - 1):
            result[1] = 1
        elif ver == vl:
            result[1] = -1
        else:
            result[1] = 0
            
        if rot == (rh - 1):
            result[2] = 1
        elif rot == rl:
            result[2] = -1
        else:
            result[2] = 0

        return np.array(result)
    
    start = time.time()
    
    horShift, verShift, rotShift = 0, 0, 0
    horAdj, verAdj, rotAdj = 0, 0, 0
    
    maxCorr = 0
    best = src
    done = False
    
    horLow, horHigh = horRange[0], horRange[1]
    verLow, verHigh = verRange[0], verRange[1]
    rotLow, rotHigh = rotRange[0], rotRange[1]
    
    scores = []
    
    while (not done):
        for i in range(horLow, horHigh):
            for j in range(verLow, verHigh):
                for k in range(rotLow, rotHigh):
                    print('hor: {}, ver: {}, rot: {}'.format(i, j, k))
                    transformed = transform(src, i, j, k)
                    corr = pearsonCompare(ref, transformed)
                    if corr > maxCorr:
                        maxCorr = corr
                        horShift = i
                        verShift = j
                        rotShift = k
                        best = transformed
        # Check if any of the shifts and angles were at the max/min of the range                
        check = checkValues(horShift, verShift, rotShift, horLow, horHigh, verLow, verHigh, rotLow, rotHigh)
        scores.append(maxCorr)
        if np.all(check == 0):
            done = True
            print(horLow, horHigh, verLow, verHigh, rotLow, rotHigh)
        else: 
            print('At least 1 of the shifts reached its extrema: hor = {}, ver = {}, rot = {}'.format(horShift, verShift, rotShift))
            horAdj, verAdj, rotAdj = np.multiply(check, np.array([horRange[1] - horRange[0], verRange[1] - verRange[0], rotRange[1] - rotRange[0]]))                      
            
            horLow += horAdj
            horHigh += horAdj
            
            verLow += verAdj
            verHigh += verAdj
            
            rotLow += rotAdj
            rotHigh += rotAdj
            
    both = cv2.add(ref, best)
    
    if seeAlignment:
        plt.figure(figsize = (5, 11))
        plt.subplot(3, 1, 1)
        plt.imshow(cv2.add(ref, src))
        plt.title('Before, score = %.0f' % pearsonCompare(ref, src))
        plt.subplot(3, 1, 2)
        plt.imshow(both)
        plt.title('After, score = %.0f\n hor: %.0f, ver: %.0f, rot: %.0f' % (maxCorr, horShift, verShift, rotShift))
        plt.subplot(3, 1, 3)
        plt.imshow(cv2.bitwise_xor(ref, best))
        plt.title('ref xor best')    
        plt.subplots_adjust(hspace = 0.2)
        
    print('%.2f s' % (time.time() - start))
    
    return cv2.bitwise_xor(ref, best)

def checkContourArea(startFrameNum, endFrameNum, spacing, dsData):
    refFrames = np.arange(startFrameNum, endFrameNum, spacing)
    radii = []
    for index, frame in enumerate(refFrames):
        A = readFrame(frame, dsData)
        cnts, hierarchy = cv2.findContours(A, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            print('frame {} does not have any contours'.format(frame))
            radii.append((refFrame[index], None))
        else :
            c = max(cnts, key = cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            radii.append((refFrames[index], radius, (x, y)))
    return radii

# calculates two points that determine a line in an image given the vx/vy (unit vector) and x0/y0 (position)
def linePoints(vx, vy, x0, y0, img):
    vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]
    dim = img.shape[:2]
    if vx == 0:
        left = (x0, 0)
        right = (x0, dim[0])
    elif vy == 0:
        left = (0, y0)
        right = (dim[1], y0)
    else: 
        m = vy / vx
        y = lambda x : m * (x - x0) + y0
        left = (-10, int(y(-10)))
        right = (max(dim), int(y(max(dim))))
    return left, right

plt.close('all')

if platform.system() == 'Darwin':
    filePath = '/Users/preston/Desktop/US_research/US00001.dcm'
else:
    filePath = 'D:\\US_rsearch\\US00001.dcm'

ds = dicom.dcmread('/US001.dcm') # Need to wait for Oliver to send me some ultrasound dicom
# dir: /Users/preston/Desktop/Image_Processing/US001.dcm
dims = (int(ds.Rows), int(ds.Columns))
dsData = ds.PixelData
bitsPerCell = ds.BitsAllocated

canvas = readFrame(1, dsData) # use the first frame as the base

refFrame = 75
A, pureA = readFrame(refFrame, dsData), readFrame(refFrame, dsData, process = False)
B = readFrame(110, dsData)
#
##comparison = shiftCompare(A, B)
#
cnts, hierarchy = cv2.findContours(A, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
withContours = cv2.drawContours(pureA, cnts, -1, (255, 0, 0), 1) # the third argument (-1) refers to which contour to draw.
                                                                 # to draw all, use -1
c = max(cnts, key = cv2.contourArea)
((x, y), radius) = cv2.minEnclosingCircle(c)


## M = cv2.moments(c)
## centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

cv2.circle(withContours, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                                                                 
plt.figure()
plt.imshow(A)
plt.figure()
plt.imshow(withContours)

vx,vy,x,y = cv2.fitLine(c,cv2.DIST_L2, 0, 0.01, 0.01)   # the last two 0.01 are recommended accuracy values
twoPoints = linePoints(vx, vy, x, y, pureA)
plt.figure()
plt.imshow(cv2.line(pureA, twoPoints[0], twoPoints[1], (255, 0, 0)))



#for i in np.arange(1, 119, 10):
#    prev = readFrame(i, dsData)
#    comparison = shiftCompare(prev, readFrame(i + 1, dsData))
#    canvas = cv2.add(canvas, comparison)
#plt.figure()
#plt.imshow(canvas)
        
#%% Boarder detection
cropped = cv2.cvtColor(frame[100:, :600], cv2.COLOR_RGB2GRAY)
#cropped = frame[100:, :600, :]
edges = cv2.Canny(cropped, 10, 50)
plt.figure()
plt.imshow(edges, 'gray')

ret, mask = cv2.threshold(cropped, 50, 255, cv2.THRESH_BINARY)

kernel = np.ones((6, 6), np.uint8)
closed = cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations = 2)

newEdges = cv2.Canny(closed, 10, 50)

contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
withContours = cv2.drawContours(cropped, contours, -1, (255, 0, 0), 3)
plt.figure()
plt.imshow(withContours)

plt.figure()
plt.imshow(newEdges, 'gray')




        
