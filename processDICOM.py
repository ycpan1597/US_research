#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 13:57:37 2019

@author: preston
"""
import pydicom as dicom
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
import platform
#%%
# Read a specific frame of the dicom file
def readFrame(frameNumber, data, dims = (600, 800), maxFrameNum = 119, constBorder = 200, process = True):
    frame = np.zeros((dims[0], dims[1], 3), np.uint8)
    if frameNumber > maxFrameNum:
        raise Exception ('{} exceeded the maximum number of frames'.format(frameNumber))
    if frameNumber == 0:
        raise Exception ('Frames start from 1')
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
    frame = cv2.copyMakeBorder(frame, constBorder, constBorder, constBorder, constBorder, cv2.BORDER_CONSTANT) # expand the frame to accommodate additional transformation
    
    if process: 
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                         # convert to grayscale (1-channel)
        ret, mask = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)            # apply a mask to increase image contrast
        kernel = np.ones((5, 5), np.uint8)   
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel, iterations = 3) # closing - removes false negative                                   # kernal size can be further optimized        
        blurred = cv2.GaussianBlur(closed, (3, 3), 20, 20)
        
        return blurred
    else: 
        return frame

def translate(src, toRight, toBottom):
    M = np.float32([[1, 0, toRight], [0, 1, toBottom]])
    dst = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))
    return dst

def rotate(src, angle):
    rows, cols = src.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)                     # rotate about the center of the image
    rotated = cv2.warpAffine(src, M, (cols, rows))
    return rotated

def transform(src, toRight, toBottom, angle):                                   # includes translation and rotation
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
# until it matches the reference image. Supports dynamic range adjustment to ensure best solution
def shiftRotateCompare(ref, src, horRange = [-5, 5], verRange = [-5, 5], rotRange = [-10, 0], seeAlignment = False):
    
    # subfunction that checks if the horizontal/vertical/rotational shift ranges should be extended
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
    
    # option to view alignment
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

# calculates the size of the enclosing circle based on the largest contour in multiple frames
def checkContourArea(startFrameNum, endFrameNum, spacing, dsData):
    refFrames = np.arange(startFrameNum, endFrameNum, spacing)
    contourInfo = []
    for index, frame in enumerate(refFrames):
        A = readFrame(frame, dsData)
        cnts, hierarchy = cv2.findContours(A, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 0:
            print('frame {} does not have any contours'.format(frame))
            contourInfo.append((refFrames[index], None))
        else :
            c = max(cnts, key = cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            contourInfo.append((refFrames[index], radius, (x, y)))
    return contourInfo

# calculates two points that determine a line in an image given the vx/vy (unit vector) and x0/y0 (position)
def linePoints(vx, vy, x0, y0, img):
    vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]
    dim = img.shape[:2]
    if vx == 0: # vertical line with infinite slope
        left = (x0, 0)
        right = (x0, dim[0])
    elif vy == 0: # horizontal line with m = 0
        left = (0, y0)
        right = (dim[1], y0)
    else: 
        m = vy / vx
        y = lambda x : m * (x - x0) + y0
        left = (-10, int(y(-10))) # -10 guarantees that the line is drawn from one end of the image to another
        right = (max(dim), int(y(max(dim))))
    return left, right

# modified version of shiftRotateCompare with a specified amount of rotation
# does not support dynamic range adjustment
def shiftCompare(ref, src, horRange, verRange, rot, visualize = False):
    start = time.time()
    horLow, horHigh = horRange[0], horRange[1]
    verLow, verHigh = verRange[0], verRange[1]
    horShift, verShift = 0, 0
    maxCorr = 0
    src = rotate(src, rot) # Assume a 60 degree rotation first
    best = src
    oldScore = pearsonCompare(ref, src)
    for i in range(horLow, horHigh):
        for j in range(verLow, verHigh):
            print('hor: {}, ver: {}'.format(i, j))
            translated = translate(src, i, j)
            corr = pearsonCompare(ref, translated)
            if corr > maxCorr:
                maxCorr = corr
                horShift = i
                verShift = j
                best = translated      
    print('This function took %.2f seconds' % (time.time() - start))
    print('hor: {}, ver: {}'.format(horShift,verShift))
    print('correlation score: {} --> {}'.format(oldScore, maxCorr))
    result = cv2.add(ref, best)
    if visualize: 
        plt.figure()
        plt.imshow(result)
    return result

#def findSimilarFrames(ref, allFrames):
    

if __name__ == '__main__':
    plt.close('all')
    
    # run on windows ('Windows') or linux ('Darwin')
    if platform.system() == 'Darwin':
        filePath = '/Users/preston/Desktop/US_research/US00001L.dcm'
    else:
        filePath = 'D:\\US_research\\US00001L.dcm'
    
    ds = dicom.dcmread(filePath)
    dims = (int(ds.Rows), int(ds.Columns))
    dsData = ds.PixelData
    bitsPerCell = ds.BitsAllocated
    
    canvas = readFrame(1, dsData) # use the first frame as the base
    
    frameA = 27 # the first time top is horizontal
    frameB = 64 # the second time top is horizontal
    frameC = 99 # the third time top is horizontal
    
    A, rgbA = readFrame(frameA, dsData), readFrame(frameA, dsData, process = False) # rgbA is a 3-channel(RGB) image that can be overlaid with the contour
    B, rgbB = readFrame(frameB, dsData), readFrame(frameB, dsData, process = False)
    C, rgbC = readFrame(frameC, dsData), readFrame(frameC, dsData, process = False)
    
#    plt.figure(figsize = (8, 12))
#    plt.subplot(3, 2, 1)
#    plt.imshow(A)
#    plt.title('component 1 ({}th)'.format(frameA))
#    plt.subplot(3, 2, 2)
#    plt.imshow(A)
#    plt.title('rotated 0 degrees')
#    plt.subplot(3, 2, 3)
#    plt.imshow(B)
#    plt.title('component 2 ({}th)'.format(frameB))
#    plt.subplot(3, 2, 4)
#    plt.imshow(rotate(B, -60))
#    plt.title('rotated -60 degrees')
#    plt.subplot(3, 2, 5)
#    plt.imshow(C)
#    plt.title('component 3 ({}th)'.format(frameC))
#    plt.subplot(3, 2, 6)
#    plt.imshow(rotate(C, -120))
#    plt.title('rotated -120 degrees')
#    plt.subplots_adjust(hspace = 0.35)
#    
#    abMatch = shiftCompare(A, B, (97, 103), (-5, 0), -60)                       # solution: hor: 101, ver: -3
#    abcMatch = shiftCompare(abMatch, C, (160, 165), (48, 53), -120)             # solution: hor: 162, ver: 51
#    
#    plt.figure(figsize = (4, 12))
#    plt.subplot(3, 1, 1)
#    plt.imshow(A)
#    plt.title('component 1 (base)')
#    plt.subplot(3, 1, 2)
#    plt.imshow(abMatch)
#    plt.title('component 1 + 2')
#    plt.subplot(3, 1, 3)
#    plt.imshow(abcMatch)
#    plt.title('component 1 + 2 + 3')
#    plt.subplots_adjust(hspace = 0.35)
#    
#    cnts, hierarchy = cv2.findContours(abcMatch, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#    c = max(cnts, key = cv2.contourArea)
#    withContours = cv2.drawContours(rgbA, [c], 0, (255, 0, 0), 1)
#    ((x, y), radius) = cv2.minEnclosingCircle(c)
#    cv2.circle(withContours, (int(x), int(y)), int(radius), (0, 255, 255), 2)
#    plt.figure(figsize = (12, 12))
#    plt.imshow(withContours)
#    plt.title('diameter: %.2f' % (radius * 2))

## find centroid
#M = cv2.moments(c)
#centroid = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

## find and draw a line that fits through one contour
#vx,vy,x,y = cv2.fitLine(c,cv2.DIST_L2, 0, 0.01, 0.01)   # the last two 0.01 are recommended accuracy values
#twoPoints = linePoints(vx, vy, x, y, rgbA)
#plt.figure()
#plt.imshow(cv2.line(rgbA, twoPoints[0], twoPoints[1], (255, 0, 0)))
#%%
# Using cv2.HoughLines to detect straight edges
plt.close('all')
edges = cv2.Canny(A, 50, 150, apertureSize = 3)
lines = cv2.HoughLines(edges, 1, np.pi/180, 105)
withLines = edges.copy()
for sublist in lines:
    r = sublist[0][0]
    theta = sublist[0][1]
    a = np.cos(theta)
    b = np.sin(theta)
    
    # found this section online; not sure why this works
    x0 = a * r
    y0 = b * r
    x1 = int(x0 + 2000 * (-b))
    y1 = int(y0 + 2000 * (a))
    x2 = int(x0 - 2000 * (-b))
    y2 = int(y0 - 2000 * (a))
    
    rgbA = cv2.line(rgbA, (x1, y1), (x2, y2), (255, 0, 0), 3)
plt.figure()
plt.imshow(edges)
plt.figure()
plt.imshow(rgbA)



        
