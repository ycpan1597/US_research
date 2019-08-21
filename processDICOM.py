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
import collections
#%%
# Read a specific frame of the dicom file
def readFrame(frameNumber, data, dims = (600, 800), maxFrameNum = 119, constBorder = 200, process = True, outputShape = 800):
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

    if process:
        final = frame[300:, :600]  # crop away the text
        final = cv2.copyMakeBorder(final, constBorder, constBorder, constBorder, constBorder, cv2.BORDER_CONSTANT)  # expand the frame to accommodate additional transformation
        final = cv2.cvtColor(final, cv2.COLOR_BGR2GRAY)                         # convert to grayscale (1-channel)
        ret, final = cv2.threshold(final, 35, 255, cv2.THRESH_BINARY)            # apply a mask to increase image contrast
        kernel = np.ones((5, 5), np.uint8)                                      # kernal size can be further optimized
        final = cv2.morphologyEx(final, cv2.MORPH_CLOSE,kernel, iterations = 3)  # closing - removes false negative
        final = cv2.GaussianBlur(final, (3, 3), 20, 20)
        final = cv2.fastNlMeansDenoising(final, h = 100)                   # h is the strength of the denoising filter
        x, y, radius = findEnclosingCircle(final)
        final = final[(int(y) - int(radius)) : (int(y) + int(radius)), (int(x) - int(radius)): (int(x) + int(radius))]
        borderWidth = int((outputShape - 2 * radius) / 2)
        final = cv2.copyMakeBorder(final, borderWidth, borderWidth, borderWidth, borderWidth, cv2.BORDER_CONSTANT)
        # final = cv2.circle(final, (int(final.shape[0]/2), int(final.shape[1]/2)), int(radius), (255, 255, 255))
        return final
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

# ref: reference image frame number, src: source image frame number
# This function shifts and rotates the source image within the give range of parameter
# until it matches the reference image. Supports dynamic range adjustment to ensure best solution

# Dynamic range adjustment might not be the best option in achieving the best image reconstruction.
# Instead, the reconstruction might actually cause subsequent frames to fit exactly to the reference
# as opposed to adding onto the reference frame

# Maybe it's better to remove translation before additional processing
def shiftRotateCompare(ref, src, horRange = [-5, 5], verRange = [-5, 5], rotRange = [-20, 0], seeAlignment = False, dynamic = True):
    # ref = readFrame(ref, dsData)
    # src = readFrame(src, dsData)
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
                    # print('hor: {}, ver: {}, rot: {}'.format(i, j, k))
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
        if np.all(check == 0) or not dynamic:
            done = True
            # print(horLow, horHigh, verLow, verHigh, rotLow, rotHigh)
        else:
            # print('At least 1 of the shifts reached its extrema: hor = {}, ver = {}, rot = {}'.format(horShift, verShift, rotShift))
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
        plt.figure(figsize = (8, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.add(ref, src))
        plt.title('Before, score = %.0f' % pearsonCompare(ref, src))
        plt.subplot(1, 3, 2)
        plt.imshow(both)
        plt.title('After, score = %.0f\n hor: %.0f, ver: %.0f, rot: %.0f' % (maxCorr, horShift, verShift, rotShift))
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.bitwise_xor(ref, best))
        plt.title('ref xor best')
        # plt.subplots_adjust(hspace = 0.4)

    print(horShift, verShift, rotShift)
    print('%.2f s' % (time.time() - start))

    return cv2.bitwise_xor(ref, best)
    # return best

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


# Simplified version of shiftRotateCompare: rotate and find the best match between two frames after removing
# translational offsets
def rotateCompare(ref, src, rotRange = [-30, 0]):
    rotLow = rotRange[0]
    rotHigh = rotRange[1]
    maxCorr = 0
    bestDeg = 0
    result = src
    for deg in range(rotLow, rotHigh):
        rotated = rotate(src, deg)
        corr = pearsonCompare(ref, rotated)
        if corr > maxCorr:
            maxCorr = corr
            bestDeg = deg
            result = rotated
    return result, bestDeg


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


def findRotationForAll(imgList, ref = 2):
    allRot = []
    for index, src in enumerate(imgList):
        _, shifts = shiftRotateCompare(ref, src)
        allRot.append(shifts)
        print('{}% done'.format((index + 1) * 100 / len(imgList)))
    return allRot


def findEnclosingCircle(img):
    image, cnts, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key=cv2.contourArea)
    ((x, y), radius) = cv2.minEnclosingCircle(c)
    return x, y, radius

# This removes the offset!
def removeOffsets(ref, imgList):
    refX, refY, radius = findEnclosingCircle(ref)
    newImgList = []
    for img in imgList:
        imgX, imgY, radius = findEnclosingCircle(img)
        newImg = translate(img, refX - imgX, refY - imgY)
        newImgList.append(newImg)
    return newImgList

def drawOneEnclosingCircle(img):
    x, y, rad = findEnclosingCircle(img)
    # canvas = np.empty((img.shape[0], img.shape[1], 3), dtype=np.uint8) * 0
    output = cv2.circle(img, (int(x), int(y)), int(rad), (255, 255, 255), 2)
    plt.figure()
    plt.imshow(output)
    plt.plot(x, y, 'ro')
    return output

def drawTwoEnclosingCircles(img1, img2):
    canvas = np.empty((img1.shape[0], img1.shape[1], 3), dtype=np.uint8) * 0
    img1X, img1Y, radius1 = findEnclosingCircle(img1)
    img2X, img2Y, radius2 = findEnclosingCircle(img2)
    canvas = cv2.circle(canvas, (int(img1X), int(img1Y)), int(radius1), (0, 255, 255), 2)
    canvas = cv2.circle(canvas, (int(img2X), int(img2Y)), int(radius2), (225, 255, 0), 2)
    plt.figure()
    plt.imshow(canvas)
    plt.plot(img1X, img1Y, 'ro')
    plt.plot(img2X, img2Y, 'go')
    return canvas


if __name__ == '__main__':
    plt.close('all')
    print('running main')
    # run on windows ('Windows') or linux ('Darwin')
    if platform.system() == 'Darwin':
        filePath = '/Users/preston/Desktop/US_research/US00001L.dcm'
    else:
        filePath = 'D:\\US_research\\US00001L.dcm'
    
    ds = dicom.dcmread(filePath)
    dims = (int(ds.Rows), int(ds.Columns))
    dsData = ds.PixelData
    bitsPerCell = ds.BitsAllocated

    start = time.time()
    result = readFrame(20, dsData)
    for i in range(20, 30):
        result = cv2.add(result, shiftRotateCompare(readFrame(i, dsData), readFrame(i + 1, dsData), horRange = [-50, 50], verRange = [-50, 50], rotRange = [-30, 0], dynamic = False))
    print(time.time() - start + 's')

    # canvas = readFrame(1, dsData) # use the first frame as the base
    #
    # frameA = 27 # the first time top is horizontal
    # frameB = 64 # the second time top is horizontal
    # frameC = 99 # the third time top is horizontal
    #
    # A, rgbA = readFrame(frameA, dsData), readFrame(frameA, dsData, process = False) # rgbA is a 3-channel(RGB) image that can be overlaid with the contour
    # B, rgbB = readFrame(frameB, dsData), readFrame(frameB, dsData, process = False)
    # C, rgbC = readFrame(frameC, dsData), readFrame(frameC, dsData, process = False)
    #
    # # print(findRotationForAll(range(2, 27), 2)) # this step takes forever to run, so don't run it again unless absolultely necessary
    #


    # ref = 2 # Frame of reference
    # topBound = 50 # Total number of frames
    #
    # # # Loads all images
    # imgList = []
    # for i in range(1, topBound):
    #     imgList.append(readFrame(i, dsData))
    # imgList = np.array(imgList)
    #
    # # Try matching after removing translational offsets
    # # Reference does not match with the source very well. Maybe we don't want a match; we just want parallel?
    # result = shiftRotateCompare(imgList[ref], imgList[30])
    # plt.figure()
    # plt.imshow(cv2.add(imgList[ref], result))

    # drawTwoEnclosingCircles(imgList[ref], rotate(result, -150))


    # # Read in shift values
    # horShifts = []
    # verShifts = []
    # import csv
    # with open('shiftvalues.csv') as csvfile:
    #     reader = csv.reader(csvfile, delimiter=',')
    #     for i, row in enumerate(reader):
    #         if i > 0:
    #             horShifts.append(int(row[1]))
    #             verShifts.append(int(row[2]))

    # the following set of images is only translated to match the original, not rotated
    # Translating the images before matching did not significantly reduce the runtime
    # imageSet = []
    # for image, hor, ver in zip(ogImageSet, horShifts, verShifts):
    #     imageSet.append(translate(image, hor, ver))
    # imageSet = np.array(imageSet)

    # output = ogImageSet[0]
    # for i in range(1, len(ogImageSet)):
    #     output = cv2.add(output, shiftRotateCompare(ogImageSet[i - 1], ogImageSet[i])[0])
    # output = imageSet[0]
    # for i in range(1, len(ogImageSet)):
    #     output = cv2.add(output, shiftRotateCompare(output, imageSet[i])[0])


    # Trying to recover the original hexagon with the rotation angle of each frame relative to a reference
    # I'm interested in seeing the actual setup used to get this footage.
    # I feel like this is indeed a circular panorama

# This is my working but less preferred method
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
#     plt.close('all')
#     edges = cv2.Canny(A, 50, 150, apertureSize = 3)
#     lines = cv2.HoughLines(edges, 1, np.pi/180, 105)
#     withLines = edges.copy()
#     for sublist in lines:
#         r = sublist[0][0]
#         theta = sublist[0][1]
#         a = np.cos(theta)
#         b = np.sin(theta)
#
#         # found this section online; not sure why this works
#         x0 = a * r
#         y0 = b * r
#         x1 = int(x0 + 2000 * (-b))
#         y1 = int(y0 + 2000 * (a))
#         x2 = int(x0 - 2000 * (-b))
#         y2 = int(y0 - 2000 * (a))
#
#         rgbA = cv2.line(rgbA, (x1, y1), (x2, y2), (255, 0, 0), 3)
#     plt.figure()
#     plt.imshow(edges)
#     plt.figure()
#     plt.imshow(rgbA)
#     plt.show()
