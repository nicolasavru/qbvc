#!/usr/bin/python2

import cv2
import numpy as np
import bottleneck as bn
from itertools import tee, izip

# https://bbs.archlinux.org/viewtopic.php?id=157992
# http://code.opencv.org/issues/2211
# cv2.namedWindow("im1", cv2.CV_WINDOW_AUTOSIZE)

import matplotlib.pyplot as plt

def GenerateSignature(fname, demo=False):
    """
    Returns [shotlen_sig, colorshift_sig, centroid_cig] of fname.
    """
    vid = cv2.VideoCapture(fname)
    #vid.set(cv2.cv.CV_CAP_PROP_CONVERT_RGB,1)

    h = []
    cb = []
    cd = []

    xres = vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    yres = vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)

    bins = np.linspace(0,255,num=17)[1:]
    downsample_number = 1
    secs = 0
    while True:

        # Can downsample the frames to only comput signature once every k frames
        for i in range(0,downsample_number):
            ret,im = vid.read()
            if not ret:
                break
        if not ret:
            break

        if ((len(h)+1) % 30 == 0):
            secs += 1
            s = '...Processed ' + str(secs) + ' seconds of video'
            print(s)
        frame = im

        h.append(histogram(frame, bins))

        frame = np.sum(frame, axis=2)

        if demo and len(h) > 100:
            #plt.ion()
            width = 10
            pic_frame = cv2.cvtColor(im,cv2.cv.CV_BGR2RGB)
            f, (ax1) = plt.subplots(1,1)
            ax1.imshow(pic_frame)
            ax1.set_title('Video Frame')
            ax1.figure.canvas.draw() 
            f, (ax1, ax2, ax3) = plt.subplots(1,3, sharey=True)
            ax1.bar(bins, h[-1][2], width=width, color='r')
            ax1.set_title('Red Color Histogram')
            ax1.set_xlabel('Red Value Bins')
            ax1.set_ylabel('Pixel Nrmber')
            ax1.figure.canvas.draw()
            ax2.bar(bins, h[-1][1], width=width, color='g')
            ax2.set_title('Green Color Histogram')
            ax2.set_xlabel('Green Value Bins')
            ax2.figure.canvas.draw()
            ax3.bar(bins, h[-1][0], width=width, color='b')
            ax3.set_title('Blue Color Histogram')
            ax3.set_xlabel('Blue Value Bins')
            ax3.figure.canvas.draw()
            plt.show()
            demo = False

        n = int(xres*yres*0.95)

        partsorted_frame = np.unravel_index(bn.argpartsort(frame.flatten(),n), [xres, yres])
        cb.append((np.mean(partsorted_frame[0][n:]),
                   np.mean(partsorted_frame[1][n:])))
        cd.append((np.mean(partsorted_frame[0][:int(xres*yres*.05)]),
                   np.mean(partsorted_frame[1][:int(xres*yres*.05)])))


    print("done")

    # calculate Manhattan distance between successive frames
    # possible use numpy.linalg.norm instead
    hist_l1 = np.sum(np.sum(np.absolute(np.diff(zip(*h), axis=1)), axis=2), axis=0)

    # normalize histogram by resolution
    # cast hist_l1n to ints for sig
    hist_l1n = hist_l1 / np.sqrt(xres*yres)

    # 10*mean seems to work well...
    boundaries = np.nonzero(hist_l1n > 10*np.mean(hist_l1n))

    # centroid movements
    cb_m = np.sqrt(np.sum(np.square(np.absolute(np.diff(cb, axis=0))), 1))/np.sqrt(xres*yres)
    cd_m = np.sqrt(np.sum(np.square(np.absolute(np.diff(cd, axis=0))), 1))/np.sqrt(xres*yres)
    c_m = cb_m + cd_m

    # t1 = np.arange(len(hist_l1n))/23.967
    # t2 = np.arange(len(c_m))/23.967
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(t1, hist_l1n)
    # plt.subplot(212)
    # plt.plot(t2, c_m)
    # plt.show()

    # for i in range(frames.shape[-1]):
    #     cv2.imshow('im1', frames[...,i])
    #     cv2.waitKey(20)

    vid.release()
    return [boundaries, hist_l1n, c_m]
    # print(frames.shape)

def histogram(frame, bins):
    """
    Returns the historgram of the 3 channels of the frame
    """
    # np.histogram is slow
    # return ([np.histogram(frame[...,0], bins=16, range=[0, 255])[0],
    #          np.histogram(frame[...,1], bins=16, range=[0, 255])[0],
    #          np.histogram(frame[...,2], bins=16, range=[0, 255])[0]])

    return ([np.bincount(np.digitize(frame[...,0].flatten(),
        bins, right=True),minlength=16),
        np.bincount(np.digitize(frame[...,1].flatten(),
            bins, right=True), minlength=16),
        np.bincount(np.digitize(frame[...,2].flatten(),
            bins, right=True), minlength=16)])


# CombineSignature - Combines the colorshift and centroid to product 1 signature
# colorshift_sig: An array of floats depicting the shifts in color in successive frames
# centroid_sig: An array of floats depicting the centroid of each frame
# Returns a comma delimited string of the combined signature interweaving 2 inputs
def CombineSignature(colorshift_sig, centroid_sig):
    """
    Returns the np arrays colorshift_sig and centroid_sig interleaved.
    """
    # TODO: strip off brackets, etc.
    combined_result = [None]*(len(colorshift_sig)+len(centroid_sig))
    combined_result[::2] = colorshift_sig.tolist()
    combined_result[1::2] = centroid_sig.tolist()
    return ','.join(map(str,combined_result))

def CostFunction(weighted_difference):
    """
    Computes the cost of the weighted difference
    """
    return weighted_difference

def SlideWindow(a, stepsize=1, width=3):
    #shape = a.shape[:-1] + (a.shape[-1] - width + 1, width)
    #strides = a.strides + (a.strides[-1],)
    #return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
    shape = a.shape[:-1] + (a.shape[-1] - width + 1, width)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def CompareSignature(sig1, sig2, cost_func=CostFunction, demo=False):
    """
    Compares sig1 and sig2. Returns True of they match and False otherwise.
    """
    window_size_factor = 1

    db_vid_sig = np.asarray([float(n) for n in sig2.split(',')])
    query_vid_sig = np.asarray([float(n) for n in sig1.split(',')])
    
    #If window_size_factor can produce non multiple of 2 in window size must fix this
    window_size = int(len(query_vid_sig)*window_size_factor)

    k = 1
    n = 0

    costs = [0] * len(db_vid_sig)
    cost_tuple = [(0,0)] * (len(db_vid_sig)/2)
    min_index = 0
    min_cost = -1

    cur_sum_costs = 0

    windows = SlideWindow(db_vid_sig, 2, window_size)

    for window in windows:
        window = np.asarray(window)
        costs[n] = np.sum(CostFunction(k*np.absolute(np.subtract(window,query_vid_sig))))
        if costs[n] < min_cost or min_cost == -1:
            min_cost = costs[n]
            min_index = n
            if demo:
                print n, costs[n]
        n += 1

    if demo:
        print "Best Match Determined to be at", round(float(min_index/2)/29.97), "seconds into video"
        
    return costs

