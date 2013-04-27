#!/usr/bin/python2

import cv2
import numpy as np

# https://bbs.archlinux.org/viewtopic.php?id=157992
# http://code.opencv.org/issues/2211
# cv2.namedWindow("im1", cv2.CV_WINDOW_AUTOSIZE)

import matplotlib.pyplot as plt

def GenerateSignature(fname):
    """
    Returns [shotlen_sig, colorshift_sig, centroid_cig] of fname.
    """
    vid = cv2.VideoCapture(fname)

    h = []
    cb = []
    cd = []

    bins = np.linspace(0,255,num=17)[1:]
    while True:
        ret,im = vid.read()
        if not ret:
            break

        print(len(h))
        frame = im

        # histogram is slow
        # h.append([np.histogram(frame[...,0], bins=16, range=[0, 255])[0],
        #           np.histogram(frame[...,1], bins=16, range=[0, 255])[0],
        #           np.histogram(frame[...,2], bins=16, range=[0, 255])[0]])

        # ~35% speed increase
        h.append([np.bincount(np.digitize(frame[...,0].flatten(),
            bins, right=True),minlength=16),
            np.bincount(np.digitize(frame[...,1].flatten(),
                bins, right=True), minlength=16),
            np.bincount(np.digitize(frame[...,2].flatten(),
                bins, right=True), minlength=16)])

        frame = np.sum(frame, axis=2)

        # TOFIX: these are the 5% brightest pixel values, not 5% brightest pixels
        cb.append(np.mean(np.nonzero(frame > 0.95*255), axis=1))
        cd.append(np.mean(np.nonzero(frame < 0.05*255), axis=1))


    print("done")

    # calculate Manhattan distance between successive frames
    # possible use numpy.linalg.norm instead
    hist_l1 = np.sum(np.sum(np.absolute(np.diff(zip(*h), axis=1)), axis=2), axis=0)

    # normalize histogram by resolution
    # cast hist_l1n to ints for sig
    xres = vid.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
    yres = vid.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
    hist_l1n = hist_l1 / np.sqrt(xres*yres)

    # 10*mean seems to work well...
    boundaries = np.nonzero(hist_l1n > 10*np.mean(hist_l1n))

    # centroid movements
    cb_m = np.sqrt(np.sum(np.square(np.absolute(np.diff(cb, axis=0))), 1))/np.sqrt(xres*yres)
    cd_m = np.sqrt(np.sum(np.square(np.absolute(np.diff(cd, axis=0))), 1))/np.sqrt(xres*yres)

    c_m = cb_m + cd_m

    return [boundaries, hist_l1n, c_m]

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

    # print(frames.shape)

def CombineSignature(colorshift_sig, centroid_sig):
    """
    Returns the np arrays colorshift_sig and centroid_sig interleaved.
    """
    # TODO: strip off brackets, etc.
    return ''.join([''.join(c).strip() for c in zip(str(colorshift_sig.tolist()),
                                             str(centroid_sig.tolist()))])


def CompareSignature(sig1, sig2):
    """
    Compares sig1 and sig2. Returns True of they match and False otherwise.
    """
    return True
