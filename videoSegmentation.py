import cv2
import numpy as np
import os

'''
video segmentation
'''

def segVideo(filePath, isViolence):
    videoIn = cv2.VideoCapture(filePath)
    if (videoIn.isOpened() == False):
        print("file open error")
    # constants
    frate = videoIn.get(cv2.CAP_PROP_FPS)   # frame rate
    TOTAL_FRAME = int(videoIn.get(cv2.CAP_PROP_FRAME_COUNT))    # number of frames in given video
    IMG_HEIGHT = 240
    IMG_WIDTH =  320



    SEG_SIZE = 30   # segment size.

    seg = np.empty((TOTAL_FRAME, IMG_HEIGHT, IMG_WIDTH, 3)) # 3 channel data

    if(isViolence==True):
        #  폭력 영상의 label = 1
        label = np.ones((TOTAL_FRAME))
    else:
        #  노멀 영상의 label = 0
        label = np.zeros((TOTAL_FRAME))
    # segment number
    idxSeg = 0

    while videoIn.isOpened():
        ret, frame = videoIn.read()
        curFrame = int(videoIn.get(cv2.CAP_PROP_POS_FRAMES))     # current frame number

        # resize (240, 320). resize함수는 변수 순서가(넓이, 높이)로 정의되어있음.
        frameResized = cv2.resize(frame,(IMG_WIDTH, IMG_HEIGHT))

        if (curFrame == TOTAL_FRAME):
            break
        seg[idxSeg,:,:,:] = (frameResized/255.0) # regularization


    videoIn.release()



    # print("done!")
    return seg, label
