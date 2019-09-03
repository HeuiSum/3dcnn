import cv2
import numpy as np
import os
import tensorflow as tf
import random
import videoSegmentation as vs  # a function for video segmentation
import baseLineCNN_model as C3D
from tensorflow.keras import datasets, layers, models


'''
폭력 영상 loading
'''
# input violence video path
filePathViolence = './Real Life Violence Dataset/Violence'
fileListViolence = os.listdir(filePathViolence)
numberOfFilesViolence = len(fileListViolence)
videoSquenceViolence =[]

# 폭력 영상 분할 및 labeling
print("Loading violence data..")
for i in range(numberOfFilesViolence//100):
    # print(filePathViolence + "/" + fileListViolence[i])
    segVio, labelVio = vs.segVideo(filePathViolence + "/" + fileListViolence[i], isViolence= True)
    # print(np.shape(segVio), np.shape(labelVio))
    videoSquenceViolence.append([segVio, labelVio])

print("done!")

'''
정상 상태 영상 loading
'''
# input violence video path
filePathNormal = './Real Life Violence Dataset/NonViolence'
fileListNormal = os.listdir(filePathNormal)
numberOfFilesNormal = len(fileListNormal)
videoSquenceNormal =[]

# 정상 영상 분할 및 labeling
print("Loading normal data..")
for i in range(numberOfFilesNormal//100):
    # print(filePathNormal + "/" + fileListNormal[i])
    segNor, labelNor = vs.segVideo(filePathNormal + "/" + fileListNormal[i], isViolence= False)
    videoSquenceNormal.append([segNor, labelNor])
print("done!")





#   data set 생성. (2000, 2)
#   dataSet[i][0][j] = i 번째 영상, j frame의 픽셀 값 (240, 320,3)
#   dataSet[i][1] = i 번째 영상의 label
dataSet =[]
dataSet.extend(videoSquenceViolence)
dataSet.extend(videoSquenceNormal)
random.shuffle(dataSet)
# print(np.shape(dataSet))
# print(np.shape(dataSet[0][0]))
# print(np.shape(dataSet[0][1]))

trainX = []
trainY =[]
for i in range(len(dataSet)):
    trainX.extend(dataSet[i][0])
    trainY.extend(dataSet[i][1])
# print(np.shape(trainX),np.shape(trainY))



'''
training procedure
'''

trainedModel = C3D.modelTrain(trainData=np.array(trainX), trainLabel=np.array(trainY),  epch = 100, batch = 10)
# C3D.modelEvaluation(trainedModel, validSet[:][0], validSet[:][1])

