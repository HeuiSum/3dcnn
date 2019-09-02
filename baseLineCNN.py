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
    # print(np.shape(segVio))
    videoSquenceViolence.append([segVio, labelVio])
print("done!")

#   data set 생성. (2000, 2)
#   dataSet[i][0][j] = i 번째 영상, j frame의 픽셀 값 (240, 320,3)
#   dataSet[i][1] = i 번째 영상의 label
dataSet =[]
dataSet.extend(videoSquenceViolence)
# print(np.shape(trainSet))
dataSet.extend(videoSquenceNormal)
print(np.shape(dataSet))
print(np.shape(dataSet[0][0]))
print(np.shape(dataSet[0][1]))
random.shuffle(dataSet)

trainSet = dataSet[0:6]
validSet = dataSet[7:10]

# print(np.shape(dataSet[0]

'''
정상 상태 영상 loading
'''
# input violence video path
filePathNormal = './Real Life Violence Dataset/NonViolence'
fileListNormal = os.listdir(filePathNormal)
numberOfFilesNormal = len(fileListNormal)
videoSquenceNormal =[]

# 폭력 영상 분할 및 labeling
print("Loading normal data..")
for i in range(numberOfFilesNormal//100):
    # print(filePathNormal + "/" + fileListNormal[i])
    segNor, labelNor = vs.segVideo(filePathNormal + "/" + fileListNormal[i], isViolence= False)
    # print(np.shape(segNor))
    videoSquenceNormal.append([segNor, labelNor])
print("done!")



#   data set 생성. (2000, 2)
#   dataSet[i][0][j] = i 번째 영상, j frame의 픽셀 값 (240, 320,3)
#   dataSet[i][1] = i 번째 영상의 label
dataSet =[]
dataSet.extend(videoSquenceViolence)
# print(np.shape(trainSet))
dataSet.extend(videoSquenceNormal)
print(np.shape(dataSet))
print(np.shape(dataSet[0][0]))
print(np.shape(dataSet[0][1]))
random.shuffle(dataSet)

trainSet = dataSet[0:6]
validSet = dataSet[7:10]

# print(np.shape(dataSet[0][1]))
# print(np.shape(dataSet[0][0]))

'''
training procedure
# '''

trainedModel = C3D.modelTrain(trainData=trainSet[0][0],trainLabel=trainSet[0][1], epch = 5)
# C3D.modelEvaluation(trainedModel,validSet[:][0], validSet[:][1])



'''
modelling
'''