import os
import pandas as pd
import cv2
import numpy as np
from keras.models import load_model
import fermiQuestions
import train

train.modeling()
model = load_model('updated_model')

# get answer key as a DataFrame object
path = 'fermiQuestions'
answerKey = os.listdir(path + '/' + 'answerKeys')
currentKey = pd.read_csv(path + '/' + 'answerKeys' + '/' + answerKey[0])

# get folder with all answer sheets
answerSheets = os.listdir(path + '/' + 'answerSheets')

# initialize a Dataframe object for final scores of all answer sheets
finalScores = pd.DataFrame({'maximum': [5] * 39})
finalScores.index += 1

# loops through each answer sheet
for file in range(0, len(answerSheets)):

    # creates folder for the answerSheet and DataFrame object for tracking points
    if not os.path.exists(path + '/' + str(file)):
        os.makedirs(path + '/' + str(file))
    points = [0] * len(currentKey)

    # get current answer sheet and Otsu's thresholding
    currSheet = cv2.imread(path + '/' + 'answerSheets' + '/' + answerSheets[file], 0)
    constantThresh = cv2.THRESH_BINARY | cv2.THRESH_OTSU
    (threshold, retSheet) = cv2.threshold(currSheet, 128, 255, constantThresh)

    # defining constant values
    rectangleValue = cv2.MORPH_RECT
    finalKernel = cv2.getStructuringElement(rectangleValue, (3, 3))
    kernelLength = np.array(currSheet).shape[1] // 80
    differenceReturn = 255 - retSheet

    # isolating vertical and horizontal lines, respectively
    verticalKernel = cv2.getStructuringElement(rectangleValue, (1, kernelLength))
    verticalImage = cv2.erode(differenceReturn, verticalKernel, iterations = 3)
    verticalLines = cv2.dilate(verticalImage, verticalKernel, iterations = 3)
    horizontalKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernelLength, 1))
    horizontalImage = cv2.erode(differenceReturn, horizontalKernel, iterations = 3)
    horizontalLines = cv2.dilate(horizontalImage, horizontalKernel, iterations = 3)

    # gets boxes from vertical and horizontal lines
    linesImage = ~cv2.addWeighted(verticalLines, 0.5, horizontalLines, 0.5, 0)
    erodedImage = cv2.erode(linesImage, finalKernel, iterations = 2)
    (threshold, retImage) = cv2.threshold(erodedImage, 128, 255, constantThresh)
    (listContours, hierarchy) = cv2.findContours(retImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    listBoxes = []
    for contour in listContours:
        listBoxes.append(cv2.boundingRect(contour))
    (listContours, listBoxes) = zip(*sorted(zip(listContours, listBoxes), reverse=True, key=lambda a: a[1][1]))

    # isolate a new image for each box that contains a question's answer
    questionNumber = 0
    for contour in listContours:
        x, y, width, height = cv2.boundingRect(contour)
        if width > 80 and ((height > 20 and width < 400) and width > 3 * height):
            questionNumber += 1
            newImage = currSheet[y + 8:y + height - 8, x:x + width]
            (threshold, newImage) = cv2.threshold(newImage, 128, 255, constantThresh)
            cv2.imwrite(path + '/' + str(file) + '/' + str(questionNumber) + '.png', newImage)

    # get team number
    os.rename(path + '/' + str(file) + '/' + str(questionNumber) + '.png',
              path + '/' + str(file) + '/' + 'teamNumber' + '.png')

    # iterate over all answers on answer sheet
    for answer in range(len(currentKey), 0, -1):
        answerNumber = len(currentKey) - answer
        if currentKey.loc[answerNumber, 'Question Type'] == 'Fermi Questions':
            prediction = fermiQuestions.analysis(file, answerNumber, path + '/' + str(file) +
                                                           '/' + str(answerNumber + 1) + '.png', model)
            # find score for answer
            answer = currentKey.loc[answerNumber, 'Answer']
            if prediction == answer:
                points[answerNumber] = 5
            elif (prediction + 1 == answer) or (prediction - 1 == answer):
                points[answerNumber] = 3
            elif (prediction + 2 == answer) or (prediction - 2 == answer):
                points[answerNumber] = 1
            else:
                points[answerNumber] = 0
    finalScores[0] = points

# fix checking answer out of analysis in fermiQuestions and get team number as well
print(finalScores)
