import os
import cv2
import pandas as pd
import numpy as np
from keras.utils import img_to_array


def process(img):

    # threshold inputted image and get edges
    inputted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    threshold, inputted = cv2.threshold(inputted, 200, 255, cv2.THRESH_BINARY)
    inputted = cv2.dilate(cv2.bitwise_not(inputted), np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.Canny(inputted, 30, 200)
    return edges


def isolate(edges, img, path):

    # initialize variables for return and constants
    images = []
    dimensions = (255, 255, 255)

    # isolate image of the digits
    contours, empty = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, dimensions, 2)
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        isolated = img[y:y + height, x:x + width]
        images.append(isolated)

    # create new image file for each digit in new folder named by question number
    for i, file in enumerate(images):
        (empty, result) = process(file)
        cv2.imwrite(path + '/' + str(i) + '.jpg', result)


def analysis(team, number, img, model):

    # create folder for question
    path = 'fermiQuestions' + '/' + str(team)
    if not os.path.exists(path + '/' + str(number)):
        os.makedirs(path + '/' + str(number))

    # process and isolate image into folder
    edges = process(img)
    isolate(edges, img, path + '/' + str(number))

    # get all digits, iterate over each one, and collate into one number
    complete = ''
    digits = os.listdir(path + '/' + str(number))
    for file in range(0, len(digits)):
        img = img_to_array(path + '/' + str(number) + '/' + str(file) + '.jpg')
        img = (img.reshape(1, 28, 28, 1).astype('float32')) / 255.0
        predicted = model.predict(img)
        complete = complete + str(np.argmax(predicted))
    complete = int(complete)
    return complete
