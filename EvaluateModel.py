import MLTools

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

import numpy as np


testData = open("census.test")
testString = testData.readlines()
inputData = MLTools.InputData(testString[1:], one_hot_columns=[1, 3, 4, 5, 6, 7, 8, 9, 13, 14], skip_columns=[2])

testX = inputData.modified_data[:, :-1]
testY = inputData.modified_data[:, -2:-1]

print(testX)
print(testY)