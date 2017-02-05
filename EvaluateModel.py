import MLTools

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical
from keras.models import model_from_json

import numpy as np


testData = open("census.test")
testString = testData.readlines()
inputData = MLTools.InputData(testString[1:], "categories.npy", one_hot_columns=[1, 3, 4, 5, 6, 7, 8, 9, 13, 14], skip_columns=[2])

testX = inputData.modified_data[:, :-1]
testY = inputData.modified_data[:, -2:-1]
testY = to_categorical(testY)

json_file = open('models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/model.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(testX, testY)
print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))