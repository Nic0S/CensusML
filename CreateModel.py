import MLTools

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

import numpy as np
import os


testData = open("census.data")
testString = testData.readlines()

# MLTools.create_categories("categories.npy", testString, one_hot_columns=[1, 3, 4, 5, 6, 7, 8, 9, 13, 14])

inputData = MLTools.InputData(testString, "categories.npy", one_hot_columns=[1, 3, 4, 5, 6, 7, 8, 9, 13, 14],
                              skip_columns=[2])

testX = inputData.modified_data[:, :-1]
testY = inputData.modified_data[:, -2:-1]

count_greater = np.count_nonzero(testY == 1)
count_less = np.count_nonzero(testY == 0)

percent_greater = count_greater / (count_greater + count_less)

print("percent greater (?): " + str(percent_greater))

testY = to_categorical(testY)


print(testX)
print(testY)


model = Sequential()
model.add(Dense(64, input_dim=testX.shape[1]))
model.add(Dense(64))
model.add(Dense(64))
model.add(Dense(2))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(testX, testY, nb_epoch=25, batch_size=32)

model_json = model.to_json()
with open("models/model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/model.h5")
print("Saved model to disk")