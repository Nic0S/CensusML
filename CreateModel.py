import MLTools

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical

import numpy as np

model_name = "model_256nr_drop20_128nr_drop20_adam_e250"


testData = open("census.data")
testString = testData.readlines()


# Automatically creates category data for one-hot features. This should only be called once for a given data model.
# Category data is not portable across different data models.
# todo: support manual category creation for non-one hot ordered categories
# todo: refactor into MLTools.py
# MLTools.create_categories("categories.npy", testString, one_hot_columns=[1, 3, 5, 6, 7, 8, 9, 13, 14])

inputData = MLTools.InputData(testString, "categories.npy", one_hot_columns=[1, 3, 5, 6, 7, 8, 9, 13, 14],
                              skip_columns=[2])

print(inputData.modified_data)
print(inputData.modified_data[:, :-2])
testX = inputData.modified_data[:, :-2]
testY = inputData.modified_data[:, -1]

count_greater = np.count_nonzero(testY == 1)
count_less = np.count_nonzero(testY == 0)

percent_greater = count_greater / (count_greater + count_less)

print("percent greater (?): " + str(percent_greater))

# testY = to_categorical(testY)


# print(testX)
# print(testY)


model = Sequential()
model.add(Dense(256, input_dim=testX.shape[1], init='normal', activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(128, init='normal', activation='relu'))
model.add(Dropout(0.20))
# model.add(Dense(128, init='normal', activation='relu'))
model.add(Dense(1, init='normal', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(testX, testY, nb_epoch=250, batch_size=32, verbose=2)

model_json = model.to_json()
with open("models/" + model_name + ".json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("models/" + model_name + ".h5")
print("Saved model to disk")

with open("models/models.list", "a") as models_file:
    models_file.write("\n" + model_name)
