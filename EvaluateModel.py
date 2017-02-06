import MLTools

from keras.utils.np_utils import to_categorical
from keras.models import model_from_json

testData = open("census.test")
testString = testData.readlines()
inputData = MLTools.InputData(testString, "categories.npy", one_hot_columns=[1, 3, 5, 6, 7, 8, 9, 13, 14], skip_columns=[2])

testX = inputData.modified_data[:, :-2]
testY = inputData.modified_data[:, -2:-1]
print(testX)
print(testY)
testY = to_categorical(testY, 2)

json_file = open('models/model_256_128_128.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("models/model_256_128_128.h5")
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(testX)
print(testY)
score = loaded_model.evaluate(testX, testY)
print ("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

print(loaded_model.metrics_names)
print(score)
