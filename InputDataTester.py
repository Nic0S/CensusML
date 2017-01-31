import MLTools


testData = open("test.data")
testString = testData.readlines()
inputData = MLTools.InputData(testString, categorize_columns=[0, 1, 3])