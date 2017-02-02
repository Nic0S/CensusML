import MLTools


testData = open("census.data")
testString = testData.readlines()
inputData = MLTools.InputData(testString, categorize_columns=[1, 3, 4, 5, 6, 7, 8, 9, 13, 14], skip_columns=[2])