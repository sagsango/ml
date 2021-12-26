import pandas as pd;
testPath = "home-data-for-ml-course/test.csv";
testData = pd.read_csv(testPath);

prdcPath = "submission.csv";
prdcData = pd.read_csv(prdcPath);

trainPath = "home-data-for-ml-course/train.csv";
trainData = pd.read_csv(trainPath);

print("trainData", len(trainData), len(trainData.columns));
print("testData", len(testData),len(testData.columns));
print("prdcData",len(prdcData) , len(prdcData.columns));

print( trainData.head()['SalePrice'] )
print( trainData.at[1,'SalePrice'] )
print( trainData.at[1,'SalePrice'] == None )


