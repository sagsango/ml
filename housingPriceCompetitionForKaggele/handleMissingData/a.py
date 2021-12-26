from pandas import read_csv
from numpy import nan
#path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.csv'
path = 'pima-indians-diabetes.csv'
data = read_csv(path,header=None)
print( data)
print( data.columns )
data[[1,2,3,4,5]] = data[[1,2,3,4,5]].replace(0, nan)
l = list( data[col].isnull().sum() for col in data.columns )
print( l )




