import pandas as pd
path = 'iris.csv'
data = pd.read_csv(path);

#select rows :  https://datatofish.com/select-rows-pandas-dataframe/
setosa = data.loc[data['species']=='setosa']
virginica = data.loc[data['species']=='virginica']

#select col :  https://datatofish.com/select-rows-pandas-dataframe/
newdata = pd.DataFrame(data,columns=['sepal_length','sepal_width'])

'''
print( setosa, virginica, newdata , sep='\n')
print( data.shape)
print(data.head())
print(data.describe())
print(data.columns)
for col in data.columns:
	print( col, data[col].dtype )
'''
	
# Make Data Frame
'''
data = pd.DataFrame({'Bob': ['I liked it.', 'It was awful.'], 
              'Sue': ['Pretty good.', 'Bland.']},
             index=['Product A', 'Product B'])
print( data )
'''

#Series
'''
data = pd.Series([30, 35, 40], index=['2015 Sales', '2016 Sales', '2017 Sales'], name='Product A')
print( data )
'''
# select 1 column
'''
print( data['sepal_length'] )
'''
# selct 1 row
'''
print( data.iloc[0] )
'''
# select the [i][j] val
'''
print( data['sepal_length'][1] )
'''
# select submatrix : iloc, loc [ row , col ], loc[ row,  col] 
'''
print(data, data.iloc[0:3,1:3], sep='\n' )
print(data, data.iloc[[0,2,3],[2,3]], sep='\n' )                          # Numbers
print(data, data.loc[[0,2,3],['petal_length','petal_width']], sep='\n' )  # Values
'''
#Manipulating the index
#data.set_index('Costum_Index')  # Note : In our data there is no index

#Aggregration
'''
print( data )
print( data['sepal_length'].sum() , data['sepal_length'].min(), data['sepal_length'].max(), data['sepal_length'].mean(),data['sepal_length'].value_counts(),data["species"].unique(), sep = '\n----------\n' )
print( data['sepal_length'].agg([sum,min,max]))
'''

# GroupBy
'''
print(data)
print( data.groupby("species").species.count() )
print( data.groupby("species")["species"].count() )
print( data.groupby(["species","sepal_width"])["species"].count() )
print( data.groupby(["species","sepal_width"])["species"].count() )
print( data.groupby(["species","sepal_width"])["petal_length"].agg([sum,min,max]) )
'''

# Sort
'''
print( data.sort_values(by='sepal_length') )
print( data.sort_values(by=['sepal_length','petal_length']) )
'''

#dtype & dtypes
# float64, int64, object
'''
print( data.dtypes )
for col in data.columns:
	print(col,data[col].dtype )
'''

#astype : change type
'''
print( data.petal_length.astype('int64') )
'''

# index.dtype
'''
print( data.index.dtype )
'''

# Missing data
'''
# Select row for prticular col missing value
print( data.loc[pd.isnull(data['species'])] )
# Replace nan with someting
print( data.species.fillna('Unknown') ) 
'''
# Repalce
'''
# Replace valus
print( data.species.replace('setosa','roseta') )
# Reanme columns
print( data.rename(columns={"sepal_length":"XYZ_length"}) )
# Rename index
print( data.rename(index={0:"First",1:"second"}) ) 
# Rename axis
print( data.rename_axis("wines", axis='rows').rename_axis("fields", axis='columns') )
'''

# Concating two table
'''
print( setosa, virginica, pd.concat([setosa, virginica]), setosa.shape, virginica.shape, pd.concat([setosa, virginica]).shape , sep = '\n---------\n')
'''
# joins : join on same index
'''
print( setosa.join(virginica, lsuffix='_LEFT', rsuffix='_RIGHT').dtypes )
'''
a = pd.DataFrame({'col1':[1,2,3,4,5],'col2':[11,12,13,14,15]},index=[0,1,2,3,4]);
b = pd.DataFrame({'col1':[21,22,23,24,25,26],'col2':[31,32,33,34,35,36]},index=[0,1,2,3,4,5]);
c = pd.DataFrame({'col1':[21,22,23,24,25,26],'col2':[31,32,33,34,35,36]},index=[0,0,1,1,2,2]);
print( a, b, a.join(b,lsuffix='left',rsuffix='right'), a.left_join(c,lsuffix='left',rsuffix='right'),sep = '\n----------------\n' )
# see : merge.data.frame : left, right, full, cross joins
