# Do not name file as: matplotlib.py
# https://www.geeksforgeeks.org/graph-plotting-in-python-set-1/
import matplotlib.pyplot as plt 

#lineplot
'''
#single Plot
x = [1,2,3] 
y = [2,4,1] 
plt.plot(x, y) 
plt.xlabel('x - axis') 
plt.ylabel('y - axis') 
plt.title('My first graph!') 
plt.show() 
'''
'''
#multiple plot
x1 = [1,2,3,4,5,6,7,8,9,10]
y1 = [10,20,31,43,53,64,60,22,50,100]
x2 = [1,10,20,23,10,4,10]
y2 = [50,58,28,50,27,50,12]
plt.plot(x1,y1,label = 'plot1')
plt.plot(x2,y2,label = 'plot2')
plt.xlabel('x - axis') 
plt.ylabel('y - axis') 
plt.title('plot title') 
plt.legend() 
plt.show()
'''
'''
#custimzed plot
x = [1,2,3,4,5,6]  
y = [2,4,1,5,2,6] 
plt.plot(x, y, color='green', linestyle='dashed', linewidth = 3, 
         marker='o', markerfacecolor='blue', markersize=12) 
plt.ylim(1,8) 
plt.xlim(1,8) 
plt.xlabel('x - axis')
plt.ylabel('y - axis') 
plt.title('Some cool customizations!') 
plt.show() 
'''

#bar chart
'''
left = [1, 2, 3, 4, 5]  
height = [10, 24, 36, 40, 5] 
tick_label = ['one', 'two', 'three', 'four', 'five'] 
plt.bar(left, height, tick_label = tick_label, 
        width = 0.8, color = ['red', 'green','yellow','pink','orange']) 
plt.xlabel('x - axis') 
plt.ylabel('y - axis') 
plt.title('My bar chart!') 
plt.show() 
'''


#Histogram
'''
frequencies = [2,5,70,40,30,45,50,45,43,40,44,60,7,13,57,18,90,77,32,21,20,40] 
range = (0, 100) 
bins = 10  
plt.hist(frequencies, bins, range, color = 'green', 
        histtype = 'bar', rwidth = 0.8) 
plt.xlabel('numbers')
plt.ylabel('freq') 
plt.title('My histogram')
plt.show() 
'''


#Scatter plot
'''
x = [1,2,3,4,5,6,7,8,9,10] 
y = [2,4,5,7,6,8,9,11,12,12]  
plt.scatter(x, y, label= "stars", color= "green", marker= "*", s=30) 
plt.xlabel('x - axis') 
plt.ylabel('y - axis') 
plt.title('My scatter plot!') 
plt.legend() 
plt.show() 
'''

#Pie Chart
'''
activities = ['eat', 'sleep', 'work', 'play'] 
slices = [3, 7, 8, 6] 
colors = ['r', 'y', 'g', 'b'] 
plt.pie(slices, labels = activities, colors=colors,  
        startangle=90, shadow = True, explode = (0, 0, 0.1, 0), 
        radius = 1.2, autopct = '%1.1f%%') 
plt.legend() 
plt.show() 
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
path = 'fifa.csv'
data = pd.read_csv(path,index_col="Date",parse_dates=True)
#print ( data )

#lineplot : x-y
'''
plt.figure(figsize=(100,20))
sns.lineplot(data=data.head(2))
plt.show()
'''

#barplot x-y
'''
sns.barplot(x=data.head().index, y=data.head()['ARG'])
plt.show()
'''

#heatmat : [x1 - x2 - x3--- xn] - y
'''
sns.heatmap(data.head(2),annot=True)
plt.show()
'''
#scatter plot [x1-x2]
'''
sns.scatterplot(x=data['ARG'], y=data['BRA'])
plt.show()
'''
#scatter plot [x1-x2-x3]
'''
sns.scatterplot(x=data['ARG'], y=data['BRA'],hue=data['ESP'])
plt.show()
'''
#lmplot [x1-x2-x3]
'''
sns.lmplot(x="ARG", y="BRA", hue="ESP", data=data.head(30))   # no of lines = diff unique val in hue
plt.show()
'''
#swarmplot
'''
print( data )
sns.swarmplot(x = data.head(50)['ARG'], y = data.head(50)['BRA'])
plt.show()
'''

# data : https://gist.githubusercontent.com/curran/a08a1080b88344b0c8a7/raw/0e7a9b0a5d22642a06d3d5b9bcbad9890c8ee534/iris.csv
path = 'iris.csv'
data = pd.read_csv(path);
#select rows :  https://datatofish.com/select-rows-pandas-dataframe/
setosa = data.loc[data['species']=='setosa']
virginica = data.loc[data['species']=='virginica']
#select col :  https://datatofish.com/select-rows-pandas-dataframe/
newdata = pd.DataFrame(data,columns=['sepal_length','sepal_width'])
print( setosa, virginica, newdata , sep='\n')

# histogram
'''
sns.histplot(data=data['petal_length'],kde=False)
sns.displot(data=data['petal_length'],kde=False)
plt.show()
'''

# histogram : more than one variable
'''
sns.histplot(data=setosa['petal_length'], label="setosa", kde=False,color='yellow')
sns.histplot(data=virginica['petal_length'], label="virginica", kde=False,color='blue')
plt.title("Histogram of Petal Lengths, by Species")
plt.legend()
plt.show()
'''

# Density plot :  x - dencity ( 1 D )
'''
sns.kdeplot(data=data['petal_length'],shade=True)
plt.show()
'''

# Density plot :  x1 -x2 - dencity ( 2 D )
sns.jointplot(x=data['sepal_length'],y=data['sepal_width'],kind='kde',shade=True)
plt.show()









