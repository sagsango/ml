#input and spilt
'''
t=int(input())
for test in range(t):
    n,d=map(int,input().split())
    A=list(map(int,input().split()))
'''
#1D array or list
'''
n = int(input());
dp = [];
N = 20000;
for i in range(N+1):
    dp.append(0);
    
a = list(map(int,input().split()))
print(a[1]);

ans = 0;
dp[0]=1;
for i in range(n):
    dp2 = [None]*(N+1);
    for j in range(N+1):
		dp2[j]=dp[j];
    for j in range (N+1):
        if( j + a[i] <= N ):
            dp2[j+a[i]]=dp2[j+a[i]] or dp[j];
    for j in range(N+1):
		dp[j]=dp2[j];
for i in range(N+1):
    if( i%2 ==0 and dp[i]!=0):
        ans+=1;
print(ans-1);
'''
#Mulidimentioan array ( list )
'''
n = 10
m = 10
k = 10
aar1D = [0]*n
arr2D = [[0]*n]*m
arr3D = [[[0]*n]*m]*k
'''
#int
'''
a = 0
'''
#float
'''
a = 0
'''
#char
'''
No char type in python
c = 'A' : string
ord('A')
chr(65)
'''
#string
'''
s = "  Hello "
s = ' Hello    '

r = s.strip() 
r = s.lstrip()
r = s.rstrip()
r = s.lower()
r = s.upper()

r = s[l:r] #NOTE: substring(l,r-1)

str.isalnum()
str.isalpha()
str.islower()
str.isnumeric()	
str.isspace()
str.istitle()
str.isupper()

len(string)
s.split()
s.split('a')
s.replace('old','new')  #NOTE: on substring

del s
'''
#operators
'''
+	Addition	x + y	
-	Subtraction	x - y	
*	Multiplication	x * y	
/	Division	x / y	
%	Modulus	x % y	
**	Exponentiation	x ** y	
//	Floor division	x // y

=	x = 5	x = 5	
+=	x += 3	x = x + 3	
-=	x -= 3	x = x - 3	
*=	x *= 3	x = x * 3	
/=	x /= 3	x = x / 3	
%=	x %= 3	x = x % 3	
//=	x //= 3	x = x // 3	
**=	x **= 3	x = x ** 3	
&=	x &= 3	x = x & 3	
|=	x |= 3	x = x | 3	
^=	x ^= 3	x = x ^ 3	
>>=	x >>= 3	x = x >> 3	
<<=	x <<= 3	x = x << 3

==	Equal	x == y	
!=	Not equal	x != y	
>	Greater than	x > y	
<	Less than	x < y	
>=	Greater than or equal to	x >= y	
<=	Less than or equal to	x <= y

and Returns True if both statements are true					x < 5 and  x < 10	
or	Returns True if one of the statements is true				x < 5 or x < 4	
not	Reverse the result, returns False if the result is true		not(x < 5 and x < 10)	

& 	AND	Sets each bit to 1 if both bits are 1
|	OR	Sets each bit to 1 if one of two bits is 1
 ^	XOR	Sets each bit to 1 if only one of two bits is 1
~ 	NOT	Inverts all the bits
<<	Zero fill left shift	Shift left by pushing zeros in from the right and let the leftmost bits fall off
>>	Signed right shift	Shift right by pushing copies of the leftmost bit in from the left, and let the rightmost bits fall off
'''
#if-else
'''
if a > b :
	print( " A > B " )

if a > b :
	print( " A > B " )
else :
	print( " A is not greater than B" )

if a > b :
	print( " A > B " )
elif a == b :
	print( " A = B " )
else :
	print( " A < B " )
'''
#loops
'''
fruits = ["apple", "banana", "cherry"]
for x in fruits:
  print(x)
for x in "banana":
  print(x)
for x in range(6):
  print(x)
for x in range(2, 6):
  print(x)
for x in range(2, 30, 3):
  print(x)
for x in [0, 1, 2]:
  pass
i = 0
while (i < 3):    
    i = i + 1
    print("Namaste")
'''
#list			 : can have diff types of data
'''
mylist = list(()) : Empty list
thislist = ["apple", "banana", "cherry", 13.5, 10303]
thatlist = list(input().strip().split())
print(thislist)
print(thislist[1])
print(thislist[2:5])
thislist[1] = "blackcurrant"
print(len(thislist))
thislist.append("orange")
thislist.insert(1, "new fruit")
print(thislist)
thislist.remove("banana")
thislist.pop()
del thislist[0]
thislist.clear()
del thislist
thislist = thatlist
thislist.sort()
thislist.reverse()

tmplist = [ ["hello",123], [1223,1213,1112] ]
yourlist = list( list( () ) ) 
yourlist = list( range(10)  )
for i in range(10):
	yourlist[i] = list( () )
yourlist[0].append("Hello")
yourlist[9].append("Namaste")
for i in range(10):
    for j in range(len(l[i])):
        print( l[i][j] )



'''
#set 			 : can have diff types of data
#                : Not sorted
'''
myset = set(()) : Empty set
thisset = {"apple", "banana", "cherry"}
thatset = set(input().strip().split())
thisset.add("orange")
print(len(thisset))
thisset.remove("banana")  # If banana not present it will give error
thisset.discard("banana") # If banana not present it will NOT give error
l = list(thisset)
l.sort()
print(l)
thisset.clear()
del thisset
'''
#dictionary		 : can have diff types of data
'''
thatdict = dict(()) : Empty dict
thisdict = {
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
x = thisdict["model"]                 # if key is not present it will give error
x = thisdict.get("model")             # if key is not present it will NOT give error
thisdict["year"] = 2018
thisdict["key" ] = "keyValue"
for x in thisdict:
  print(x,thisdict[x])
thatdict = thisdict
del thisdict["key"]
thisdict.clear()
del thisdict

#2D
myfamily = {
  "child1" : {
    "name" : "Emil",
    "year" : 2004
  },
  "child2" : {
    "name" : "Tobias",
    "year" : 2007
  },
  "child3" : {
    "name" : "Linus",
    "year" : 2011
  }
}
yourfamily = dict( dict( () ) )
yourfamily[ "child1" ] = dict( () )
yourfamily[ "child1" ][ "name" ] = "Emil"
yourfamily[ "child1" ][ "year" ] = "2004"
print(  len( yourfamily )  )
print(  len( yourfamily[ "child1" ] )  )
print( yourfamily.get("child100000000") == None )
'''
#function
'''
def my_function():
  print("Hello from a function")
def my_function(x):
  return 5 * x
def tri_recursion(k):
  if(k > 0):
    result = k + tri_recursion(k - 1)
  else:
    result = 0
  return result
def myfunction():
  pass
def myFun(*argv):  
    for arg in argv:  
        print (arg) 
myFun('Hello', 'Welcome', 'to', 'GeeksforGeeks')
def myFun(**kwargs):  
    for key in kwargs:
        print (key, kwargs[key])
myFun(first ='Geeks', mid ='for', last='Geeks')    
'''
#OPPs
'''
#single Inharitanse
class Person:
  def __init__(self, name, age):
    self.name = name
    self.age = age
  def myfunc(self):
    print("Hello my name is " + self.name)
p1 = Person("John", 36)
p1.myfunc()
class Student(Person):
  def __init__(self, name, age, year):
    #super().__init__(name, age)
    Person.__init__(self,name, age)
    self.graduationyear = year
x = Student("Peter",9,39)
x.myfunc()
#Multiple inheritance
class Base1(object): 
    def __init__(self): 
        self.str1 = "Geek1"
        print("Base1") 
  
class Base2(object): 
    def __init__(self): 
        self.str2 = "Geek2"        
        print("Base2") 
  
class Derived(Base1, Base2): 
    def __init__(self): 
        Base1.__init__(self) 
        Base2.__init__(self) 
        print("Derived") 
    def printStrs(self): 
        print(self.str1, self.str2)
        #print(str1, str2)     #DO Not work
ob = Derived() 
ob.printStrs() 
#private Member : Not aviable for child : __varName
'''
#File Handling
'''
file = open("myfile.txt","r")		 	 #r			offset=0
file = open("myfile.txt","r+")			 #r+w		offset=0	
file = open("myfile.txt","w")            #w         offset=0
file = open("myfile.txt","w+")           #w+r       offset=0
file = open("myfile.txt","a")            #w         offset=end of file
file = open("myfile.txt","a+")           #r+w       offset=end of file
file.write("This line will be written in the file")
for line in file:                        #Read file line by line
	print(line)
print( file.read()  )                    #read whole file 
print( file.read(10) )					 #read first 10 char
print( file.readline() )                 #read only 1 line
print( file.readline(10) )               #read only first 10 char of a line
file.seek(0)                             #set offset 
file.close()

#Run bellow code : try with diff open mode and seek
file = open("filedemo.txt","a+")
file.write("Hello! How are you?\n")
file.write("I am fine. What about you?\n")
file.write("    This         is         last line    !!!\n      ")
file.seek(0)
print(file.read())
file.seek(0)
for line in file:
	curline = line.strip().split()
	for word in curline:
		print( word, end=" ")
	print("")
'''
#Modules and Packages
#Regular Expression
#Error Handling
'''
a = 10
b = 20
c = a * b
assert( c == 200 )
'''
#Exception handling








#Parse csv file with standard library
'''
def parse_file(datafile):
    d = dict(dict())
    file = open(datafile,"r")
    l = list(file.readline().strip().split(","))
    n = len(l)
    for i in range(10):
		d[i] = dict(())
		line = list(file.readline().strip().split(","))
		for j in range(len(line)):
			d[i][l[j]]=line[j]
    return d
def test():
	d = parse_file("Downloads/beatles-diskography.csv")
	firstline = {'Title': 'Please Please Me', 'UK Chart Position': '1', 'Label': 'Parlophone(UK)', 'Released': '22 March 1963', 'US Chart Position': '-', 'RIAA Certification': 'Platinum', 'BPI Certification': 'Gold'}
	tenthline = {'Title': '', 'UK Chart Position': '1', 'Label': 'Parlophone(UK)', 'Released': '10 July 1964', 'US Chart Position': '-', 'RIAA Certification': '', 'BPI Certification': 'Gold'}
	assert( d[0] == firstline )
	assert( d[9] == tenthline )
test()
'''






'''
#Parse csv file with pandas
#download File https://www.kaggle.com/dansbecker/melbourne-housing-snapshot
import pandas as pd;
filePath = "Downloads/melb_data.csv";
fileData = pd.read_csv(filePath);

print("#All column Name...");
for col in fileData:
	print( col , end = ",");
print("");
print( fileData.columns );

print("#summary...");
print( fileData.describe() );    # summary
#print( fileData.describe()["Id"]["max"] );
#print( fileData["Rooms"] );
#print( fileData.Rooms );
#print( fileData["Rooms"][0:5] );
#print( type(fileData["Rooms"]) );
print("#First Five data...");
print( fileData.head() );        # first k data : head(k)
print("#All data...");
print( fileData );               # all data
# dropna drops missing values (think of na as "not available")
extracted_data = fileData.dropna(axis=0);

y = extracted_data.Price;
data_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
X = extracted_data[ data_features ];

#Building Your Model
from sklearn.tree import DecisionTreeRegressor;
data_model = DecisionTreeRegressor(random_state=1);
data_model.fit(X,y); # Mapping for know x -> y
print("#Making predictions for the following house features:");
print(X)
print("#The predictions are");
print(data_model.predict(X)); # mapping for unknown x -> y

#Note Data types are NOT same
predicted_data = list(data_model.predict(X));
actual_data    = list(extracted_data["Price"]);  # just use y 

print( type(predicted_data), len(predicted_data), type(actual_data),len(actual_data) );
print("#First 100 predictions:");
for i in range(100):
	print( "Actual:",actual_data[i],"  ", "Pridected:",predicted_data[i], sep = "", end = '\n'  );
print("#Prediction which are diff");
for i in range(len(actual_data)):
	a = float(actual_data[i]);
	b = float(predicted_data[i]);
	if( a != b ):
		print("Prediction is Diff=>","Actual:", a,"  ","Pridiction:", b," ","Diff:", round(abs(a-b),0) );
'''





'''
import pandas as pd;
filePath = "Downloads/melb_data.csv";
fileData = pd.read_csv(filePath);
fileData = fileData[ ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude','Price'] ];
fileData.dropna(axis=0);
data_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude'];
X = fileData[ data_features ];
y = fileData.Price;
from sklearn.model_selection import train_test_split;
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0);
from sklearn.tree import DecisionTreeRegressor;
data_model = DecisionTreeRegressor(random_state=1);
data_model.fit(train_X,train_y); 
val_predictions = data_model.predict(val_X);
from sklearn.metrics import mean_absolute_error;
print("Mean absolute error using DecisionTreeRegressor:",mean_absolute_error(val_y, val_predictions));
def get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y):
	model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0);
	model.fit(train_X, train_y);
	preds_val = model.predict(val_X)
	mae =  mean_absolute_error(val_y, preds_val)
	return mae;
best_mae = 1e9;
best_leaf_node = 0;
for i in range(2,10000,10): #MaxLeafNode >= 2
	#print(i,get_mae(i,train_X,val_X,train_y,val_y));
	cur_mae = get_mae(i,train_X,val_X,train_y,val_y);
	if( cur_mae < best_mae ):
		best_mae = cur_mae;
		best_leaf_node = i;

final_model = DecisionTreeRegressor(max_leaf_nodes=best_leaf_node,random_state=0);
final_model.fit(X, y);
#Now is ready for prediction on some unknown X, using DicisionTreeRegressor

#RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor;
from sklearn.metrics import mean_absolute_error;
forest_model = RandomForestRegressor(random_state=1);
forest_model.fit(train_X, train_y);
forest_predictions = forest_model.predict(val_X);
print("Mean absolute error using RandomForestRegressor:",mean_absolute_error(val_y, forest_predictions));
'''



# NOT WORK BECAUSE NaN some of the values
'''
import pandas as pd;
filePath = "home-data-for-ml-course/train.csv";
fileData = pd.read_csv(filePath);
dscrp = list(fileData.describe().columns);
fileData = fileData[ dscrp ];
fileData = fileData.dropna(axis=0);
redudent = ['Id','YearRemodAdd','SalePrice'];
for colName in redudent:
	dscrp.remove(colName);
X = fileData[ dscrp ];
y = fileData.SalePrice;

from sklearn.model_selection import train_test_split;
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0);
from sklearn.ensemble import RandomForestRegressor;
from sklearn.metrics import mean_absolute_error;
''' '''
def get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y):
	model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0);
	model.fit(train_X, train_y);
	preds_val = model.predict(val_X)
	mae =  mean_absolute_error(val_y, preds_val)
	return mae;
best_mae = 1e9;
best_leaf_node = 0;
for i in range(2,500,1): #MaxLeafNode >= 2
	#print(i,get_mae(i,train_X,val_X,train_y,val_y));
	cur_mae = get_mae(i,train_X,val_X,train_y,val_y);
	if( cur_mae < best_mae ):
		best_mae = cur_mae;
		best_leaf_node = i;
	print( "Leaf Node...",i, "mae:",cur_mae );
print( best_mae, best_leaf_node );
#18755.911577644532 310  [230,500->]
#18746.64400778906  289
''' '''
final_model = RandomForestRegressor(max_leaf_nodes=289, random_state=0);
final_model.fit(X, y);
testPath = "home-data-for-ml-course/test.csv";
testData = pd.read_csv(testPath);
Xt = testData[ dscrp ];
Xt = Xt.dropna(axis=0);
final_prdc = final_model.predict(Xt);

output = pd.DataFrame({'Id' : testData.Id, 'SalePrice': final_prdc});
output.to_csv('submission.csv', indx=False);
'''



#Works submission 1,2,3
'''
import pandas as pd;
filePath = "home-data-for-ml-course/train.csv";
fileData = pd.read_csv(filePath);
testPath = "home-data-for-ml-course/test.csv";
testData = pd.read_csv(testPath);
Dscrp = list(());
for col in testData.describe():
	tmp = testData[col];
	l1 = len(tmp);
	tmp = tmp.dropna(axis=0);
	l2 = len(tmp);
	if( l1 == l2 ):
		Dscrp.append(col);
dscrp = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
Dscrp.remove('Id');
Dscrp.remove('YearRemodAdd');
dscrp = Dscrp;
''' '''
testData = testData[ dscrp ];
print( len(testData) );
testData = testData.dropna(axis=0);
print( len(testData) );
print( len(dscrp), len(Dscrp) );
print( dscrp );
print( Dscrp );
''' '''
print( dscrp );
dtrain = list(fileData.describe().columns);
print( len(dscrp), len(dtrain) );
''' '''
dscrp.append('SalePrice');
fileData = fileData[ dscrp ];
fileData = fileData.dropna(axis=0);
dscrp.remove('SalePrice');
X = fileData[ dscrp ];
y = fileData.SalePrice;
from sklearn.model_selection import train_test_split;
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0);
from sklearn.ensemble import RandomForestRegressor;
from sklearn.metrics import mean_absolute_error;

''' '''
def get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y):
	model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0);
	model.fit(train_X, train_y);
	preds_val = model.predict(val_X)
	mae =  mean_absolute_error(val_y, preds_val)
	return mae;
best_mae = 1e9;
best_leaf_node = 0;
for i in range(400,600,1): #MaxLeafNode >= 2
	#print(i,get_mae(i,train_X,val_X,train_y,val_y));
	cur_mae = get_mae(i,train_X,val_X,train_y,val_y);
	if( cur_mae < best_mae ):
		best_mae = cur_mae;
		best_leaf_node = i;
	print( "Leaf Node...",i, "mae:",cur_mae );
print( best_mae, best_leaf_node );
''' '''
final_model = RandomForestRegressor(max_leaf_nodes=517, random_state=0);
final_model.fit(X, y);
Xt = testData[ dscrp ];
final_prdc = final_model.predict(Xt);
output = pd.DataFrame({'Id' : testData.Id, 'SalePrice': final_prdc});
output.to_csv('submission.csv', index=False); 
'''



#WORKS SUBMISSION 4
'''
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('home-data-for-ml-course/train.csv', index_col='Id')
X_test_full = pd.read_csv('home-data-for-ml-course/test.csv', index_col='Id')

# Obtain target and predictors
y = X_full.SalePrice
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = X_full[features].copy()
X_test = X_test_full[features].copy()

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,random_state=0)

from sklearn.ensemble import RandomForestRegressor

# Define the models
model_1 = RandomForestRegressor(n_estimators=50, random_state=0)
model_2 = RandomForestRegressor(n_estimators=100, random_state=0)
model_3 = RandomForestRegressor(n_estimators=100, criterion='mae', random_state=0)
model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)

models = [model_1, model_2, model_3, model_4, model_5]
from sklearn.metrics import mean_absolute_error

# Function for comparing different models
def score_model(model, X_t=X_train, X_v=X_valid, y_t=y_train, y_v=y_valid):
    model.fit(X_t, y_t)
    preds = model.predict(X_v)
    return mean_absolute_error(y_v, preds)

best_model = models[0];
best_mae  = 1e9;
for i in range(0, len(models)):
    mae = score_model(models[i])
    #print("Model %d MAE: %d" % (i+1, mae))
    if( mae < best_mae ):
    	best_mae = mae;
    	best_model = models[i];

# Define a model
my_model = best_model # Your code here

# Fit the model to the training data
my_model.fit(X, y)

# Generate test predictions
preds_test = my_model.predict(X_test)

# Save predictions in format used for competition scoring
output = pd.DataFrame({'Id': X_test.index,'SalePrice': preds_test})
output.to_csv('submission.csv', index=False)
'''


'''
import pandas as pd;
trainPath = 'home-data-for-ml-course/train.csv';
testPath  = 'home-data-for-ml-course/test.csv';
trainData = pd.read_csv(trainPath);
testData  = pd.read_csv(testPath);

feature   = list(trainData.describe().columns);
if( 'Id' in feature ):
	feature.remove('Id');
if( 'SalePrice' in feature ):
	feature.remove('SalePrice');
#all int type feature are present in our list

X = trainData[ feature ];
y = trainData.SalePrice;
test_X = testData[ feature ];

from sklearn.model_selection import train_test_split;
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0);
cols_with_missing = [ col for col in train_X.columns if train_X[col].isnull().any() ];

# Make copy to avoid changing original data (when imputing)
train_X_plus = train_X.copy();
val_X_plus = val_X.copy();
test_X_plus = test_X.copy();
X_plus = X.copy();

# Make new columns indicating what will be imputed
for col in cols_with_missing :
	train_X_plus[ col + '_was_missing' ] = train_X_plus[col].isnull();
	val_X_plus[ col + '_was_missing' ] = val_X_plus[col].isnull();
	test_X_plus[ col + '_was_missing'] = test_X_plus[col].isnull();
	X_plus[ col + '_was_missing' ] = X_plus[ col ].isnull();

#Imputation
from sklearn.impute import SimpleImputer;
my_imputer = SimpleImputer();
imputed_train_X_plus = pd.DataFrame(my_imputer.fit_transform(train_X_plus));
imputed_val_X_plus = pd.DataFrame(my_imputer.transform(val_X_plus));
imputed_X_plus = pd.DataFrame(my_imputer.fit_transform(X_plus));
imputed_test_X_plus = pd.DataFrame(my_imputer.transform(test_X_plus));

# Imputation removed column names; put them back
imputed_train_X_plus.columns = train_X_plus.columns;
imputed_val_X_plus.columns = val_X_plus.columns;
imputed_X_plus.columns = X_plus.columns;
imputed_test_X_plus.columns = test_X_plus.columns;

from sklearn.ensemble import RandomForestRegressor;
from sklearn.metrics import mean_absolute_error;
def get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y):
	model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0);
	model.fit(train_X, train_y);
	preds_val = model.predict(val_X)
	mae =  mean_absolute_error(val_y, preds_val)
	return mae;
	
''' '''
best_mae = 1e9;
best_leaf_nodes = 1e9;
for max_leaf_nodes in range(600,800,1):
	cur_mae = get_mae(max_leaf_nodes,imputed_train_X_plus,imputed_val_X_plus,train_y,val_y);
	print( max_leaf_nodes,cur_mae );
	if( cur_mae < best_mae ):
		best_mae = cur_mae;
		best_leaf_nodes = max_leaf_nodes;
#print( best_leaf_nodes, best_mae );
# 624 17601.654668381463
''' '''
best_leaf_nodes = 624;

final_model = RandomForestRegressor(max_leaf_nodes=best_leaf_nodes, random_state=0);
final_model.fit(imputed_X_plus,y);
preds_y = final_model.predict(imputed_test_X_plus);

output = pd.DataFrame({'Id': testData.Id,'SalePrice': preds_y})
output.to_csv('submission.csv', index=False)
'''






'''
import pandas as pd;
trainPath = 'home-data-for-ml-course/train.csv';
testPath  = 'home-data-for-ml-course/test.csv';
trainData = pd.read_csv(trainPath);
testData  = pd.read_csv(testPath);

feature   = list(trainData.describe().columns);
if( 'Id' in feature ):
	feature.remove('Id');
if( 'SalePrice' in feature ):
	feature.remove('SalePrice');
#all int type feature are present in our list

X = trainData[ feature ];
y = trainData.SalePrice;
test_X = testData[ feature ];

from sklearn.model_selection import train_test_split;
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0);
cols_with_missing = [ col for col in train_X.columns if train_X[col].isnull().any() ];

# Make copy to avoid changing original data (when imputing)
train_X_plus = train_X.copy();
val_X_plus = val_X.copy();
test_X_plus = test_X.copy();
X_plus = X.copy();

# Make new columns indicating what will be imputed
for col in cols_with_missing :
	train_X_plus[ col + '_was_missing' ] = train_X_plus[col].isnull();
	val_X_plus[ col + '_was_missing' ] = val_X_plus[col].isnull();
	test_X_plus[ col + '_was_missing'] = test_X_plus[col].isnull();
	X_plus[ col + '_was_missing' ] = X_plus[ col ].isnull();

#Imputation
from sklearn.impute import SimpleImputer;
my_imputer = SimpleImputer();
imputed_train_X_plus = pd.DataFrame(my_imputer.fit_transform(train_X_plus));
imputed_val_X_plus = pd.DataFrame(my_imputer.transform(val_X_plus));
imputed_X_plus = pd.DataFrame(my_imputer.fit_transform(X_plus));
imputed_test_X_plus = pd.DataFrame(my_imputer.transform(test_X_plus));

# Imputation removed column names; put them back
imputed_train_X_plus.columns = train_X_plus.columns;
imputed_val_X_plus.columns = val_X_plus.columns;
imputed_X_plus.columns = X_plus.columns;
imputed_test_X_plus.columns = test_X_plus.columns;
''' '''
from sklearn.ensemble import RandomForestRegressor;
from sklearn.metrics import mean_absolute_error;
def get_mae(max_leaf_nodes,train_X,val_X,train_y,val_y):
	model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0);
	model.fit(train_X, train_y);
	preds_val = model.predict(val_X)
	mae =  mean_absolute_error(val_y, preds_val)
	return mae;
	
''' '''
best_mae = 1e9;
best_leaf_nodes = 1e9;
for max_leaf_nodes in range(600,800,1):
	cur_mae = get_mae(max_leaf_nodes,imputed_train_X_plus,imputed_val_X_plus,train_y,val_y);
	print( max_leaf_nodes,cur_mae );
	if( cur_mae < best_mae ):
		best_mae = cur_mae;
		best_leaf_nodes = max_leaf_nodes;
#print( best_leaf_nodes, best_mae );
# 624 17601.654668381463
''' '''
best_leaf_nodes = 624;
#model_4 = RandomForestRegressor(n_estimators=200, min_samples_split=20, random_state=0)
#model_5 = RandomForestRegressor(n_estimators=100, max_depth=7, random_state=0)
def get_mae(max_leaf_nodes,max_depth,min_samples_split,train_X,val_X,train_y,val_y):
	model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes,max_depth=max_depth,min_samples_split=min_samples_split,random_state=0);
	model.fit(train_X, train_y);
	preds_val = model.predict(val_X)
	mae =  mean_absolute_error(val_y, preds_val)
	return mae;
''' '''
best_mae = 1e9;
best_depth = 1e9;
best_split = 1e9;
for dep in range(2,50):
	for split in range(2,3):
		cur_mae = get_mae(best_leaf_nodes,dep,split,imputed_train_X_plus,imputed_val_X_plus,train_y,val_y);
		print( dep, split,cur_mae );
		if( cur_mae < best_mae ):
			best_mae = cur_mae;
			best_depth = dep;
			best_split = split;
print(best_mae, best_depth, best_split );
''' '''
best_depth = 21;
best_split = 2;
final_model = RandomForestRegressor(max_leaf_nodes=best_leaf_nodes,max_depth=best_depth,min_samples_split=best_split,random_state=0);
final_model.fit(imputed_X_plus,y);
preds_y = final_model.predict(imputed_test_X_plus);

output = pd.DataFrame({'Id': testData.Id,'SalePrice': preds_y})
output.to_csv('submission.csv', index=False) '''





'''

https://www.kaggle.com/c/home-data-for-ml-course/overview

Note: Not sure, steps are in correct order   | but if you think about which mean value will be most suitable then seem right
first reding the given data split into train and valid
then prepare data and model,validate for tain<->valid
after that select proper final model and predict given<->asked

TODO: if missing values are get replaced by mean of gievn and asked entries then prediction would improved?

'''
import pandas as pd;
filePath1 = 'home-data-for-ml-course/train.csv';
filePath2 = 'home-data-for-ml-course/test.csv';
########################################           Prepare Data         ##################################################
fileData1 = pd.read_csv(filePath1);
fileData2 = pd.read_csv(filePath2);
#Remove Rows form missing traget['SalePrice']
fileData1 = fileData1.dropna(axis=0,subset=['SalePrice']);
#Extract X and y from known mapping
X_given = fileData1.drop(['SalePrice'],axis=1);
y_given = fileData1.SalePrice;
X_asked = fileData2;
#-------------------------------#-------------------------#----------------------------------#---------------------------#
from sklearn.preprocessing import LabelEncoder;
X_given_copy = X_given.copy();
X_asked_copy = X_asked.copy();
object_col = [];
for col in X_given.columns:
	if( X_given[col].dtype == 'object' and X_given[col].isnull().any() == 0 and X_asked[col].isnull().any() == 0 ):
		object_col.append(col);
label_encoder = LabelEncoder();
print( '#object_col...',object_col,sep='\n');
for col in object_col:
	X_given_copy[col] = label_encoder.fit_transform(X_given[col]);
	X_asked_copy[col] = label_encoder.transform(X_asked[col]);
X_given = X_given_copy;
X_asked = X_asked_copy;
#------------------------------#--------------------------#----------------------------------#----------------------------#
#Remove Irrelevent Column form X, [ In this case only numeric cols will be taken]
X_given = X_given.select_dtypes(exclude=['object']);    # see: X_given[col_i].dtype
X_asked = X_asked.select_dtypes(exclude=['object']);
#Remove Cols If there are a lot of missing entries >= 35-50 % in geivn or asked data [ Here we will not remove anything ]
print('# For X_geivn...',X_given.shape,sep='\n');                # Shape of training data (num_rows, num_columns)
missing_val_count_by_column_X_given = (X_given.isnull().sum());  # Number of missing values in each column of training data
print(missing_val_count_by_column_X_given[missing_val_count_by_column_X_given > 0]);
print('# For X_asked...',X_asked.shape,sep='\n');                # Shape of training data (num_rows, num_columns)
missing_val_count_by_column_X_asked = (X_asked.isnull().sum());  # Number of missing values in each column of training data
print(missing_val_count_by_column_X_asked[missing_val_count_by_column_X_asked > 0]);
#Do imputation with extra col that value was not present nitially [ because we will train model wich given data so asked missing col not imp ]
cols_with_missing = [ col for col in X_given.columns if X_given[col].isnull().any() ];
X_given_copy  = X_given.copy();  # have to make copy before adding the new col
X_asked_copy  = X_asked.copy();
for col in cols_with_missing :
	X_given_copy[ col + '_was_missing' ] = X_given_copy[col].isnull();
	X_asked_copy[ col + '_was_missing' ] = X_asked_copy[col].isnull();
X_given = X_given_copy;
X_asked = X_asked_copy;
print('# For X_geivn...',X_given.shape,sep='\n');  
print('# For X_asked...',X_asked.shape,sep='\n');
#Imputation
from sklearn.impute import SimpleImputer;
my_imputer = SimpleImputer();
imputed_X_given = pd.DataFrame(my_imputer.fit_transform(X_given));  # fit : clculate relenet data to be fitted, transform : fill data
imputed_X_asked = pd.DataFrame(my_imputer.transform(X_asked));      # fill data learned form given data
# Imputation removed column names; put them back 
imputed_X_given.columns = X_given.columns;
imputed_X_asked.columns = X_asked.columns;
X_given = imputed_X_given;
X_asked = imputed_X_asked;
#######################################            model and validate          #################################################
# Break off geiven data into train and valid
from sklearn.model_selection import train_test_split;
X_train, X_valid, y_train, y_valid = train_test_split(X_given, y_given, train_size=0.8, test_size=0.2,random_state=0)
from sklearn.ensemble import RandomForestRegressor;
from sklearn.metrics import mean_absolute_error;
def get_mae(max_leaf_nodes,X_train,X_valid,y_train,y_valid):
	model = RandomForestRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0);
	model.fit(X_train, y_train);
	preds_val = model.predict(X_valid)
	mae =  mean_absolute_error(y_valid, preds_val)
	return mae;
best_mae = 1e9;
best_nodes = 1e9;
for cur_nodes in range(620,640,1):
	cur_mae = get_mae(cur_nodes,X_train,X_valid,y_train,y_valid);
	if( cur_mae < best_mae ):
		best_mae = cur_mae;
		best_nodes = cur_nodes;
	print( cur_nodes, cur_mae );
print('best_mae:', best_mae,'best_nodes:',best_nodes);
#best_mae: 18322.96307980756 best_nodes: 561
#####################################               predict 	               ##################################################
final_model = RandomForestRegressor(max_leaf_nodes=best_nodes, random_state=0);  
final_model.fit(X_given,y_given);
preds_y = final_model.predict(X_asked);

output = pd.DataFrame({'Id': fileData2.Id,'SalePrice': preds_y});
output.to_csv('submission.csv', index=False);





