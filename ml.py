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
forest_model = RandomForestRegressor(random_state=1);
forest_model.fit(train_X, train_y);
forest_predictions = forest_model.predict(val_X);
print("Mean absolute error using RandomForestRegressor:",mean_absolute_error(val_y, forest_predictions));





