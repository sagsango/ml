import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
X = pd.read_csv('home-data-for-ml-course/train.csv', index_col='Id') 
X_test = pd.read_csv('home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X.SalePrice
X.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll drop columns with missing values
#cols_with_missing = [col for col in X.columns if X[col].isnull().any()  ]   # Origional 
cols_with_missing = [col for col in X.columns if X[col].isnull().any() or X_test[col].isnull().any() ]
X.drop(cols_with_missing, axis=1, inplace=True)
X_test.drop(cols_with_missing, axis=1, inplace=True)

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# function for comparing different approaches
def score_dataset(cur_nodes,X_train, X_valid, y_train, y_valid):
    model = RandomForestRegressor(max_leaf_nodes=cur_nodes,n_estimators=100, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    return mean_absolute_error(y_valid, preds);
    
# Columns that are object 
object_cols = [ col for col in X_train if X_train[col].dtype == 'object' ];
# Columns that will be one-hot encoded
low_cardinality_cols = [col for col in object_cols if X_train[col].nunique() < 10]

# Columns that will be dropped from the dataset
high_cardinality_cols = list(set(object_cols)-set(low_cardinality_cols))
from sklearn.preprocessing import OneHotEncoder

object_cols = [ col for col in X_train if X_train[col].dtype == 'object'];

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(X_train[low_cardinality_cols]))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(X_valid[low_cardinality_cols]))

# One-hot encoding removed index; put it back
OH_cols_train.index = X_train.index.copy();
OH_cols_valid.index = X_valid.index.copy();

object_cols = [col for col in X_train if X_train[col].dtype == 'object']
# Remove categorical columns (will replace with one-hot encoding)
num_X_train = X_train.drop(object_cols, axis=1)
num_X_valid = X_valid.drop(object_cols, axis=1)

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([num_X_valid, OH_cols_valid], axis=1)


best_nodes = 1e9;
best_mae = 1e9;
for cur_nodes in range(2,1000,50):
	cur_mae = score_dataset(cur_nodes,OH_X_train, OH_X_valid, y_train, y_valid);
	if( cur_mae < best_mae ):
		best_mae = cur_mae;
		best_nodes = cur_nodes;
	print( cur_nodes, cur_mae);
print( best_nodes, best_mae );
#640 17477.973912198056


OH_cols_test = pd.DataFrame(OH_encoder.transform(X_test[low_cardinality_cols]));
OH_cols_test.index = X_test.index;
num_X_test = X_test.drop(object_cols,axis=1);
OH_X_test = pd.concat([num_X_test,OH_cols_test],axis=1);

best_nodes = 640
fmodel = RandomForestRegressor(max_leaf_nodes=best_nodes,n_estimators=100, random_state=0)
fmodel.fit(OH_X_train,y_train)
preds = fmodel.predict(OH_X_test)

output = pd.DataFrame({'Id': X_test.index,'SalePrice': preds});
output.to_csv('submission.csv', index=False);


	


