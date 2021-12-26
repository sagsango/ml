# https://www.kaggle.com/sagsango/exercise-cross-validation/edit
import pandas as pd
from sklearn.model_selection import train_test_split

# Read the data
train_data = pd.read_csv('home-data-for-ml-course/train.csv', index_col='Id')
test_data = pd.read_csv('home-data-for-ml-course/test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = train_data.SalePrice              
train_data.drop(['SalePrice'], axis=1, inplace=True)

# Select numeric columns only
numeric_cols = [cname for cname in train_data.columns if train_data[cname].dtype in ['int64', 'float64']]
X = train_data[numeric_cols].copy()
X_test = test_data[numeric_cols].copy()

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])

from sklearn.model_selection import cross_val_score
# Multiply by -1 since sklearn calculates *negative* MAE
def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.
    
    Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    # Replace this body with your own code
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators=n_estimators, random_state=0))
        ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=3,
                              scoring='neg_mean_absolute_error')
    return scores.mean();
    
   
results = dict(());
for n_estimators in range(50,401,50):
    print( "for n_estimators:",n_estimators,".......");
    results[n_estimators]=(get_score(n_estimators) );

# Graph betwwen n_estimator(no of trees in forest) and mean(mae of 3 folt cv)
import matplotlib.pyplot as plt
plt.plot(list(results.keys()), list(results.values()))
plt.show()

