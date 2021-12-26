PROJECT_ID = 'kaggle-playground-170215'
BUCKET_NAME = 'automl-tutorial-alexis'

DATASET_DISPLAY_NAME = 'taxi_fare_dataset'
TRAIN_FILEPATH =  "home-data-for-ml-course/train.csv"
TEST_FILEPATH = "home-data-for-ml-course/test.csv"

TARGET_COLUMN = 'SalePrice'
ID_COLUMN = 'Id'

MODEL_DISPLAY_NAME = 'tutorial_model'
TRAIN_BUDGET = 1460

# Import the class defining the wrapper
from automl_tables_wrapper import AutoMLTablesWrapper

# Create an instance of the wrapper
amw = AutoMLTablesWrapper(project_id=PROJECT_ID,
                          bucket_name=BUCKET_NAME,
                          dataset_display_name=DATASET_DISPLAY_NAME,
                          train_filepath=TRAIN_FILEPATH,
                          test_filepath=TEST_FILEPATH,
                          target_column=TARGET_COLUMN,
                          id_column=ID_COLUMN,
                          model_display_name=MODEL_DISPLAY_NAME,
                          train_budget=TRAIN_BUDGET)
# Create and train the model
amw.train_model()
# Get predictions
amw.get_predictions()
submission_df = pd.read_csv("submission.csv")
submission_df.head()


