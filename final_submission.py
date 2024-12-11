# Import the required libraries
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import holidays
import utils
import seaborn as sns

from skrub import TableVectorizer
from scipy.stats import uniform, randint

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV,
    KFold,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import root_mean_squared_error

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor


# Import the data (Training data)
train_data = pd.read_parquet(Path("data") / "train.parquet") # Train data
train_data.set_index("date", inplace=True) # Set date as the index

# Import the data (additional weather data)
external_data = pd.read_csv(Path("data") / "external_data.csv") # Add the weather data
external_data["date"] = pd.to_datetime(external_data["date"]) # Convert dates to datetime objects
external_data.set_index("date", inplace=True) # set date as the index

# Remove duplicates and keep first occurance
external_data.drop_duplicates(keep="first", inplace=True)

# Remove features with more than 50% missing values
threshold = 0.5
bool_drop = external_data.isna().mean() >= threshold
external_data.drop(external_data.columns[bool_drop], axis=1, inplace=True)

# Upsample the weather data to hourly frequency
external_data_resampled = external_data.resample("h").interpolate(
    method="time"
)
external_data_resampled = external_data_resampled.reset_index() # reset the index

# Merge the upsampled weather data with the train data on the date
train_merged = pd.merge(train_data, external_data_resampled, how="inner", on="date")

# Encode the dates
train_merged = utils._encode_dates(train_merged)

# Selected columns to train on
train_columns = [
    "counter_id", "latitude", "longitude", "is_holiday", "month",
    "day", "hour", "is_weekend",
    "is_lockdown", # binary feature for covid 19
    "ff",          # wind_speed
    "t",           # temperature_k
    "vv",          # visibility_h
    "ww",          # present_weather
    "n",           # total cloudiness
    "etat_sol",    # soil condition
    "ht_neige",    # total snow height
    "rr1",         # total rain (1h)
]

# Separation of data into features and target
X = train_merged[train_columns]
y = train_merged["log_bike_count"]

# Creating model pipeline:
# Preproccessing - TableVectorizer()
# Model - RandomForestRegressor() with parameters from RandomizedSearchCV()
model = make_pipeline(TableVectorizer(n_jobs=-1), 
                      RandomForestRegressor(n_estimators=100,
                                            max_depth=49,
                                            min_samples_split=5))

# Train test split on the data to fit the model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=3010
)

# Fit the model
model.fit(X_train, y_train)

test_data = pd.read_parquet(Path("data") / "final_test.parquet")
test_data_merged = test_data.merge(external_data_resampled, on="date", how="inner")
test_data_merged = utils._encode_dates(test_data_merged)[X.columns]

submission = model.predict(test_data_merged[X.columns])
pd.Series(submission).to_frame().rename_axis("Id").rename(
    columns={0: "log_bike_count"}
).to_csv("final_submission.csv")