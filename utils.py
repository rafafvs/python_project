import os

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import holidays

problem_title = "Bike count prediction"
_target_column_name = "log_bike_count"
# A type (class) which will be used to create wrapper objects for y_pred


def get_cv(X, y, random_state=0):
    cv = TimeSeriesSplit(n_splits=8)
    rng = np.random.RandomState(random_state)

    for train_idx, test_idx in cv.split(X):
        # Take a random sampling on test_idx so it's that samples are not consecutives.
        yield train_idx, rng.choice(test_idx, size=len(test_idx) // 3, replace=False)


def get_train_data(path="data/train.parquet"):
    data = pd.read_parquet(path)
    # Sort by date first, so that time based cross-validation would produce correct results
    data = data.sort_values(["date", "counter_name"])
    y_array = data[_target_column_name].values
    X_df = data.drop([_target_column_name, "bike_count"], axis=1)
    return X_df, y_array

def _encode_dates(X):
    lockdown_periods = (
        ["17-03-2020", "11-05-2020"],
        ["30-10-2020", "15-12-2020"],
        ["03-04-2021", "03-05-2021"],
    )

    lockdown_periods = [
        [pd.to_datetime(start, dayfirst=True), pd.to_datetime(end, dayfirst=True)]
        for start, end in lockdown_periods
    ]

    X = X.copy()  # modify a copy of X

    # Encode the date information from the date column
    X["year"] = X["date"].dt.year
    X["month"] = X["date"].dt.month
    X["day"] = X["date"].dt.day
    X["weekday"] = X["date"].dt.weekday + 1
    X["hour"] = X["date"].dt.hour

    X["is_weekend"] = (X["weekday"] > 5).astype(int)
    X["is_holiday"] = (
        X["date"].apply(lambda x: 1 if x in holidays.FR() else 0).astype(int)
    )
    X["is_lockdown"] = (
        X["date"].apply(
            lambda x: any(start <= x <= end for start, end in lockdown_periods)
        )
    ).astype(int)

    return X.drop(columns=["date"])
