import numpy as np
import pandas as pd


# 1. Transform Date
def process_date(df):
    # 3. Transform the data
    # Date transformation
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month.astype(np.int8)
    df['day'] = df['Date'].dt.day.astype(np.int8)
    df['day_of_week'] = df['Date'].dt.dayofweek.astype(np.int8)
    df['is_weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(np.int8)
    df = df.drop('Date', axis=1)
    return df


# 2. Target Variable Treatment
def process_target(df):

    # Create log transformation of volume
    df['volume_log'] = np.log1p(df['Sum of Volume'] + 63, dtype=np.float32)

    # Remove the original volume column
    df = df.drop('Sum of Volume', axis=1)

    print(df['volume_log'].head())
    return df


def process_categorical(df):
    categorical_columns = ['Port of Entry', 'Region', 'Mode']
    df_encoded = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns, dtype=np.int8)
    return df_encoded