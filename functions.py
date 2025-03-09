import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from lightgbm.callback import log_evaluation
import joblib
# import os

# Read the CSV file
def load_dataset(filename):
    try:
        # Read the CSV file
        df = pd.read_csv(filename)

        return df

    except FileNotFoundError:
        print(f"Error: File not found")
        raise
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")
        raise


def handle_data(df):
    # 1. Check and handle duplicates
    print("Shape before removing duplicates:", df.shape)
    duplicates = df.duplicated().sum()
    print(f"Number of duplicate rows: {duplicates}")

    # Remove duplicates
    if duplicates > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print("Shape after removing duplicates:", df.shape)

    # 2. Check missing values
    missing_values = df.isnull().sum()
    print("\nMissing values in each column:")
    print(missing_values)

    # Handle missing values
    # For categorical columns
    categorical_columns = ['Port of Entry', 'Region', 'Mode']
    if missing_values.sum() > 0:
        for col in categorical_columns:
            if df[col].isnull().any():
                # Fill with 'Unknown' or most frequent value
                df[col] = df[col].fillna(df[col].mode()[0])
                print(f"Filled missing values in {col} with mode")

    # For numerical column (Sum of Volume)
    if df['Sum of Volume'].isnull().any():
        # Fill with median because mean is sensitive to outliers
        df['Sum of Volume'] = df['Sum of Volume'].fillna(df['Sum of Volume'].median())
        print("Filled missing values in Sum of Volume with median")
    return df

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

def report(df_encoded):
    """Save transformed dataset and generate reports"""
    # Save the transformed dataset
    # Create a directory for the transformed data if it doesn't exist
    # output_dir = 'transformed_data'
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    #
    # # Save as CSV
    # csv_path = os.path.join(output_dir, 'transformed_dataset.csv')
    # df_encoded.to_csv(csv_path, index=False)
    #
    # # Save as pickle (preserves data types)
    # pickle_path = os.path.join(output_dir, 'transformed_dataset.pkl')
    # df_encoded.to_pickle(pickle_path)
    #
    # # Print information about the saved files
    # print("\nTransformed dataset information:")
    # print(f"Shape of transformed dataset: {df_encoded.shape}")
    # print(f"\nFiles saved:")
    # print(f"1. CSV file: {csv_path}")
    # print(f"2. Pickle file: {pickle_path}")

    # Print reports
    print("\nTransformed dataset information:")
    print(f"Shape of transformed dataset: {df_encoded.shape}")

    print("\nSample of transformed data:")
    print(df_encoded.head())

    print("\nMemory usage of transformed dataset:")
    print(df_encoded.memory_usage(deep=True).sum() / 1024**2, "MB")

    print("\nData Quality Report:")
    print("-" * 50)
    print("1. Number of rows:", len(df_encoded))
    print("2. Number of columns:", len(df_encoded.columns))
    print("3. Missing values after transformation:", df_encoded.isnull().sum().sum())
    print("4. Memory usage:", df_encoded.memory_usage(deep=True).sum() / 1024**2, "MB")
    print("5. Data types:")
    print(df_encoded.dtypes.value_counts())


def analyze_statistics(df):
    """
    Analyze statistical measures of the dataset
    """
    print("\nStatistical Analysis:")
    print("-" * 50)

    # For numeric columns
    numeric_stats = df.select_dtypes(include=[np.number]).describe()
    print("\nNumeric Columns Statistics:")
    print(numeric_stats)

    # For the target variable (Sum of Volume)
    if 'Sum of Volume' in df.columns:
        print("\nTarget Variable (Sum of Volume) Statistics:")
        print(f"Mean: {df['Sum of Volume'].mean():.4f}")
        print(f"Median: {df['Sum of Volume'].median():.4f}")
        print(f"Min: {df['Sum of Volume'].min():.4f}")
        print(f"Max: {df['Sum of Volume'].max():.4f}")
        print(f"Mode: {df['Sum of Volume'].mode().values[0]:.4f}")
        print(f"Standard Deviation: {df['Sum of Volume'].std():.4f}")
        print(f"Variance: {df['Sum of Volume'].var():.4f}")


    # For the target variable (volume_log)
    if 'volume_log' in df.columns:
        print("\nTarget Variable (volume_log) Statistics:")
        print(f"Mean: {df['volume_log'].mean():.4f}")
        print(f"Median: {df['volume_log'].median():.4f}")
        print(f"Min: {df['volume_log'].min():.4f}")
        print(f"Max: {df['volume_log'].max():.4f}")
        print(f"Mode: {df['volume_log'].mode().values[0]:.4f}")
        print(f"Standard Deviation: {df['volume_log'].std():.4f}")
        print(f"Variance: {df['volume_log'].var():.4f}")


    # For categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print("\nCategorical Columns Mode (Most Frequent Values):")
        for col in categorical_cols:
            mode_value = df[col].mode().values[0]
            mode_count = df[col].value_counts().iloc[0]
            mode_percentage = (mode_count / len(df)) * 100
            print(f"\n{col}:")
            print(f"Mode: {mode_value}")
            print(f"Frequency: {mode_count} ({mode_percentage:.2f}%)")
            print("\nValue Counts:")
            print(df[col].value_counts().head())


def split_data(df_encoded, target_column='volume_log', test_size=0.15, val_size=0.15, random_state=42):

    # Split data into training, validation, and test sets.

    # Separate features and target
    X = df_encoded.drop(target_column, axis=1)
    y = df_encoded[target_column]

    # First split: separate test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # Second split: separate validation set from remaining data
    # Adjust validation size to account for the reduced dataset size
    val_size_adjusted = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_adjusted,
        random_state=random_state,
        shuffle=True
    )

    # Print split sizes
    print("\nData Split Summary:")
    print("-" * 50)
    print(f"Total records: {len(df_encoded)}")
    print(f"Training set: {len(X_train)} records ({len(X_train)/len(df_encoded)*100:.1f}%)")
    print(f"Validation set: {len(X_val)} records ({len(X_val)/len(df_encoded)*100:.1f}%)")
    print(f"Test set: {len(X_test)} records ({len(X_test)/len(df_encoded)*100:.1f}%)")

    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgboost(X_train, X_val, y_train, y_val):
    # XGBoost parameters
    xgb_params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.1,
        'max_depth': 6,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }

    # Train XGBoost model
    model_xgb = xgb.XGBRegressor(**xgb_params)
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=100
    )

    return model_xgb

def train_lightgbm(X_train, X_val, y_train, y_val):
    # LightGBM parameters
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.1,
        'num_leaves': 31,
        'max_depth': -1,
        'n_estimators': 1000,
        'early_stopping_rounds': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }

    # Train LightGBM model
    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[log_evaluation(period=100)]
    )

    return model_lgb

def evaluate_model(model, X, y, model_name):
    # Make predictions
    predictions = model.predict(X)

    # Calculate metrics
    mse = mean_squared_error(y, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, predictions)

    print(f"\n{model_name} Evaluation Metrics:")
    print("-" * 50)
    print(f"RMSE: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")

    return predictions, rmse, r2

def feature_importance(model, X_train, model_name):
    if model_name == "XGBoost":
        importance = model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)
    else:  # LightGBM
        importance = model.feature_importances_
        feature_imp = pd.DataFrame({
            'feature': X_train.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

    print(f"\n{model_name} Top 10 Important Features:")
    print("-" * 50)
    print(feature_imp.head(10))

    return feature_imp


def save_model_info(model, X_train, metrics, model_name, save_path=None):
    # If no save path specified, create default
    if save_path is None:
        save_path = f'trained_{model_name}_model_info.joblib'

    # Create model info dictionary
    model_info = {
        'model': model,
        'model_type': model_name,
        'feature_names': X_train.columns.tolist(),
        'metrics': {
            'rmse': metrics['rmse'],
            'r2': metrics['r2']
        },
        'n_features': len(X_train.columns),
        'training_shape': X_train.shape
    }

    # Save the model info
    print(f"\nSaving {model_name.upper()} model and preprocessing information...")
    joblib.dump(model_info, save_path)
    print(f"Model saved as '{save_path}'")

    print("\nModel Performance:")
    print(f"RMSE: {model_info['metrics']['rmse']:.4f}")
    print(f"R² Score: {(model_info['metrics']['r2'] * 100):.2f}%")

    return save_path