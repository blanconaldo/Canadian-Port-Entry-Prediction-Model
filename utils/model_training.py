from sklearn.model_selection import train_test_split
import xgboost as xgb
import lightgbm as lgb
from lightgbm.callback import log_evaluation


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