from utils.data_loader import *
from utils.data_processor import *
from utils.feature_engineering import *
from utils.model_training import *
from utils.model_evaluation import *


def main():
    # Define file name
    filename = "canada_port_entries.csv"

    try:
        # Load the dataset
        df = load_dataset(filename)

        # Handle duplicates and missing values
        df = handle_data(df)

        # Process date features
        df = process_date(df)

        # Transform target variable
        df = process_target(df)

        # Process categorical variables
        df = process_categorical(df)

        # Split the data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

        # Print feature information
        print("\nFeature Information:")
        print("-" * 50)
        print(f"Number of features: {X_train.shape[1]}")
        print("\nFeature names:")
        print(X_train.columns.tolist())

        # Train XGBoost model
        print("\nTraining XGBoost model...")
        xgb_model = train_xgboost(X_train, X_val, y_train, y_val)

        # Evaluate XGBoost
        xgb_pred, xgb_rmse, xgb_r2 = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
        feature_importance(xgb_model, X_train, "XGBoost")

        # Train LightGBM model
        print("\nTraining LightGBM model...")
        lgb_model = train_lightgbm(X_train, X_val, y_train, y_val)

        # Evaluate LightGBM
        lgb_pred, lgb_rmse, lgb_r2 = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
        feature_importance(lgb_model, X_train, "LightGBM")

        # Saving best performing model
        lgb_metrics = {'rmse': lgb_rmse, 'r2': lgb_r2}
        save_model_info(lgb_model, X_train, lgb_metrics, 'lightgbm')

        # Compare models
        print("\nModel Comparison:")
        print("-" * 50)
        print(f"XGBoost RMSE: {xgb_rmse:.4f}, R²: {xgb_r2 * 100:.2f}%")
        print(f"LightGBM RMSE: {lgb_rmse:.4f}, R²: {lgb_r2 * 100:.2f}%")

        if lgb_r2 > xgb_r2:
            print("Model chosen: LightGBM since it's the higher performing model with")
        else:
            print("Model chosen: XGBoost since it's the higher performing model" )


    except Exception as e:
        print(f"An error occurred: {str(e)}")


if __name__ == "__main__":
    main()
