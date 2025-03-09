import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib


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


def evaluate_predictions(y_true, y_pred):
    """Calculate various metrics for model evaluation"""
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred)
    }
    return metrics\
