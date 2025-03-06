import matplotlib.pyplot as plt
import numpy as np


def make_predictions(model_info, test_data):
    """Make predictions using the loaded model"""
    model = model_info['model']
    feature_names = model_info['feature_names']

    # Ensure test data has all required features
    missing_cols = set(feature_names) - set(test_data.columns)
    for col in missing_cols:
        test_data[col] = 0

    # Select and order features according to model's feature names
    X_test = test_data[feature_names]

    # Make predictions
    predictions = model.predict(X_test)

    # Transform predictions back to original scale
    predictions_original_scale = np.exp(predictions) - 63

    return predictions_original_scale



def plot_results(y_true, y_pred):
    """Create visualizations for prediction analysis"""
    # Scatter plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Actual vs Predicted Values')
    plt.tight_layout()
    plt.show()

    # Distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(y_true - y_pred, bins=50, alpha=0.5)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.tight_layout()
    plt.show()

    # Time series plot (for first 100 samples)
    plt.figure(figsize=(15, 6))
    sample_size = min(100, len(y_true))
    plt.plot(range(sample_size), y_true[:sample_size], label='Actual', alpha=0.7)
    plt.plot(range(sample_size), y_pred[:sample_size], label='Predicted', alpha=0.7)
    plt.xlabel('Sample Index')
    plt.ylabel('Volume')
    plt.title('Actual vs Predicted Values (First 100 Samples)')
    plt.legend()
    plt.tight_layout()
    plt.show()