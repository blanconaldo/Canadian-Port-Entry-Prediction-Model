import joblib

def load_model(model_path='../trained_lightgbm_model_info.joblib'):
    """Load the trained model and its information"""
    try:
        model_info = joblib.load(model_path)
        print("Model loaded successfully")
        return model_info
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

