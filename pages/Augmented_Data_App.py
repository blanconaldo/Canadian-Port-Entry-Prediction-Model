from utils.streamlit_pages import *
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utils.data_loader import *
from utils.data_processor import *
from utils.feature_engineering import *
import numpy as np
import pandas as pd
import joblib

# Check prerequisites
if not check_page_prerequisites(7):
    st.warning("Please complete previous pages first!")
    st.stop()

# Add this page to visited pages
add_page_number(7)

st.title("Augmented Model Training and Evaluation")
st.markdown("This page loads the best model and evaluates its performance on the fake dataset.")

try:
    # Load the model
    st.subheader("Loading Model")
    with st.spinner("Loading the trained model..."):
        try:
            model_info = joblib.load('trained_lightgbm_model_info.joblib')
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()

    # Prepare test data
    st.subheader("Preparing Fake Data")
    with st.spinner("Preparing test data..."):
        # Load and process data
        filename = "fake_data.csv"
        df = load_dataset(filename)
        df = handle_data(df)
        df = process_date(df)
        df = process_target(df)

        # Store original volume
        original_volume = np.exp(df['volume_log']) - 63

        # Process categorical variables
        df = process_categorical(df)
        st.success("Test data prepared successfully!")

    # Make predictions
    st.subheader("Making Predictions")
    with st.spinner("Generating predictions..."):
        model = model_info['model']
        feature_names = model_info['feature_names']

        # Ensure test data has all required features
        missing_cols = set(feature_names) - set(df.columns)
        for col in missing_cols:
            df[col] = 0

        # Select and order features
        X_test = df[feature_names]

        # Make predictions
        predictions = model.predict(X_test)
        predictions_original_scale = np.exp(predictions) - 63
        st.success("Predictions completed!")

    # Calculate metrics
    st.subheader("Model Performance Metrics")
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(original_volume, predictions_original_scale)),
        'MAE': mean_absolute_error(original_volume, predictions_original_scale),
        'R2': r2_score(original_volume, predictions_original_scale)
    }

    # Display metrics in a nice format
    col1, col2, col3 = st.columns(3)
    col1.metric("RMSE", f"{metrics['RMSE']:,.2f}")
    col2.metric("MAE", f"{metrics['MAE']:,.2f}")
    col3.metric("R² Score", f"{metrics['R2']:.4f}")

    # Create visualizations
    st.subheader("Visualization of Results")

    # 1. Scatter plot
    fig_scatter = px.scatter(
        x=original_volume,
        y=predictions_original_scale,
        labels={'x': 'Actual Values', 'y': 'Predicted Values'},
        title='Actual vs Predicted Values'
    )
    fig_scatter.add_trace(
        go.Scatter(x=[original_volume.min(), original_volume.max()],
                  y=[original_volume.min(), original_volume.max()],
                  mode='lines', name='Perfect Prediction',
                  line=dict(color='red', dash='dash'))
    )
    st.plotly_chart(fig_scatter)

    # 2. Distribution of errors
    errors = original_volume - predictions_original_scale
    fig_hist = px.histogram(
        errors,
        title='Distribution of Prediction Errors',
        labels={'value': 'Prediction Error', 'count': 'Frequency'}
    )
    st.plotly_chart(fig_hist)

    # 3. Time series plot
    sample_size = min(100, len(original_volume))
    fig_time = go.Figure()
    fig_time.add_trace(go.Scatter(
        y=original_volume[:sample_size],
        name='Actual',
        mode='lines'
    ))
    fig_time.add_trace(go.Scatter(
        y=predictions_original_scale[:sample_size],
        name='Predicted',
        mode='lines'
    ))
    fig_time.update_layout(
        title='Actual vs Predicted Values (First 100 Samples)',
        xaxis_title='Sample Index',
        yaxis_title='Volume'
    )
    st.plotly_chart(fig_time)

    # Store results in session state
    results_df = pd.DataFrame({
        'Actual_Volume': original_volume,
        'Predicted_Volume': predictions_original_scale,
        'Absolute_Error': abs(original_volume - predictions_original_scale)
    })
    st.session_state.prediction_results = results_df
    st.success("Model evaluation completed successfully!")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
