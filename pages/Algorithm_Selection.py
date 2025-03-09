from utils.model_training import *
from utils.model_evaluation import *
from utils.streamlit_pages import *

# Check prerequisites
if not check_page_prerequisites(4):
    st.warning("Please complete previous pages first!")
    st.stop()

# Add this page to visited pages
add_page_number(4)

st.title("Algorithm Selection")
st.markdown("This page handles model training and selection between XGBoost and LightGBM.")

# Warning about XGBoost training time
st.warning("Note: XGBoost training might take approximately 5 minutes to complete.")

# Information about data splitting
st.info("Data will be split into: 70% training, 15% validation, and 15% test sets.")

try:
    # Get engineered data from session state
    if 'df_engineered' not in st.session_state:
        st.error("Please complete the feature engineering page first!")
        st.stop()

    df = st.session_state.df_engineered

    # Split the data
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Display feature information
    st.subheader("Feature Information")
    st.write(f"Number of features: {X_train.shape[1]}")
    st.write("Feature names:", X_train.columns.tolist())

    col1, col2 = st.columns(2)

    # Train XGBoost model
    with col1:
        if st.button("Train XGBoost Model"):
            with st.spinner("Training XGBoost model... (This may take about 5 minutes)"):
                xgb_model = train_xgboost(X_train, X_val, y_train, y_val)
                xgb_pred, xgb_rmse, xgb_r2 = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
                xgb_importance = feature_importance(xgb_model, X_train, "XGBoost")

                # Store results in session state
                st.session_state.xgb_results = {
                    'model': xgb_model,
                    'rmse': xgb_rmse,
                    'r2': xgb_r2,
                    'importance': xgb_importance
                }
                st.success("XGBoost training completed!")

                # Show XGBoost results
                st.write(f"XGBoost RMSE: {xgb_rmse:.4f}")
                st.write(f"XGBoost R²: {xgb_r2 * 100:.2f}%")

    # Train LightGBM model
    with col2:
        if st.button("Train LightGBM Model"):
            with st.spinner("Training LightGBM model..."):
                lgb_model = train_lightgbm(X_train, X_val, y_train, y_val)
                lgb_pred, lgb_rmse, lgb_r2 = evaluate_model(lgb_model, X_test, y_test, "LightGBM")
                lgb_importance = feature_importance(lgb_model, X_train, "LightGBM")

                # Store results in session state
                st.session_state.lgb_results = {
                    'model': lgb_model,
                    'rmse': lgb_rmse,
                    'r2': lgb_r2,
                    'importance': lgb_importance
                }
                st.success("LightGBM training completed!")

                # Show LightGBM results
                st.write(f"LightGBM RMSE: {lgb_rmse:.4f}")
                st.write(f"LightGBM R²: {lgb_r2 * 100:.2f}%")

    # Only show model comparison and save best model if both models have been trained
    if 'xgb_results' in st.session_state and 'lgb_results' in st.session_state:
        st.subheader("Model Comparison")

        # Create comparison table
        comparison_df = pd.DataFrame({
            'Metric': ['RMSE', 'R²'],
            'XGBoost': [
                f"{st.session_state.xgb_results['rmse']:.4f}",
                f"{st.session_state.xgb_results['r2'] * 100:.2f}%"
            ],
            'LightGBM': [
                f"{st.session_state.lgb_results['rmse']:.4f}",
                f"{st.session_state.lgb_results['r2'] * 100:.2f}%"
            ]
        })
        st.table(comparison_df)

        # Save best performing model
        if st.session_state.lgb_results['r2'] > st.session_state.xgb_results['r2']:
            best_model = "LightGBM"
            metrics = {'rmse': st.session_state.lgb_results['rmse'],
                      'r2': st.session_state.lgb_results['r2']}
            save_model_info(st.session_state.lgb_results['model'],
                          X_train, metrics, 'lightgbm')
        else:
            best_model = "XGBoost"
            metrics = {'rmse': st.session_state.xgb_results['rmse'],
                      'r2': st.session_state.xgb_results['r2']}
            save_model_info(st.session_state.xgb_results['model'],
                          X_train, metrics, 'xgboost')

        st.success(f"Model chosen: {best_model} since it's the higher performing model")
        st.session_state.model_selected = True
    else:
        st.warning("Please train both models to proceed to the next page.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
