import numpy as np
import pandas as pd
from utils.streamlit_pages import *

# Check prerequisites
if not check_page_prerequisites(3):
    st.warning("Please complete previous pages first!")
    st.stop()

# Add this page to visited pages
add_page_number(3)

st.title("Feature Engineering")
st.markdown("This page performs automated feature engineering on the dataset.")

try:
    # Check if processed dataframe exists in session state
    if 'df_processed' not in st.session_state or st.session_state.df_processed is None:
        st.error("Please complete the Data Preprocessing page first!")
        st.stop()

    # Get the processed dataframe from session state
    df = st.session_state.df_processed.copy()

    # Show original data info
    st.subheader("Original Data Overview")
    st.write("Original Shape:", df.shape)
    st.write("Original Columns:", df.columns.tolist())

    # 1. Date Transformation
    st.markdown("### 1. Date Transformation")
    st.info("Converting date column into year, month, day, day_of_week, and is_weekend features...")

    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month.astype(np.int8)
    df['day'] = df['Date'].dt.day.astype(np.int8)
    df['day_of_week'] = df['Date'].dt.dayofweek.astype(np.int8)
    df['is_weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(np.int8)
    df = df.drop('Date', axis=1)

    st.success("Date transformation completed!")

    # 2. Target Variable Treatment
    st.markdown("### 2. Target Variable Treatment")
    st.info("Applying log transformation to Sum of Volume...")

    df['volume_log'] = np.log1p(df['Sum of Volume'] + 63, dtype=np.float32)
    df = df.drop('Sum of Volume', axis=1)

    st.success("Target variable transformation completed!")

    # 3. Categorical Encoding
    st.markdown("### 3. Categorical Encoding")
    st.info("Applying one-hot encoding to categorical variables...")

    categorical_columns = ['Port of Entry', 'Region', 'Mode']
    df = pd.get_dummies(df, columns=categorical_columns, prefix=categorical_columns, dtype=np.int8)

    st.success("Categorical encoding completed!")

    # Show final data info
    st.subheader("Final Data Overview")
    st.write("Final Shape:", df.shape)
    st.write("Final Columns:", df.columns.tolist())
    st.write("Preview of transformed data:")
    st.write(df.head())

    # Save transformed data
    st.session_state.df_engineered = df
    st.success("Feature engineering completed! You can now proceed to the next page.")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")