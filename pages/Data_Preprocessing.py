from utils.streamlit_pages import *

# Check prerequisites
if not check_page_prerequisites(2):
    st.warning("Please complete previous pages first!")
    st.stop()

# Add this page to visited pages
add_page_number(2)

# Page title
st.title("Data Preprocessing")
st.markdown("This page handles data cleaning and preprocessing steps.")

# Check if dataframe exists in session state
if 'df' not in st.session_state:
    st.error("Please load the dataset from the Data Metrics page first!")
    st.stop()

# Get the dataframe from session state
df = st.session_state.df.copy()

# Show original data info
st.subheader("Original Data Overview")
col1, col2 = st.columns(2)
with col1:
    st.write("Original Shape:", df.shape)
with col2:
    st.write("Original Memory Usage:", f"{df.memory_usage().sum() / 1024**2:.2f} MB")

# Duplicate Analysis
st.subheader("Duplicate Analysis")
duplicates = df.duplicated().sum()
st.write(f"Number of duplicate rows: {duplicates}")

if duplicates > 0:
    remove_duplicates = st.checkbox("Remove duplicate rows?")
    if remove_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
        st.success(f"Removed {duplicates} duplicate rows. New shape: {df.shape}")

# Missing Values Analysis
st.subheader("Missing Values Analysis")
missing_values = df.isnull().sum()
st.write("Missing values in each column:")
st.write(missing_values)

if missing_values.sum() > 0:
    st.write("Handle missing values:")

    # Categorical columns
    categorical_columns = ['Port of Entry', 'Region', 'Mode']
    for col in categorical_columns:
        if df[col].isnull().any():
            method = st.radio(
                f"Choose method to handle missing values in {col}:",
                ['Mode (Most frequent)', 'Fill with "Unknown"'],
                key=f"missing_{col}"
            )
            if method == 'Mode (Most frequent)':
                df[col] = df[col].fillna(df[col].mode()[0])
                st.success(f"Filled missing values in {col} with mode: {df[col].mode()[0]}")
            else:
                df[col] = df[col].fillna("Unknown")
                st.success(f"Filled missing values in {col} with 'Unknown'")

    # Numerical column
    if df['Sum of Volume'].isnull().any():
        method = st.radio(
            "Choose method to handle missing values in Sum of Volume:",
            ['Median', 'Mean', 'Zero'],
            key="missing_volume"
        )
        if method == 'Median':
            df['Sum of Volume'] = df['Sum of Volume'].fillna(df['Sum of Volume'].median())
            st.success(f"Filled missing values in Sum of Volume with median")
        elif method == 'Mean':
            df['Sum of Volume'] = df['Sum of Volume'].fillna(df['Sum of Volume'].mean())
            st.success(f"Filled missing values in Sum of Volume with mean")
        else:
            df['Sum of Volume'] = df['Sum of Volume'].fillna(0)
            st.success(f"Filled missing values in Sum of Volume with zero")

# Show processed data info
st.subheader("Processed Data Overview")
col1, col2 = st.columns(2)
with col1:
    st.write("New Shape:", df.shape)
with col2:
    st.write("New Memory Usage:", f"{df.memory_usage().sum() / 1024**2:.2f} MB")

# Show sample of processed data
st.subheader("Sample of Processed Data")
st.write(df.head())

# Data Quality Metrics
st.subheader("Data Quality Metrics")
quality_metrics = {
    "Total Records": len(df),
    "Complete Rows": df.dropna().shape[0],
    "Missing Value Rows": len(df) - df.dropna().shape[0],
    "Duplicate Rows": duplicates
}
st.write(quality_metrics)

# Save processed data to session state
if st.button("Save Processed Data"):
    st.session_state.df_processed = df
    st.success("Processed data saved successfully! You can now proceed to the next page.")

# Optional: Download processed data
if st.button("Download Processed Data"):
    csv = df.to_csv(index=False)
    st.download_button(
        label="Download CSV",
        data=csv,
        file_name="processed_data.csv",
        mime="text/csv"
    )