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