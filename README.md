# Entry Into Canadian Ports Prediction Model

## Project Description
This Streamlit application analyzes and predicts entry patterns into Canadian ports. It provides a comprehensive data pipeline for processing historical port entry data, generating synthetic data, and making predictions using machine learning models. The system helps understand trends and patterns in port entries, which can be valuable for resource allocation and planning.

## Installation
Before running this application, ensure you have Python 3.10 installed along with the following dependencies:
```
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.2.0
xgboost>=2.0.0
lightgbm>=4.0.0
joblib>=1.3.0
plotly>=5.13.0
matplotlib>=3.7.0
streamlit>=1.27.0
python-dateutil>=2.8.2
pytz>=2023.3
numba>=0.56.4
```

To install the required packages, run:
```bash
pip install -r requirements.txt
```

# Data

The originial data was found at the Open Government Portal of Canada at https://search.open.canada.ca/opendata/
The approach for normalizing the data and handling the outliers was logarithmic transformation.
Fake data was created by knowing what exactly is important in our features. For example, when creating fake date values, we focused on giving more importance to weekends and months in the summer, and the correlation was showed in the plots.

# Basic Usage

1. **Starting the Application**
```bash
streamlit run Home.py
```

2. **Navigation**
- Use the sidebar to navigate between different pages
- Follow the sequential order of pages for best results:
  1. Data Metrics
  2. Data Preprocessing
  3. Feature Engineering
  4. Algorithm Selection
  5. Model Training
  6. Augmented Data Analysis

3. **Data Flow**
- The application maintains state between pages
- Each page processes data and passes it to the next
- Results from previous steps are required for subsequent operations

4. **Viewing Results**
- Each page displays relevant metrics and visualizations
- Progress is automatically saved in the session state
- Key metrics and plots are shown after each processing step

5. **Model Predictions**
- After completing the pipeline, predictions can be viewed in the Model Training page
- Results include performance metrics and visualization of predictions

Note: The application is designed to run sequentially. Ensure you complete each step before moving to the next page for optimal results.

## Troubleshooting
- If you encounter a "NoneType" error, return to the home page and restart the sequence
- Ensure all required data files are in the correct directory
- Check that all dependencies are properly installed


## Features
The application includes several components:
- Data metrics analysis and visualization
- Data preprocessing and cleaning
- Data augmentation for enhanced analysis
- Feature engineering
- Algorithm selection and model comparison
- Model training and evaluation
- Augmented data analysis
- Interactive visualizations

## Getting Started
1. Clone the repository
2. Install the required dependencies
3. Run the Streamlit application:
```bash
streamlit run Home.py
```

## Application Structure
The application follows a sequential workflow:
1. Data Metrics: Overview and analysis of port entry data
2. Data Preprocessing: Data cleaning and preparation
3. Data Augmentation: Enhancement of dataset
4. Feature Engineering: Creation and transformation of features
5. Algorithm Selection: Model selection and evaluation
6. Model Training: Training and validation of selected model
7. Augmented Data Analysis: Application of augmented data
