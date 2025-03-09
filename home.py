from utils.streamlit_pages import *

# Initialize session state for page tracking if not exists
if 'visited_pages' not in st.session_state:
    st.session_state.visited_pages = set()

# Page configuration
st.set_page_config(
    page_title="Entry into Canadian Ports",
    page_icon="🇨🇦",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        /* Main title styling */
        .main-title {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 42px;
            font-weight: 700;
            color: #1E3D59;
            margin-bottom: 30px;
            text-align: center;
        }

        /* Subtitle styling */
        .subtitle {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 24px;
            font-weight: 500;
            color: #17A2B8;
            margin-bottom: 20px;
        }

        /* Section header styling */
        .section-header {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 28px;
            font-weight: 600;
            color: #2C3E50;
            margin-top: 30px;
            margin-bottom: 15px;
        }

        /* Body text styling */
        .body-text {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 18px;
            line-height: 1.6;
            color: #34495E;
            margin-bottom: 15px;
        }

        /* Highlight text */
        .highlight {
            background-color: #F7F9FC;
            padding: 20px;
            border-radius: 5px;
            border-left: 5px solid #17A2B8;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.markdown("### Navigation")
    st.markdown("Current page: **Home**")

# Main content
st.markdown('<h1 class="main-title">Entry into Canadian Ports</h1>', unsafe_allow_html=True)

st.markdown('<p class="subtitle">Welcome to the Canadian Ports Entry Analysis System</p>', unsafe_allow_html=True)

# Overview section
st.markdown('<h2 class="section-header">Overview</h2>', unsafe_allow_html=True)
st.markdown('''
    <div class="body-text">
    This application provides comprehensive analysis and modeling of entry patterns into Canadian ports.
    Navigate through different sections to explore data, perform analysis, and view predictions.
    </div>
    ''', unsafe_allow_html=True)

# Application Structure section
st.markdown('<h2 class="section-header">Application Structure</h2>', unsafe_allow_html=True)
st.markdown('''
    <div class="highlight">
    <p class="body-text">The application is organized into the following sections:</p>
    <ol class="body-text">
        <li><strong>Data Metrics:</strong> Overview and analysis of port entry data</li>
        <li><strong>Data Preprocessing:</strong> Data cleaning and preparation</li>
        <li><strong>Data Augmentation:</strong> Enhancement of dataset</li>
        <li><strong>Feature Engineering:</strong> Creation and transformation of features</li>
        <li><strong>Algorithm Selection:</strong> Model selection and evaluation</li>
        <li><strong>Model Training:</strong> Training and validation of selected model</li>
        <li><strong>Augmented Data App:</strong> Application of augmented data analysis</li>
    </ol>
    </div>
    ''', unsafe_allow_html=True)

# Instructions section
st.markdown('<h2 class="section-header">Instructions</h2>', unsafe_allow_html=True)
st.markdown('''
    <div class="body-text">
    <ul>
        <li>Navigate through pages in the specified order using the sidebar</li>
        <li>Each page builds upon the previous one</li>
        <li>Ensure completion of each step before proceeding to the next</li>
        <li>View results and visualizations at each stage</li>
    </ul>
    </div>
    ''', unsafe_allow_html=True)

# Data Information section
st.markdown('<h2 class="section-header">Dataset Information</h2>', unsafe_allow_html=True)
st.markdown('''
    <div class="highlight body-text">
    The analysis is based on historical data of entries into Canadian ports, including:
    <ul>
        <li>Port of entry locations</li>
        <li>Entry volumes</li>
        <li>Transportation modes</li>
        <li>Regional distribution</li>
        <li>Temporal patterns</li>
        <li>Data was gather from the Open Government Portal of Canada at https://search.open.canada.ca/opendata/</li>
    </ul>
    </div>
    ''', unsafe_allow_html=True)

# Initialize first page as visited
add_page_number(0)

# Create .streamlit/config.toml file for page order
import os

config_dir = ".streamlit"
config_file = os.path.join(config_dir, "config.toml")

if not os.path.exists(config_dir):
    os.makedirs(config_dir)

config_content = """
[theme]
primaryColor="#17A2B8"
backgroundColor="#FFFFFF"
secondaryBackgroundColor="#F7F9FC"
textColor="#34495E"

[server]
runOnSave = true

[browser]
gatherUsageStats = false

[pages]
order = [
    "01_Data_metrics.py",
    "02_Data_preprocessing.py",
    "03_Data_augmentation.py",
    "04_Feature_engineering.py",
    "05_Algorithm_selection.py",
    "06_Model_training.py",
    "07_Augmented_data_app.py"
]
"""

with open(config_file, "w") as f:
    f.write(config_content)