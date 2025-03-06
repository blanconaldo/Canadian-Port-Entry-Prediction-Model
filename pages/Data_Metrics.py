from utils.data_loader import *
import plotly.express as px
import pandas as pd
from utils.streamlit_pages import *


if not check_page_prerequisites(1):  # Change number for each page
    st.warning("Please complete previous pages first!")
    st.stop()

add_page_number(1)

df = load_dataset("canada_port_entries.csv")
st.session_state['df'] = df  # Store in session state

# Convert date format
df['Date'] = pd.to_datetime(df['Date']).dt.date  # This will convert to YYYY-MM-DD format

# Set page configuration
st.set_page_config(page_title="Traveler Volume Dashboard", layout="wide")

# Page title
st.title("Traveler Volume Dashboard")
st.subheader("Monitor the volume of travelers across Canadian ports.")

# Filters
region = st.sidebar.multiselect("Select Region:",
                              options=sorted(df["Region"].unique()),
                              default=sorted(df["Region"].unique()))
date = st.sidebar.selectbox("Select Date:",
                           options=sorted(df["Date"].unique()),
                           index=0)

# Filter data
filtered_df = df[(df["Region"].isin(region)) & (df["Date"] == date)]

# Key metrics
col1, col2 = st.columns(2)
with col1:
    st.metric("Total Traveler Volume", f"{filtered_df['Sum of Volume'].sum():,}")
with col2:
    st.metric("Number of Ports", f"{filtered_df['Port of Entry'].nunique()}")

# Bar chart
st.write("### Traveler Volume by Port")
fig = px.bar(
    filtered_df,
    x="Mode",
    y="Sum of Volume",
    color="Region",
    title="Traveler Volume by Mode of Travel",
    labels={"Sum of Volume": "Volume of Travelers"}
)
st.plotly_chart(fig, use_container_width=True)

# Line chart
st.write("### Traveler Volume Trend by Date and Region")
trend_data = df.groupby(["Date", "Region"])["Sum of Volume"].sum().reset_index()
fig_trend = px.line(
    trend_data,
    x="Date",
    y="Sum of Volume",
    color="Region",
    title="Traveler Volume Trend",
    labels={"Sum of Volume": "Volume of Travelers"}
)
st.plotly_chart(fig_trend, use_container_width=True)

# Data table
st.write("### Filtered Data")
st.dataframe(filtered_df)