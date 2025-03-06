from utils.streamlit_pages import *
import plotly.graph_objects as go
from fake_data import *

# Check prerequisites
if not check_page_prerequisites(6):
    st.warning("Please complete previous pages first!")
    st.stop()

# Add this page to visited pages
add_page_number(6)

st.title("Data Generation and Validation")
st.markdown("This page generates synthetic data based on patterns from the real dataset and validates the results.")

try:
    # Load real data
    real_data_file = "canada_port_entries.csv"
    with st.spinner("Loading real data..."):
        real_df = load_dataset(real_data_file)
        st.success("Real data loaded successfully!")

    # Generate fake data
    st.subheader("Generating Synthetic Data")
    with st.spinner("Generating synthetic data based on real patterns..."):
        fake_df = generate_fake_data(real_data_file)
        st.success("Synthetic data generated successfully!")

    # Display data comparison
    st.subheader("Data Comparison")

    # 1. Volume Distribution Comparison
    st.markdown("### Volume Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Real Data Statistics**")
        st.dataframe(real_df['Sum of Volume'].describe())

    with col2:
        st.markdown("**Synthetic Data Statistics**")
        st.dataframe(fake_df['Sum of Volume'].describe())

    # Volume distribution plot
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Histogram(x=real_df['Sum of Volume'], name='Real Data', opacity=0.7))
    fig_vol.add_trace(go.Histogram(x=fake_df['Sum of Volume'], name='Synthetic Data', opacity=0.7))
    fig_vol.update_layout(title='Volume Distribution Comparison',
                         barmode='overlay',
                         xaxis_title='Volume',
                         yaxis_title='Count')
    st.plotly_chart(fig_vol)

    # 2. Mode Distribution Comparison
    st.markdown("### Mode Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Real Data Mode Distribution**")
        st.dataframe(real_df['Mode'].value_counts(normalize=True))

    with col2:
        st.markdown("**Synthetic Data Mode Distribution**")
        st.dataframe(fake_df['Mode'].value_counts(normalize=True))

    # Mode distribution plot
    fig_mode = go.Figure()
    fig_mode.add_trace(go.Bar(
        x=real_df['Mode'].value_counts(normalize=True).index,
        y=real_df['Mode'].value_counts(normalize=True).values,
        name='Real Data'
    ))
    fig_mode.add_trace(go.Bar(
        x=fake_df['Mode'].value_counts(normalize=True).index,
        y=fake_df['Mode'].value_counts(normalize=True).values,
        name='Synthetic Data'
    ))
    fig_mode.update_layout(title='Mode Distribution Comparison',
                          barmode='group',
                          xaxis_title='Mode',
                          yaxis_title='Proportion')
    st.plotly_chart(fig_mode)

    # 3. Region Distribution Comparison
    st.markdown("### Region Distribution")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Real Data Region Distribution**")
        st.dataframe(real_df['Region'].value_counts(normalize=True))

    with col2:
        st.markdown("**Synthetic Data Region Distribution**")
        st.dataframe(fake_df['Region'].value_counts(normalize=True))

    # 4. Monthly Patterns
    st.markdown("### Monthly Volume Patterns")
    real_monthly = real_df.groupby(pd.to_datetime(real_df['Date']).dt.month)['Sum of Volume'].mean()
    fake_monthly = fake_df.groupby(pd.to_datetime(fake_df['Date']).dt.month)['Sum of Volume'].mean()

    fig_monthly = go.Figure()
    fig_monthly.add_trace(go.Scatter(
        x=real_monthly.index,
        y=real_monthly.values,
        name='Real Data',
        mode='lines+markers'
    ))
    fig_monthly.add_trace(go.Scatter(
        x=fake_monthly.index,
        y=fake_monthly.values,
        name='Synthetic Data',
        mode='lines+markers'
    ))
    fig_monthly.update_layout(title='Monthly Volume Pattern Comparison',
                            xaxis_title='Month',
                            yaxis_title='Average Volume')
    st.plotly_chart(fig_monthly)

    # Store generated data in session state
    st.session_state.synthetic_data = fake_df
    st.success("Data generation and validation completed successfully!")

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
