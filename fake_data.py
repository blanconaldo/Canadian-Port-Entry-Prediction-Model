from functions import *
import random
from datetime import datetime, timedelta

def analyze_real_data(real_data_file):
    """Analyze patterns in real data to inform fake data generation"""
    df = load_dataset(real_data_file)

    # Calculate overall statistics
    overall_mean = df['Sum of Volume'].mean()
    min_volume = max(0, df['Sum of Volume'].min())  # Changed: ensure non-negative minimum
    max_volume = df['Sum of Volume'].quantile(0.99)  # Changed: use 99th percentile instead of max

    # Calculate port-specific statistics
    port_stats = df.groupby('Port of Entry')['Sum of Volume'].agg(['mean', 'std']).to_dict()

    # Calculate mode-specific statistics
    mode_stats = df.groupby('Mode')['Sum of Volume'].agg(['mean', 'std']).to_dict()

    # Calculate seasonal patterns
    monthly_factors = df.groupby(pd.to_datetime(df['Date']).dt.month)['Sum of Volume'].mean()
    # Changed: normalize monthly factors
    monthly_factors = monthly_factors / monthly_factors.mean()

    # Calculate port-mode joint probabilities
    port_mode_counts = df.groupby(['Port of Entry', 'Mode']).size()
    port_mode_probs = (port_mode_counts / len(df)).to_dict()

    # Create region mapping
    region_mapping = df.groupby('Port of Entry')['Region'].first().to_dict()

    return {
        'port_stats': port_stats,
        'mode_stats': mode_stats,
        'monthly_factors': monthly_factors,
        'port_mode_probs': port_mode_probs,
        'region_mapping': region_mapping,
        'overall_mean': overall_mean,
        'min_volume': min_volume,  # Changed: using new min_volume
        'max_volume': max_volume   # Changed: using new max_volume
    }


def generate_realistic_date(start_date=datetime(2025, 1, 1), end_date=datetime(2026, 12, 31)):
    """Generate a random date within the specified range with weekday bias"""
    days = (end_date - start_date).days
    random_date = start_date + timedelta(days=random.randint(0, days))

    # Add bias towards weekdays (70% weekdays, 30% weekends)
    while random_date.weekday() >= 5 and random.random() < 0.7:
        random_date = start_date + timedelta(days=random.randint(0, days))

    return random_date

def generate_realistic_volume(date, port, mode, stats):
    """Generate realistic volume based on historical patterns"""
    try:
        # Get port-specific parameters
        port_mean = stats['port_stats']['mean'][port]
        port_std = stats['port_stats']['std'][port]

        # Calculate log-space parameters
        if port_mean > 0:
            mu = np.log(port_mean)
            sigma = np.sqrt(np.log1p((port_std/port_mean)**2))
        else:
            mu = np.log(1)
            sigma = 0.1

        # Generate base volume using lognormal distribution with proper parameters
        base_volume = np.random.lognormal(
            mean=mu,
            sigma=sigma
        )

        # Apply scaled adjustments
        mode_factor = min(stats['mode_stats']['mean'][mode] / stats['overall_mean'], 2.0)

        # Extract month from date and get seasonal factor
        month_num = date.month  # Extract month number from datetime object
        seasonal_factor = stats['monthly_factors'][month_num] / stats['monthly_factors'].mean()

        # Calculate final volume with controlled multiplication
        final_volume = base_volume * np.sqrt(mode_factor * seasonal_factor)

        # Clip to historical bounds with buffer
        min_vol = max(stats['min_volume'], 0)  # Ensure non-negative minimum
        max_vol = min(stats['max_volume'], port_mean * 10)  # Limit maximum to 10x port mean

        return int(np.clip(final_volume, min_vol, max_vol))

    except Exception as e:
        print(f"Error generating volume for port {port} and mode {mode}: {str(e)}")
        return int(stats['overall_mean'])


def generate_port_mode_pair(stats):
    """Generate a realistic port-mode combination"""
    port_mode_pair = random.choices(
        list(stats['port_mode_probs'].keys()),
        weights=list(stats['port_mode_probs'].values())
    )[0]
    return port_mode_pair[0], port_mode_pair[1]

def generate_fake_data(real_data_file, num_samples=96000):
    """Generate fake data based on patterns in real data"""
    # Load and analyze real data patterns
    stats = analyze_real_data(real_data_file)

    data = []
    for _ in range(num_samples):
        date = generate_realistic_date()
        port, mode = generate_port_mode_pair(stats)
        volume = generate_realistic_volume(date, port, mode, stats)

        data.append({
            'Date': date,
            'Port of Entry': port,
            'Region': stats['region_mapping'][port],
            'Mode': mode,
            'Sum of Volume': volume
        })

    # Create DataFrame and sort by date
    df = pd.DataFrame(data)
    df = df.sort_values('Date')

    return df

def validate_generated_data(real_df, fake_df):
    """Compare distributions and patterns between real and generated data"""
    # Compare volume distributions
    print("\nVolume Distribution Comparison:")
    print("-" * 50)
    print("Real data:")
    print(real_df['Sum of Volume'].describe())
    print("\nFake data:")
    print(fake_df['Sum of Volume'].describe())

    # Compare mode distributions
    print("\nMode Distribution Comparison:")
    print("-" * 50)
    print("Real data:")
    print(real_df['Mode'].value_counts(normalize=True))
    print("\nFake data:")
    print(fake_df['Mode'].value_counts(normalize=True))

    # Compare region distributions
    print("\nRegion Distribution Comparison:")
    print("-" * 50)
    print("Real data:")
    print(real_df['Region'].value_counts(normalize=True))
    print("\nFake data:")
    print(fake_df['Region'].value_counts(normalize=True))

    # Compare monthly patterns
    real_monthly = real_df.groupby(pd.to_datetime(real_df['Date']).dt.month)['Sum of Volume'].mean()
    fake_monthly = fake_df.groupby(pd.to_datetime(fake_df['Date']).dt.month)['Sum of Volume'].mean()

    print("\nMonthly Volume Pattern Comparison:")
    print("-" * 50)
    print("Real data monthly averages:")
    print(real_monthly)
    print("\nFake data monthly averages:")
    print(fake_monthly)

if __name__ == "__main__":
    # Example usage
    real_data_file = "canada_port_entries.csv"

    try:
        # Generate fake data
        print("Generating fake data...")
        fake_df = generate_fake_data(real_data_file)

        # Load real data for validation
        real_df = load_dataset(real_data_file)

        # Validate the generated data
        validate_generated_data(real_df, fake_df)

        # Save the fake data
        # output_file = 'fake_data.csv'
        # fake_df.to_csv(output_file, index=False)
        # print(f"\nFake data saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")