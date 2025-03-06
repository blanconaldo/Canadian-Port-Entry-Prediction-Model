import pandas as pd


def load_dataset(filename):
    try:
        # Read the CSV file
        df = pd.read_csv(filename)

        return df

    except FileNotFoundError:
        print(f"Error: File not found")
        raise
    except Exception as e:
        print(f"An error occurred while loading the file: {str(e)}")
        raise