import pandas as pd
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def clean_data(file_path, output_path):
    data = pd.read_csv(file_path)
    logging.info(f"Initial data shape: {data.shape}")

    # Attempt to clean non-numeric characters from numeric columns
    for col in data.columns:
        data[col] = data[col].replace({',': '', '$': ''}, regex=True)  # Remove commas and dollar signs
        data[col] = pd.to_numeric(data[col], errors='coerce')  # Convert to numeric, NaN if conversion fails

    logging.info(f"Data types after conversion:\n{data.dtypes}")

    # Drop columns with non-numeric data and fill missing values
    numeric_data = data.select_dtypes(include=[np.number])
    numeric_data.fillna(numeric_data.mean(), inplace=True)

    logging.info(f"Numeric data shape after removing non-numeric columns: {numeric_data.shape}")

    if numeric_data.shape[1] == 0:
        logging.error("The data must contain at least one numeric column.")
        raise ValueError("The data must contain at least one numeric column.")

    numeric_data.to_csv(output_path, index=False)
    logging.info(f"Cleaned data saved to {output_path}")

# Example usage:
input_file = 'market_data.csv'
output_file = 'market_data_cleaned_auto.csv'

clean_data(input_file, output_file)