import pandas as pd
import argparse
import os
from glob import glob


# Example terminal usage:
# python script_name.py data/ cleaned_data/

def process_chembl_data(input_csv, output_csv):
    try:
        # Try reading with utf-8 first, then fallback
        try:
            df = pd.read_csv(input_csv, on_bad_lines='skip')
        except UnicodeDecodeError:
            df = pd.read_csv(input_csv, encoding='ISO-8859-1', on_bad_lines='skip')

        # Strip whitespace from column names
        df.columns = df.columns.str.strip()

        # Try with semicolon separator if default fails
        if not {'Molecule ChEMBL ID', 'Standard Value', 'Smiles'}.issubset(df.columns):
            df = pd.read_csv(input_csv, sep=';', encoding='ISO-8859-1', on_bad_lines='skip')
            df.columns = df.columns.str.strip()

        required_columns = ['Molecule ChEMBL ID', 'Standard Value', 'Smiles']

        if not all(col in df.columns for col in required_columns):
            print(f"Skipping {input_csv}: required columns not found.")
            return

        df = df[required_columns]
        df = df.dropna()
        df.to_csv(output_csv, index=False)
        print(f"Processed data saved to {output_csv}")
    except Exception as e:
        print(f"Failed to process {input_csv}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process multiple ChEMBL dataset CSV files in a folder.")
    parser.add_argument("data_folder", help="Path to the folder containing CSV files")
    parser.add_argument("output_folder", help="Path to the folder where processed CSV files will be saved")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    csv_files = glob(os.path.join(args.data_folder, "*.csv"))

    for input_csv in csv_files:
        filename = os.path.basename(input_csv)
        output_csv = os.path.join(args.output_folder, filename)
        process_chembl_data(input_csv, output_csv)
