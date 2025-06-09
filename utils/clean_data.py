import pandas as pd
import argparse
import os
from glob import glob


# Example terminal usage:
# python sc#ript_name.py data/ cleaned_data/


def process_chembl_data(input_csv: str, output_csv: str) -> None:
    try:
        # Próba wczytania z różnymi kodowaniami i separatorami
        try:
            df: pd.DataFrame = pd.read_csv(input_csv, on_bad_lines="skip")
        except UnicodeDecodeError:
            df = pd.read_csv(input_csv, encoding="ISO-8859-1", on_bad_lines="skip")

        # Czyszczenie nazw kolumn
        df.columns = df.columns.str.strip().str.replace('"', '')

        # Jeśli brakuje wymaganych kolumn, spróbuj wczytać z separatorem średnikowym
        required_columns = ["Molecule ChEMBL ID", "Standard Value", "Smiles"]
        if not set(required_columns).issubset(df.columns):
            try:
                df = pd.read_csv(input_csv, sep=";", encoding="ISO-8859-1", on_bad_lines="skip")
                df.columns = df.columns.str.strip().str.replace('"', '')
            except Exception as e:
                print(f"Failed to read with semicolon separator: {e}")

        # Dodatkowe kolumny, które chcemy zachować
        additional_columns = [
            "Source Description",
            "Assay Description",
            "Assay Type",
            "Assay Organism",
            "Assay Tissue Name",
            "Document Journal",
            "Document Year",
            "Target Name",
            "Standard Type",
            "Standard Units",
            "pChEMBL Value",
            "Molecular Weight",
            "AlogP"
        ]

        # Sprawdź, które dodatkowe kolumny istnieją w danych
        available_additional = [col for col in additional_columns if col in df.columns]

        # Jeśli nadal brakuje wymaganych kolumn, zakończ przetwarzanie
        if not set(required_columns).issubset(df.columns):
            print(f"Skipping {input_csv}: required columns not found.")
            print(f"Available columns: {df.columns.tolist()}")
            return

        # Zachowaj wymagane i dostępne dodatkowe kolumny
        columns_to_keep = required_columns + available_additional
        df = df[columns_to_keep]

        # Usuń wiersze z brakującymi wartościami w wymaganych kolumnach
        df = df.dropna(subset=required_columns)

        # Konwersja typów danych dla kolumn liczbowych
        numeric_cols = ["Standard Value", "Molecular Weight", "AlogP"]
        for col in numeric_cols:
            if col in df.columns:
                # Usuń cudzysłowie jeśli istnieją
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace('"', '')
                # Konwertuj na liczby, obsługując błędy
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Zapisz przetworzone dane
        df.to_csv(output_csv, index=False)
        print(f"Processed data saved to {output_csv}")
        print(f"Columns kept: {columns_to_keep}")

    except Exception as e:
        print(f"Failed to process {input_csv}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process multiple ChEMBL dataset CSV files in a folder."
    )
    parser.add_argument("data_folder", help="Path to the folder containing CSV files")
    parser.add_argument(
        "output_folder",
        help="Path to the folder where processed CSV files will be saved",
    )
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    csv_files = glob(os.path.join(args.data_folder, "*.csv"))

    for input_csv in csv_files:
        filename = os.path.basename(input_csv)
        output_csv = os.path.join(args.output_folder, filename)
        process_chembl_data(input_csv, output_csv)
