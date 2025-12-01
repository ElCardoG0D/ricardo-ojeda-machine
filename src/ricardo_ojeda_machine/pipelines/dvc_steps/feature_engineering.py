import pandas as pd
import os

def main():
    print("=== RUNNING FEATURE ENGINEERING ===")

    input_path = "data/02_intermediate/intakes_outcomes_clean.csv"
    output_path = "data/05_model_input/features_dataset.csv"

    df = pd.read_csv(input_path)

    if "DateTime_intake" in df.columns and "DateTime_outcome" in df.columns:
        df["DateTime_intake"] = pd.to_datetime(df["DateTime_intake"])
        df["DateTime_outcome"] = pd.to_datetime(df["DateTime_outcome"])

        df["length_of_stay_days"] = (
            df["DateTime_outcome"] - df["DateTime_intake"]
        ).dt.days.fillna(0)

    os.makedirs("data/05_model_input", exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Saved features to: {output_path}")

if __name__ == "__main__":
    main()
