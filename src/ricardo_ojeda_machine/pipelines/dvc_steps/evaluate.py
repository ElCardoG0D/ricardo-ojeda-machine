import pandas as pd
import os

def main():
    print("=== RUNNING EVALUATION ===")

    clf_path = "07_model_output/classification_results.csv"
    reg_path = "07_model_output/regression_results.csv"

    df_clf = pd.read_csv(clf_path)
    df_reg = pd.read_csv(reg_path)

    os.makedirs("08_reporting", exist_ok=True)

    summary = pd.concat([
        df_clf.assign(type="classification"),
        df_reg.assign(type="regression")
    ])

    summary.to_csv("08_reporting/evaluation_summary.csv", index=False)

    print("Saved: 08_reporting/evaluation_summary.csv")

if __name__ == "__main__":
    main()
