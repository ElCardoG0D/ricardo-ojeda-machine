import pandas as pd
import os

def main():
    print("=== RUNNING DATA PREPARATION ===")

    # Archivos reales del proyecto
    intakes_path = "data/01_raw/Austin_Animal_Center_Intakes__10_01_2013_to_05_05_2025_.csv"
    outcomes_path = "data/01_raw/Austin_Animal_Center_Outcomes__10_01_2013_to_05_05_2025_.csv"
    licenses_path = "data/01_raw/Seattle_Pet_Licenses.csv"

    # Directorio intermedio
    out_dir = "data/02_intermediate"
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------
    # 1) Cargar RAW datasets
    # ---------------------------
    df_intakes = pd.read_csv(intakes_path)
    df_outcomes = pd.read_csv(outcomes_path)
    df_licenses = pd.read_csv(licenses_path)

    # ---------------------------
    # 2) Transformaciones simples
    # ---------------------------
    df_intakes["source"] = "intake"
    df_outcomes["source"] = "outcome"
    df_licenses["source"] = "license"

    # Guardar archivos transformados (lo que DVC espera)
    intakes_out = os.path.join(out_dir, "intakes_transformed.csv")
    outcomes_out = os.path.join(out_dir, "outcomes_transformed.csv")
    licenses_out = os.path.join(out_dir, "licenses_transformed.csv")

    df_intakes.to_csv(intakes_out, index=False)
    df_outcomes.to_csv(outcomes_out, index=False)
    df_licenses.to_csv(licenses_out, index=False)

    print("Saved transformed files:")
    print(" →", intakes_out)
    print(" →", outcomes_out)
    print(" →", licenses_out)

    # ---------------------------
    # 3) Crear dataset combinado
    # ---------------------------
    df_combined = pd.concat([df_intakes, df_outcomes], axis=0)

    combined_out = os.path.join(out_dir, "intakes_outcomes_clean.csv")
    df_combined.to_csv(combined_out, index=False)

    print("Saved combined file:")
    print(" →", combined_out)

if __name__ == "__main__":
    main()
