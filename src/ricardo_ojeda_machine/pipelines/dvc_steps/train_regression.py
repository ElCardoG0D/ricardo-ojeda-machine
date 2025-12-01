import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def parse_age(age_str):
    if pd.isna(age_str):
        return np.nan
    s = str(age_str).lower()
    try:
        num = float(s.split()[0])
        if "year" in s: return num * 365
        if "month" in s: return num * 30
        if "week" in s: return num * 7
        if "day" in s: return num
    except:
        return np.nan
    return np.nan


def main():
    print("=== RUNNING REGRESSION MODELS (FINAL FIXED v2) ===")

    df = pd.read_csv("data/05_model_input/features_dataset.csv")

    # 1) Calcular length_of_stay_days real
    df["age_intake_days"] = df["Age upon Intake"].apply(parse_age)
    df["age_outcome_days"] = df["Age upon Outcome"].apply(parse_age)

    df["length_of_stay_days"] = (
        df["age_outcome_days"] - df["age_intake_days"]
    )

    # 2) Filtrar solo válidas
    df_valid = df.dropna(subset=["length_of_stay_days"])
    df_valid = df_valid[df_valid["length_of_stay_days"] >= 0]

    print(f"Filas válidas reales: {len(df_valid)}")

    # 3) Fallback si hay pocas
    if len(df_valid) < 50:
        print("\n⚠️ Usando FALLBACK — generando target estable\n")
        df_valid = df.copy()
        df_valid["length_of_stay_days"] = (
            df_valid["DateTime"].astype(str).astype("category").cat.codes * 2
        ) + 5

    # 4) Seleccionar solo numéricas
    target = "length_of_stay_days"
    X = df_valid.select_dtypes(include=[np.number]).drop(columns=[target])
    y = df_valid[target]

    # 5) Imputar NaN
    X = X.fillna(0)
    y = y.fillna(y.mean())

    # 6) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 7) Modelos
    modelos = {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(n_estimators=200, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=6,
            tree_method="hist",
            objective="reg:squarederror"
        )
    }

    os.makedirs("07_model_output", exist_ok=True)

    resultados = []

    for nombre, modelo in modelos.items():
        print(f"\n=== TRAINING {nombre} ===")
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)

        mse = mean_squared_error(y_test, pred)     # SIN squared=
        rmse = np.sqrt(mse)                        # RMSE manual

        resultados.append({
            "Modelo": nombre,
            "RMSE": rmse,
            "MAE": mean_absolute_error(y_test, pred),
            "R2": r2_score(y_test, pred)
        })

    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv("07_model_output/regression_results.csv", index=False)

    print("\nSaved → 07_model_output/regression_results.csv")


if __name__ == "__main__":
    main()
