import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)


def main():
    print("=== RUNNING MODELING ===")

    input_path = "data/05_model_input/features_dataset.csv"
    df = pd.read_csv(input_path)

    target = "Outcome Type"

    # eliminar filas sin target
    df = df.dropna(subset=[target])

    # separar X e y
    X = df.drop(columns=[target])
    y = df[target]

    # codificar target para XGBoost
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # columnas categóricas
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    # columnas con alta cardinalidad
    high_card = [c for c in cat_cols if X[c].nunique() >= 10]

    # columnas con baja cardinalidad
    low_card = [c for c in cat_cols if X[c].nunique() < 10]

    # columnas numéricas
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()

    # Preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("low_card", OneHotEncoder(handle_unknown="ignore"), low_card),
            ("high_card", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), high_card),
        ],
        remainder="drop"
    )

    # modelos
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            tree_method="hist"
        ),
    }

    os.makedirs("07_model_output", exist_ok=True)

    resultados = []

    for name, model in models.items():
        print(f"\n=== TRAINING {name} ===")

        clf = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        clf.fit(X_train, y_train)

        pred = clf.predict(X_test)

        # solo calculamos ROC si es binaria
        prob = None
        if hasattr(clf, "predict_proba") and len(np.unique(y)) == 2:
            prob = clf.predict_proba(X_test)[:, 1]

        resultados.append({
            "Modelo": name,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(y_test, pred, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, pred, average="weighted", zero_division=0),
            "F1": f1_score(y_test, pred, average="weighted", zero_division=0),
            "ROC_AUC": roc_auc_score(y_test, prob) if prob is not None else None
        })

    resultados_df = pd.DataFrame(resultados)
    resultados_df.to_csv("07_model_output/classification_results.csv", index=False)

    print("\nSaved: 07_model_output/classification_results.csv")


if __name__ == "__main__":
    main()
