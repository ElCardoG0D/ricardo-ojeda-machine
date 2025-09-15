import pandas as pd

# 1) Limpiar datos
def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Elimina nulos básicos y duplicados"""
    df = df.drop_duplicates()
    df = df.dropna(how="all")
    return df

# 2) Transformar fechas
def transformar_fechas(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte columnas de fecha a datetime si existen"""
    for col in df.columns:
        if "Date" in col or "date" in col or "DateTime" in col:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def unir_datasets(intakes: pd.DataFrame, outcomes: pd.DataFrame, licenses: pd.DataFrame):
    """Une intakes, outcomes y licenses estandarizando nombres de columnas."""

    # Normalizar nombres de columnas (minúsculas, sin espacios)
    for df in [intakes, outcomes, licenses]:
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    # Confirmar que existe la columna
    print("📌 Columnas intakes:", intakes.columns.tolist())
    print("📌 Columnas outcomes:", outcomes.columns.tolist())
    print("📌 Columnas licenses:", licenses.columns.tolist())

    if "animal_id" not in intakes.columns or "animal_id" not in outcomes.columns:
        raise KeyError("❌ No se encontró la columna 'animal_id' en intakes u outcomes")

    # Merge principal
    merged = pd.merge(
        intakes, outcomes,
        on="animal_id", how="inner",
        suffixes=("_intake", "_outcome")
    )

    # Merge con licenses si tiene la columna
    if "animal_id" in licenses.columns:
        merged = pd.merge(merged, licenses, on="animal_id", how="left")
    else:
        print("⚠️ Licenses no tiene columna 'animal_id', se omite el merge.")

    return merged

