import pandas as pd
import logging

logger = logging.getLogger(__name__)

def validate_dataset(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Valida que el dataset no esté vacío y registra su tamaño.
    """
    assert not df.empty, f"El dataset {name} está vacío."
    logger.info(f"{name} cargado con {df.shape[0]} filas y {df.shape[1]} columnas.")
    return df
