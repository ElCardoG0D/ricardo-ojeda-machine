# Este archivo permite que la carpeta 'pipelines' sea un paquete Python.
# No debe importar nada directamente.

from .pipeline import create_pipeline

__all__ = ["create_pipeline"]