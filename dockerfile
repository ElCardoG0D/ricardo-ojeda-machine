# ============================================================
# 🐍 BASE IMAGE
# ============================================================
FROM python:3.10-slim

# ============================================================
# 🧰 VARIABLES DE ENTORNO
# ============================================================
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AIRFLOW_HOME=/opt/airflow

# ============================================================
# 📦 DEPENDENCIAS DEL SISTEMA
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    python3-dev \
    libxml2-dev \
    libxslt-dev \
    rustc \
    cargo \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 📂 CREAR DIRECTORIOS DE AIRFLOW Y PROYECTO
# ============================================================
RUN mkdir -p /opt/airflow /opt/airflow/dags /opt/airflow/logs /opt/airflow/plugins /opt/airflow/project
WORKDIR /opt/airflow/project

# ============================================================
# 📋 COPIAR REQUERIMIENTOS SI EXISTE
# ============================================================
COPY requirements.txt ./

# ============================================================
# 🧠 INSTALAR LIBRERÍAS PYTHON (bloques separados para estabilidad)
# ============================================================
RUN pip install --upgrade pip setuptools wheel

# --- Fijar numpy < 2 (evita incompatibilidades con PyArrow y sklearn) ---
RUN pip install "numpy==1.26.4"

# --- Airflow y Kedro base ---
RUN pip install \
    apache-airflow==2.9.3 \
    kedro==0.19.12 \
    kedro-viz==11.1.0 \
    --no-cache-dir

# --- Kedro Datasets completos ---
RUN pip install \
    kedro-datasets==1.8.0 \
    kedro-datasets[pandas] \
    kedro-datasets[parquet] \
    kedro-datasets[excel] \
    kedro-datasets[json] \
    kedro-datasets[spark] \
    kedro-datasets[sql] \
    --no-cache-dir

# --- DVC, ML y librerías científicas ---
RUN pip install \
    "dvc[all]" \
    scikit-learn==1.5.2 \
    xgboost \
    lightgbm \
    pandas==2.2.2 \
    matplotlib \
    seaborn \
    joblib \
    pyarrow==16.1.0 \
    boto3 \
    sqlalchemy \
    openpyxl \
    fastparquet \
    --no-cache-dir

# ============================================================
# 👤 CREAR USUARIO AIRFLOW (no root)
# ============================================================
RUN useradd -ms /bin/bash airflow
USER airflow

# ============================================================
# 🌀 COMANDO POR DEFECTO
# ============================================================
CMD ["bash", "-c", "airflow db init && airflow webserver & airflow scheduler"]
