# ============================================================
# BASE OFICIAL DE APACHE AIRFLOW
# ============================================================
FROM apache/airflow:2.7.2-python3.10

# ============================================================
# PASAR A ROOT PARA INSTALAR DEPENDENCIAS
# ============================================================
USER root

# 🔧 Dependencias necesarias para XGBoost, LightGBM y Scikit-Learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    cmake \
    build-essential \
    libssl-dev \
    libffi-dev \
    libopenblas-dev \
    libatlas-base-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# ESTRUCTURA DE DIRECTORIOS
# ============================================================
RUN mkdir -p /opt/airflow/project && \
    mkdir -p /opt/airflow/dags

# Copiar DAGs de Airflow
COPY airflow/dags/ /opt/airflow/dags/

# Copiar TODO el proyecto Kedro al contenedor
COPY . /opt/airflow/project/

# Ajustar permisos (importante para Airflow)
RUN chown -R airflow: /opt/airflow

# ============================================================
# VOLVER AL USUARIO airflow
# ============================================================
USER airflow

# ============================================================
# INSTALAR DEPENDENCIAS (requirements.txt)
# ============================================================
COPY requirements.txt /opt/airflow/requirements.txt

# Evitar problemas de instalación
RUN pip install --upgrade pip setuptools wheel

# Instalar todo el ecosistema que usa tu proyecto
RUN pip install --no-cache-dir -r /opt/airflow/requirements.txt

# ============================================================
# INICIALIZACIÓN AUTOMÁTICA (DB + USUARIO + SCHEDULER + WEBSERVER)
# ============================================================
CMD ["bash", "-c", "\
    airflow db init && \
    airflow users create \
        --username admin \
        --password 1234 \
        --firstname Admin \
        --lastname User \
        --role Admin \
        --email admin@example.com || true && \
    airflow scheduler & \
    airflow webserver \
"]
