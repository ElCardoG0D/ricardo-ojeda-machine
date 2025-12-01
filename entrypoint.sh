#!/bin/bash
set -e

echo ">>> Inicializando Airflow DB..."
airflow db init

echo ">>> Creando usuario admin si no existe..."
airflow users create \
    --username admin \
    --firstname admin \
    --lastname user \
    --role Admin \
    --email admin@example.com \
    --password 1234 || true

echo ">>> Iniciando Scheduler..."
airflow scheduler &

echo ">>> Iniciando Webserver..."
exec airflow webserver
