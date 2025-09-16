# 🐾 Proyecto Machine Learning – Predicción de Adopciones en Refugios de Animales

Este proyecto sigue la metodología **CRISP-DM** y se desarrolló con el framework **Kedro**, para predecir la probabilidad de adopción de animales en refugios de EE.UU.  

---

## 📌 Objetivos
- Analizar los registros de ingresos (intakes) y egresos (outcomes) de animales.
- Identificar patrones que influyen en la adopción o en otros desenlaces (retorno, eutanasia, etc.).


---

## ⚙️ Fases del Proyecto

### 1. **Business Understanding**
- Definimos el problema: ¿qué factores influyen en que un animal sea adoptado?
- Stakeholders: refugios, veterinarios, adoptantes y la comunidad.
- Impacto: mejorar la gestión y aumentar las adopciones.

### 2. **Data Understanding**
- Exploración de tres datasets principales:
  - **Intakes** (animales ingresados).
  - **Outcomes** (resultado del egreso).
  - **Licenses** (registro de licencias, opcional).
- Análisis con boxplots, outliers, nulos y matriz de correlación.
- Detección de relaciones entre edad, especie, sexo, condición y outcome.

### 3. **Data Preparation**
- Limpieza de datos nulos.
- Estandarización de columnas (fechas, edad en días y años).
- Unión de datasets con `Animal ID`.
- Tratamiento de outliers en edades y estancias.

### 4. **Feature Engineering**
- Creación de variables nuevas:
  - Categorías de edad (`cachorro`, `joven`, `adulto`, `senior`).
  - Categorías de estancia (`corto`, `medio`, `largo`, `muy largo`).
  - Estacionalidad (`intake_season`).
  - Codificación de variables categóricas con `LabelEncoder`.

### 5. **Modeling**
- Modelos implementados:
  - **Regresión Logística**.
  - **Random Forest** (mejor desempeño).
- Métricas obtenidas:
  - Regresión logística: accuracy ≈ 78%.
  - Random Forest: accuracy ≈ 94%, AUC ≈ 0.99.

---

## 📊 Herramientas
- **Python 3.12**
- **Kedro** (pipelines reproducibles).
- **Scikit-learn** (modelos ML).
- **Matplotlib / Seaborn** (visualización).
- **Kedro-Viz** (visualización de pipelines).
- **GitHub** (control de versiones).

---

## 📌 Resultados Clave
- Los **perros y gatos** son los más adoptados.
- Los **animales jóvenes** tienen mayor probabilidad de adopción.
- La **duración de la estancia** influye directamente: estancias largas reducen las adopciones.
- Random Forest es el modelo recomendado para predicción en producción.

---
