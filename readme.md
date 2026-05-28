# Pipeline MLOps - Modelo Predictivo de Riesgo Crediticio

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115.4-009688.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.40.1-FF4B4B.svg)
![MLOps](https://img.shields.io/badge/MLOps-CI%2FCD%20Ready-8A2BE2.svg)

---

## 🗺️ Índice del Proyecto
- [Pipeline MLOps - Modelo Predictivo de Riesgo Crediticio](#pipeline-mlops---modelo-predictivo-de-riesgo-crediticio)
  - [🗺️ Índice del Proyecto](#️-índice-del-proyecto)
  - [📌 Descripción del Proyecto](#-descripción-del-proyecto)
  - [🔍 Hallazgos Clave del Análisis Exploratorio de Datos (EDA)](#-hallazgos-clave-del-análisis-exploratorio-de-datos-eda)
  - [🧠 Decisiones Clave del Modelado](#-decisiones-clave-del-modelado)
  - [🏗️ Arquitectura del Proyecto](#️-arquitectura-del-proyecto)
  - [📁 Estructura del Repositorio](#-estructura-del-repositorio)
  - [📋 Requisitos Previos](#-requisitos-previos)
  - [🚀 Instalación y Ejecución Local](#-instalación-y-ejecución-local)
    - [1. Clonar el repositorio y configurar el entorno](#1-clonar-el-repositorio-y-configurar-el-entorno)
    - [2. Ejecutar la API REST de forma nativa](#2-ejecutar-la-api-rest-de-forma-nativa)
    - [3. Ejecutar mediante contenedores Docker](#3-ejecutar-mediante-contenedores-docker)
  - [🔌 Endpoints de la API](#-endpoints-de-la-api)
    - [Métodos Disponibles:](#métodos-disponibles)
  - [📊 Pipeline de MLOps](#-pipeline-de-mlops)
    - [Monitoreo de Data Drift (Streamlit)](#monitoreo-de-data-drift-streamlit)

---

## 📌 Descripción del Proyecto

Este repositorio contiene un pipeline completo de Machine Learning Operations (MLOps) diseñado para el desarrollo, empaquetado y monitoreo de un modelo predictivo supervisado de clasificación binaria. El objetivo de negocio es anticipar la probabilidad de impago/mora de nuevos clientes (identificando la variable objetivo `Pago_atiempo` donde `0` es moroso y `1` es un buen pagador).

El sistema cuenta con un procesamiento robusto de variables financieras, una arquitectura de microservicios contenerizada para inferencia en tiempo real y una interfaz analítica interactiva encargada de auditar la estabilidad de la población en producción, quedando completamente preparado para su integración automática dentro de herramientas de integración continua como Jenkins.

---

## 🔍 Hallazgos Clave del Análisis Exploratorio de Datos (EDA)

El análisis matemático, bivariable y multivariable sobre el conjunto de datos históricos destapó características críticas de la población, sirviendo de justificación técnica para el posterior diseño de las transformaciones:

* **Variables de Alto Poder Discriminante:** Se identificó que `puntaje_datacredito`, `edad_cliente` y `huella_consulta` poseen la mayor capacidad de separación analítica para segmentar a los clientes morosos.
* **Depuración de Anomalías y Placeholders:** Se descubrió que la columna `puntaje` original actuaba como un falso predictor al contener un valor estático idéntico en el 87.4% de las filas, justificando su eliminación. De igual manera, se aislaron puntajes negativos imposibles en el buró de crédito para ser tratados en el pipeline de limpieza.
* **Presencia de Multicolinealidad:** La variable `cuota_pactada` presentó una correlación lineal casi perfecta ($r = 0.995$) con la combinación de `capital_prestado` y `plazo_meses`, amenazando la estabilidad de los modelos lineales si no era descartada.
* **Falta de Variabilidad:** La columna `saldo_mora_codeudor` se identificó como un vector de varianza cero al estar constituida exclusivamente por ceros y registros nulos, aportando nulo valor predictivo.
* **Sesgo de Distribución:** Los montos y saldos financieros mostraron una distribución asimétrica severa hacia la derecha, haciendo obligatorio el uso de transformaciones matemáticas logarítmicas.

---

## 🧠 Decisiones Clave del Modelado

* **Estrategia ante Desbalance Extremo:** Dado que los registros de morosidad representan únicamente el ~4.75% de la base de datos, el entrenamiento evitó métricas engañosas como el *Accuracy*. Los clasificadores se optimizaron con base en el **PR-AUC (Precision-Recall AUC)** y el F1-Score enfocados en la clase minoritaria (`0`), empleando parámetros de pesos balanceados (`class_weight`, `scale_pos_weight`) en lugar de generar filas sintéticas artificiales.
* **Calibración Dinámica de Umbrales:** Se desacopló el umbral estándar de decisión del 50%. Mediante optimización iterativa, se determinó un umbral específico encapsulado dentro del artefacto final para maximizar la detección de impagos sin disparar falsos positivos masivos.
* **Encapsulamiento del Estado:** El preprocesamiento de ingeniería de características fitteado (`ft_pipeline.pkl`) viaja de manera independiente al modelo matemático seleccionado (`model.pkl`), asegurando que las transformaciones se ejecuten en producción de manera idéntica al entorno de entrenamiento.

---

## 🏗️ Arquitectura del Proyecto

El sistema está estructurado como una solución modular e inmutable de tres capas:
1. **Capa de Transformación y Datos (Core):** Un pipeline basado en `scikit-learn` y `feature-engine` que recibe los datos tabulares limpios y construye las variables numéricas y categóricas requeridas.
2. **Capa de Servicio (Backend):** Una API REST de alto rendimiento construida sobre FastAPI que consume los archivos serializados y expone puntos de acceso HTTP protegidos para inferencias en tiempo real.
3. **Capa de Monitoreo (Frontend):** Una aplicación de tablero interactivo que procesa la información entrante en producción para evaluar el desvío poblacional.

---

## 📁 Estructura del Repositorio

Para cumplir con las políticas automatizadas de infraestructura en producción (pipelines de Jenkins), se respeta la siguiente estructura de carpetas:

```text
mlops_pipeline/
├── src/
│   ├── Cargar_datos.ipynb              # Fase 1: Ingesta y validación de fuentes
│   ├── comprension_eda.ipynb           # Fase 2: Análisis Exploratorio de Datos (EDA)
│   ├── ft_engineering.py               # Fase 3: Pipeline de Ingeniería de Características
│   ├── model_training_evaluation.py    # Fase 4: Modelado, optimización y evaluación
│   ├── model_deploy.py                 # Fase 5: API REST con FastAPI
│   └── model_monitoring.py             # Fase 6: Dashboard interactivo de Streamlit
├── Base_de_datos.csv                   # Dataset original de entrenamiento
├── Dockerfile                          # Empaquetado e infraestructura del contenedor
├── requirements.txt                    # Declaración explícita de dependencias de Python
├── .gitignore                          # Reglas de exclusión para datos y artefactos temporales
└── README.md                           # Documentación técnica general (este archivo)

```

---

## 📋 Requisitos Previos

* Python 3.12 o superior instalado.
* Docker Desktop (opcional, para despliegue contenerizado).
* Memoria RAM mínima recomendada: 8 GB.

---

## 🚀 Instalación y Ejecución Local

### 1. Clonar el repositorio y configurar el entorno

```bash
# Crear un entorno virtual
python -m venv .venv

# Activar el entorno virtual (Windows)
.\venv\Scripts\activate

# Instalar el árbol completo de dependencias
pip install -r requirements.txt

```

### 2. Ejecutar la API REST de forma nativa

```bash
# Iniciar el servidor local de desarrollo mediante Uvicorn
uvicorn src.model_deploy:app --host 0.0.0.0 --port 8000 --reload

```

### 3. Ejecutar mediante contenedores Docker

```bash
# Construir la imagen de Docker basada en la receta del Dockerfile
docker build -t mlops-credit .

# Instanciar el contenedor mapeando los puertos de red
docker run -p 8000:8000 mlops-credit

```

---

## 🔌 Endpoints de la API

Con la API en ejecución, puedes acceder a la interfaz gráfica interactiva de documentación (Swagger UI) a través de la ruta: `http://localhost:8000/docs`.

### Métodos Disponibles:

* **`GET /`**: Endpoint de diagnóstico de salud (*healthcheck*), verifica que los artefactos del modelo e ingeniería estén correctamente cargados en memoria.
* **`POST /predict`**: Endpoint principal de inferencia. Recibe un objeto JSON con las características crudas de un nuevo cliente, procesa la información de manera transparente a través del pipeline e implementa el umbral calibrado para clasificar si el cliente se encuentra `"al_dia"` o `"moroso"`.

---

## 📊 Pipeline de MLOps

### Monitoreo de Data Drift (Streamlit)

Para garantizar la estabilidad del modelo a lo largo del tiempo, se incluye un panel analítico basado en el **Population Stability Index (PSI)**, algoritmo estándar de la industria bancaria y crediticia internacional.

Este componente mide de forma cuantitativa el cambio en la distribución de las variables clave del mercado real frente al comportamiento original del entrenamiento:

```bash
# Lanzar el servidor e interfaz web del dashboard de monitoreo
streamlit run src/model_monitoring.py

```

* **Interpretación del Semáforo PSI:**
* **PSI < 0.1:** Verde 🟢 (Población estable; el modelo sigue siendo confiable).
* **0.1 ≤ PSI < 0.2:** Amarillo 🟡 (Desviación moderada; requiere supervisión del equipo analítico).
* **PSI ≥ 0.2:** Rojo 🔴 (Data Drift crítico detectado; se requiere ejecutar un reentrenamiento preventivo del modelo).