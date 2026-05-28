"""
model_deploy.py
===============

API REST con FastAPI para el despliegue del modelo de riesgo crediticio.
Carga los artefactos generados en la Fase 3 y 4, procesa datos crudos a
través del pipeline de feature engineering y devuelve la predicción
basada en el umbral calibrado.

Autor: Harrison
Versión: V1.3.0
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import joblib
import __main__  # Para asegurar que las clases custom estén disponibles para joblib
from pathlib import Path
from typing import Optional, Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager

# Importamos las clases custom para que joblib pueda reconstruir el pipeline
from ft_engineering import DerivedFeaturesCreator, DataCleaner, TipoCreditoGrouper

# <-- NUEVO: Inyectamos las clases en __main__ para que joblib las encuentre
__main__.DerivedFeaturesCreator = DerivedFeaturesCreator
__main__.DataCleaner = DataCleaner
__main__.TipoCreditoGrouper = TipoCreditoGrouper

# ============================================================
# CONSTANTES DE CONFIGURACIÓN
# ============================================================

# Resolución robusta de rutas: subimos un nivel desde src/ hasta la raíz
RAIZ: Path = Path(__file__).parent.parent
RUTA_MODEL: Path = RAIZ / "model.pkl"
RUTA_PIPELINE: Path = RAIZ / "ft_pipeline.pkl"

# Diccionario global para almacenar los artefactos cargados en memoria
state = {}

# ============================================================
# ESQUEMAS DE VALIDACIÓN (Pydantic)
# ============================================================

class CreditoInput(BaseModel):
    """
    Esquema de entrada para el endpoint de predicción.
    Espera las columnas originales del dataset (sin procesar).
    """
    tipo_credito: Any  # Acepta str o int, el pipeline lo agrupa luego
    capital_prestado: float
    plazo_meses: int
    edad_cliente: int
    tipo_laboral: str
    salario_cliente: float
    total_otros_prestamos: float
    cuota_pactada: float
    puntaje: float
    puntaje_datacredito: float
    cant_creditosvigentes: int
    huella_consulta: float
    saldo_mora: float
    saldo_total: float
    saldo_principal: float
    creditos_sectorFinanciero: int
    creditos_sectorCooperativo: int
    creditos_sectorReal: int
    
    # Opcionales (el pipeline sabe cómo imputarlos si llegan como None)
    promedio_ingresos_datacredito: Optional[float] = None
    saldo_mora_codeudor: float = 0.0
    tendencia_ingresos: Optional[str] = None


class PredictionOutput(BaseModel):
    """Esquema de respuesta de la API."""
    prediccion: str
    probabilidad_moroso: float
    umbral_utilizado: float

# ============================================================
# LIFESPAN Y CARGA DE MODELOS
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Maneja el ciclo de vida de la aplicación.
    Carga los artefactos pesados UNA SOLA VEZ al arrancar el servidor.
    """
    if not RUTA_MODEL.exists() or not RUTA_PIPELINE.exists():
        raise RuntimeError(
            f"Faltan artefactos. Verificá que existan:\n"
            f"- {RUTA_MODEL}\n- {RUTA_PIPELINE}"
        )
    
    # Carga en memoria
    state["model_artifact"] = joblib.load(RUTA_MODEL)
    state["pipeline"] = joblib.load(RUTA_PIPELINE)
    
    yield  # La API corre mientras está en este punto
    
    # Limpieza al apagar (opcional)
    state.clear()

# Instanciamos la app
app = FastAPI(
    title="API Riesgo Crediticio - MLOps",
    description="Inferencia en tiempo real con mitigación de desbalance.",
    version="V1.3.0",
    lifespan=lifespan
)

# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/health")
def health_check():
    """Endpoint de monitoreo para asegurar que la API está viva."""
    return {
        "status": "ok", 
        "modelo": "LogisticRegression", 
        "version": "V1.3.0"
    }

@app.post("/predict", response_model=PredictionOutput)
def predict(data: CreditoInput):
    """
    Recibe los datos crudos de un cliente, aplica el pipeline de feature 
    engineering y devuelve la predicción de riesgo crediticio.
    """
    try:
        # 1. Extraemos los artefactos del estado de la app
        pipeline = state["pipeline"]
        artifact = state["model_artifact"]
        model = artifact["model"]
        threshold = artifact["threshold"]
        feat_names = artifact["feature_names"]

        # 2. Convertimos el JSON entrante a DataFrame (1 sola fila)
        # Usamos model_dump() (Pydantic v2) en lugar del deprecado dict()
        df_input = pd.DataFrame([data.model_dump()])

        # 3. Aplicar Feature Engineering
        X_proc_array = pipeline.transform(df_input)
        
        # 4. Reconstruir DataFrame con los nombres correctos para el modelo
        X_proc_df = pd.DataFrame(X_proc_array, columns=feat_names)

        # 5. Predicción probabilística
        # Buscamos robustamente la columna que corresponde a la clase 0
        idx_clase_0 = list(model.classes_).index(0)
        proba_0 = float(model.predict_proba(X_proc_df)[0, idx_clase_0])

        # 6. Aplicar la regla de decisión calibrada
        prediccion = "moroso" if proba_0 >= threshold else "al_dia"

        return PredictionOutput(
            prediccion=prediccion,
            probabilidad_moroso=proba_0,
            umbral_utilizado=threshold
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))