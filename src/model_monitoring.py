"""
model_monitoring.py
===================

Dashboard interactivo con Streamlit para el monitoreo de Data Drift
en el modelo predictivo de riesgo crediticio.

Calcula el Population Stability Index (PSI) comparando la distribución
de los datos de entrenamiento (referencia) con nuevos lotes de datos
(producción) para detectar si la población ha cambiado significativamente.

Autor: Harrison
Versión: V1.4.0
"""

# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import streamlit as st
from pathlib import Path

# ============================================================
# CONSTANTES DE CONFIGURACIÓN
# ============================================================

# Resolución robusta de rutas hacia la raíz del proyecto
RAIZ: Path = Path(__file__).parent.parent
RUTA_XTRAIN: Path = RAIZ / "X_train_processed.csv"

# Variables clave a monitorear (salidas de tu feature engineering)
VARIABLES_MONITOREO = [
    "median__puntaje_datacredito", 
    "passthrough__edad_cliente",
    "passthrough__huella_consulta"
]

# ============================================================
# FUNCIONES DE CÁLCULO DE DRIFT
# ============================================================

def calculate_psi(expected: pd.Series, actual: pd.Series, buckets: int = 10) -> float:
    """
    Calcula el Population Stability Index (PSI) entre dos distribuciones.

    Parameters
    ----------
    expected : pd.Series
        Vector de datos de referencia (entrenamiento).
    actual : pd.Series
        Vector de datos actuales (producción).
    buckets : int, optional
        Cantidad de bins en los que se dividirá la distribución. Por defecto 10.

    Returns
    -------
    float
        Valor del PSI calculado.
    """
    # Definir los límites de los bins basados en la distribución esperada (entrenamiento)
    bins = np.histogram_bin_edges(expected, bins=buckets)
    
    # Calcular los porcentajes de observaciones en cada bin
    expected_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=bins)[0] / len(actual)
    
    # Reemplazar ceros por un valor muy pequeño para evitar errores en el logaritmo
    expected_pct = np.where(expected_pct == 0, 0.0001, expected_pct)
    actual_pct = np.where(actual_pct == 0, 0.0001, actual_pct)
    
    # Aplicar la fórmula matemática del PSI
    psi_value = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    
    return float(psi_value)

def interpretar_psi(psi_value: float) -> str:
    """
    Interpreta el valor del PSI según los estándares de la industria.

    Parameters
    ----------
    psi_value : float
        Valor del PSI calculado.

    Returns
    -------
    str
        Texto explicativo y color asociado.
    """
    if psi_value < 0.1:
        return "🟢 Sin cambios significativos (PSI < 0.1)"
    elif 0.1 <= psi_value < 0.2:
        return "🟡 Cambio moderado, vigilar (0.1 ≤ PSI < 0.2)"
    else:
        return "🔴 Cambio significativo, ALERTA DE DRIFT (PSI ≥ 0.2)"

# ============================================================
# INTERFAZ DE STREAMLIT (UI)
# ============================================================

def main():
    """Función principal que renderiza el dashboard de Streamlit."""
    
    st.set_page_config(page_title="Monitoreo MLOps", page_icon="📊", layout="wide")
    
    st.title("📊 Dashboard de Monitoreo: Data Drift (PSI)")
    st.markdown("---")
    
    # 1. Cargar datos de referencia (Training)
    try:
        df_ref = pd.read_csv(RUTA_XTRAIN)
        st.sidebar.success(f"Datos base cargados: {df_ref.shape[0]} filas")
    except Exception as e:
        st.error(f"Error al cargar datos de referencia. Asegúrate de tener {RUTA_XTRAIN.name} en la raíz.")
        return

    st.sidebar.markdown("### 📂 Cargar nuevos datos")
    st.sidebar.info("Sube un CSV procesado para comparar contra el entrenamiento.")
    
    # 2. Subida de archivo de producción
    archivo_subido = st.sidebar.file_uploader("Sube los datos de producción", type=["csv"])
    
    if archivo_subido is not None:
        df_prod = pd.read_csv(archivo_subido)
        st.success(f"Datos de producción cargados exitosamente: {df_prod.shape[0]} filas")
        
        st.markdown("### 📈 Análisis de Estabilidad Poblacional (PSI)")
        
        # Crear columnas para el layout visual
        cols = st.columns(len(VARIABLES_MONITOREO))
        
        for idx, col_name in enumerate(VARIABLES_MONITOREO):
            if col_name in df_prod.columns and col_name in df_ref.columns:
                # Calcular el PSI
                psi_val = calculate_psi(df_ref[col_name], df_prod[col_name])
                
                # Mostrar en una métrica
                with cols[idx]:
                    st.metric(label=f"Variable: {col_name.split('__')[-1]}", value=f"{psi_val:.4f}")
                    st.caption(interpretar_psi(psi_val))
            else:
                st.warning(f"La variable {col_name} no se encuentra en ambos datasets.")

# ============================================================
# EJECUCIÓN
# ============================================================
if __name__ == "__main__":
    main()