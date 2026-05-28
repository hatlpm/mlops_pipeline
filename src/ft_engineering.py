"""
ft_engineering.py
=================

Pipeline de ingeniería de características para el modelo predictivo de
riesgo crediticio. Implementa todas las decisiones tomadas durante el EDA
(notebook comprension_eda.ipynb, V1.0.3).

El script puede ejecutarse de forma independiente para generar los
artefactos de entrenamiento, o ser importado desde otros módulos
(model_training_evaluation.py, model_deploy.py) para reutilizar el
pipeline.

Artefactos generados al ejecutar como script:
    - ft_pipeline.pkl: pipeline serializado y fitteado sobre X_train.
    - X_train_processed.csv, X_test_processed.csv: features procesadas.
    - y_train.csv, y_test.csv: variable objetivo separada.

Autor: Harrison
Versión: V1.1.0
"""

# ============================================================
# IMPORTS
# ============================================================

# --- Manejo de datos ---
import pandas as pd
import numpy as np

# --- Serialización del pipeline ---
import joblib

# --- Tipado estático (mejora legibilidad y autocompletado) ---
from pathlib import Path
from typing import Tuple

# --- Sklearn: núcleo del pipeline ---
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split

# --- Feature-engine: transformaciones tabulares listas para usar ---
from feature_engine.outliers import Winsorizer



# ============================================================
# CONSTANTES DE CONFIGURACIÓN
# ============================================================

# --- Rutas (relativas a la raíz del proyecto, no a src/) ---
RUTA_DATASET: Path = Path("../Base_de_datos.csv")
RUTA_PIPELINE_OUT: Path = Path("../ft_pipeline.pkl")
RUTA_XTRAIN_OUT: Path = Path("../X_train_processed.csv")
RUTA_XTEST_OUT: Path = Path("../X_test_processed.csv")
RUTA_YTRAIN_OUT: Path = Path("../y_train.csv")
RUTA_YTEST_OUT: Path = Path("../y_test.csv")

# --- Variable objetivo ---
TARGET: str = "Pago_atiempo"

# --- Parámetros del split ---
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# --- Valores especiales detectados en el EDA ---
PUNTAJE_PLACEHOLDER: float = 95.227787   # valor que aparece en el 87.4% de "puntaje"
EDAD_MAX: int = 85                       # tope realista para capear outliers de edad
TIPOS_CREDITO_RAROS: list = [6, 7, 68]   # códigos con frecuencia mínima → agrupar en "otro"

# --- Categorías válidas conocidas de tendencia_ingresos ---
# Cualquier valor fuera de esta lista (ej: números contaminados) se trata como NaN
TENDENCIAS_VALIDAS: list = ["Creciente", "Estable", "Decreciente"]

# --- Columnas que se eliminan directamente (no aportan o son redundantes) ---
COLS_DROP_DIRECTO: list = [
    "fecha_prestamo",        # solo 1 año de datos, no extraemos features temporales
    "saldo_mora_codeudor",   # 100% ceros + 5.5% nulos, cero varianza
    "cuota_pactada",         # redundante con capital_prestado/plazo_meses (r=0.995)
    "puntaje",               # 87.4% placeholder, sin señal real
]

# --- Columnas numéricas según su tratamiento ---
COLS_LOG_CAP: list = [          # capeo p99 + log(x+1)
    "salario_cliente",
    "total_otros_prestamos",
    "cant_creditosvigentes",   # conteo con outlier extremo (max=242)
]
COLS_LOG: list = [              # solo log(x+1), imputar con mediana si hay nulos
    "capital_prestado",
    "promedio_ingresos_datacredito",
]
COLS_ZERO_LOG: list = [         # imputar nulos con 0 + log(x+1) (créditos nuevos)
    "saldo_total",
    "saldo_principal",
    "saldo_mora",
]
COLS_MEDIAN_ONLY: list = [      # solo imputar mediana, sin log
    "puntaje_datacredito",
    "tasa_implicita",
]
COLS_PASSTHROUGH: list = [      # pasan sin transformar (numéricas ya limpias)
    "edad_cliente",
    "plazo_meses",
    "huella_consulta",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "tiene_datacredito",     # feature derivada — debe sobrevivir al drop
    "tiene_puntaje_real",    # feature derivada — debe sobrevivir al drop
]

# --- Columnas categóricas ---
COL_TIPO_LABORAL: str = "tipo_laboral"          # binaria: Empleado / Independiente
COL_TENDENCIA: str = "tendencia_ingresos"       # multiclase con contaminación → OneHot
COL_TIPO_CREDITO: str = "tipo_credito"          # multiclase, agrupar raros → OneHot

# ============================================================
# TRANSFORMADORES CUSTOM
# ============================================================


class DerivedFeaturesCreator(BaseEstimator, TransformerMixin):
    """
    Crea features derivadas antes de que el pipeline elimine columnas fuente.

    Debe ejecutarse PRIMERO en el pipeline porque usa columnas que serán
    eliminadas en pasos posteriores (cuota_pactada, puntaje).

    Parameters
    ----------
    Ninguno. No aprende parámetros del dataset.

    Returns
    -------
    pd.DataFrame
        DataFrame original más las columnas nuevas:
        - tasa_implicita
        - tiene_datacredito
        - tiene_puntaje_real
    """

    def fit(self, X: pd.DataFrame, y=None) -> "DerivedFeaturesCreator":
        """No aprende parámetros; retorna self para cumplir interfaz sklearn."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame
            Dataset con columnas originales intactas.

        Returns
        -------
        pd.DataFrame
            Dataset con tres columnas adicionales.
        """
        # Trabajamos sobre una copia para no mutar el DataFrame original
        X = X.copy()

        # --- Feature 1: tasa implícita del crédito ---
        # Aproxima la tasa de interés efectiva del préstamo.
        # Usamos np.where para evitar división por cero si capital_prestado es 0.
        X["tasa_implicita"] = np.where(
            X["capital_prestado"] > 0,
            (X["cuota_pactada"] * X["plazo_meses"]) / X["capital_prestado"],
            np.nan,   # si capital es 0, la tasa no tiene sentido
        )

        # --- Feature 2: flag de si el cliente tiene registro en DataCrédito ---
        # promedio_ingresos_datacredito es NaN cuando el cliente no tiene historial.
        X["tiene_datacredito"] = (
            X["promedio_ingresos_datacredito"].notna().astype(int)
        )

        # --- Feature 3: flag de si el puntaje es real o placeholder ---
        # El valor 95.227787 fue detectado en el EDA como placeholder del sistema.
        # 0 = placeholder (sin puntaje real), 1 = puntaje real
        X["tiene_puntaje_real"] = (
            (X["puntaje"] != PUNTAJE_PLACEHOLDER).astype(int)
        )

        return X
    
    def get_feature_names_out(self, input_features=None) -> list:
        """
        Devuelve los nombres de columnas de salida: las de entrada más las
        tres features derivadas creadas en transform.

        Parameters
        ----------
        input_features : array-like, optional
            Nombres de columnas de entrada provistos por sklearn.

        Returns
        -------
        list
            Nombres de columnas de salida.
        """
        # input_features llega como las columnas que entraron a este paso.
        # Le sumamos las tres columnas nuevas que creamos en transform.
        nuevas = ["tasa_implicita", "tiene_datacredito", "tiene_puntaje_real"]
        return list(input_features) + nuevas


class DataCleaner(BaseEstimator, TransformerMixin):
    """
    Limpia valores imposibles y contaminaciones detectadas en el EDA.

    Tratamientos aplicados:
    - tendencia_ingresos: convierte valores numéricos (contaminación) a NaN.
    - puntaje_datacredito: convierte valores negativos a NaN (imposibles).
    - edad_cliente: capea en EDAD_MAX para eliminar outliers extremos.

    Parameters
    ----------
    Ninguno. No aprende parámetros del dataset.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "DataCleaner":
        """No aprende parámetros; retorna self para cumplir interfaz sklearn."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame
            Dataset con columnas originales o con features derivadas ya creadas.

        Returns
        -------
        pd.DataFrame
            Dataset con valores imposibles corregidos.
        """
        X = X.copy()

        # --- Limpiar contaminación en tendencia_ingresos ---
        # Los 58 valores numéricos detectados en EDA no son categorías válidas.
        # pd.to_numeric intenta convertir; los que no son números quedan como NaN.
        # Luego mapeamos: si el resultado no es NaN, el original era un número → NaN.
        es_numero = pd.to_numeric(X[COL_TENDENCIA], errors="coerce").notna()
        X.loc[es_numero, COL_TENDENCIA] = np.nan   # reemplazar contaminados con NaN

        # --- Limpiar puntaje_datacredito negativo ---
        # Un puntaje de crédito negativo es imposible en cualquier sistema.
        X.loc[X["puntaje_datacredito"] < 0, "puntaje_datacredito"] = np.nan

        # --- Capear edad_cliente en EDAD_MAX ---
        # Edades mayores a 85 son probablemente errores de ingreso de datos.
        X["edad_cliente"] = X["edad_cliente"].clip(upper=EDAD_MAX)

        return X
    
    def get_feature_names_out(self, input_features=None) -> list:
        """
        Devuelve los nombres de columnas de salida. DataCleaner no agrega
        ni elimina columnas, solo modifica valores, por lo que la salida
        es idéntica a la entrada.

        Parameters
        ----------
        input_features : array-like, optional
            Nombres de columnas de entrada provistos por sklearn.

        Returns
        -------
        list
            Nombres de columnas de salida (iguales a la entrada).
        """
        return list(input_features)


class TipoCreditoGrouper(BaseEstimator, TransformerMixin):
    """
    Agrupa los códigos de tipo_credito con frecuencia mínima en la categoría 'otro'.

    Los códigos en TIPOS_CREDITO_RAROS (6, 7, 68) representan una fracción
    pequeña del dataset. Dejarlos como categorías separadas genera columnas
    OneHot con muy poca señal. Agruparlos reduce ruido.

    Parameters
    ----------
    Ninguno. Los códigos a agrupar están definidos en la constante
    TIPOS_CREDITO_RAROS.
    """

    def fit(self, X: pd.DataFrame, y=None) -> "TipoCreditoGrouper":
        """No aprende parámetros; retorna self para cumplir interfaz sklearn."""
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Parameters
        ----------
        X : pd.DataFrame
            Debe contener la columna tipo_credito.

        Returns
        -------
        pd.DataFrame
            Dataset con tipo_credito agrupado.
        """
        X = X.copy()

        # Convertir a string para que el OneHotEncoder posterior lo trate
        # como categórica, independientemente de si los códigos son int o float
        X[COL_TIPO_CREDITO] = X[COL_TIPO_CREDITO].astype(str)

        # Reemplazar códigos raros por la etiqueta "otro"
        X[COL_TIPO_CREDITO] = X[COL_TIPO_CREDITO].where(
            ~X[COL_TIPO_CREDITO].isin([str(c) for c in TIPOS_CREDITO_RAROS]),
            other="otro",
        )

        return X
    
    def get_feature_names_out(self, input_features=None) -> list:
        """
        Devuelve los nombres de columnas de salida. TipoCreditoGrouper solo
        modifica los valores de tipo_credito, no agrega ni elimina columnas.

        Parameters
        ----------
        input_features : array-like, optional
            Nombres de columnas de entrada provistos por sklearn.

        Returns
        -------
        list
            Nombres de columnas de salida (iguales a la entrada).
        """
        return list(input_features)
    
# ============================================================
# CONSTRUCCIÓN DEL PIPELINE
# ============================================================

def build_pipeline() -> Pipeline:
    """
    Construye y retorna el pipeline completo de preprocesamiento.

    El pipeline NO está fitteado al retornarse. Debe llamarse
    fit() o fit_transform() sobre X_train antes de usar transform().

    El orden de los pasos es obligatorio:
    1. DerivedFeaturesCreator: usa columnas que serán eliminadas después.
    2. DataCleaner: limpia imposibles antes de imputar.
    3. ColumnTransformer: imputa, escala, encodea y dropea en paralelo.

    Parameters
    ----------
    Ninguno.

    Returns
    -------
    Pipeline
        Pipeline de sklearn listo para fittear.
    """

    # ----------------------------------------------------------
    # PASO 3A: sub-pipelines por grupo de columnas
    # Cada sub-pipeline define qué transformaciones recibe cada grupo
    # ----------------------------------------------------------

    # Grupo 1: capeo de outliers (IQR) + log(x+1)
    # Winsorizer aprende los límites del train → transforma train y test con ellos
    # Usamos np.log1p (log(x+1)) en vez de LogTransformer para tolerar ceros
    # (ej: total_otros_prestamos o cant_creditosvigentes pueden ser 0)
    pipe_log_cap = Pipeline([
        ("capper", Winsorizer(
            capping_method="iqr",   # usa IQR para calcular los límites
            tail="right",           # solo capea la cola derecha (outliers altos)
            fold=3,                 # límite = Q3 + 3*IQR, más conservador que el default
        )),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ])

    # Grupo 2: solo log(x+1), con imputación de mediana si hay nulos
    pipe_log = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),  # log(x+1), tolera x=0
    ])

    # Grupo 3: imputar con 0 + log(x+1)
    # saldo_mora, saldo_total, saldo_principal: nulo = crédito nuevo = saldo 0
    pipe_zero_log = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ])

    # Grupo 4: solo imputar con mediana, sin transformar escala
    pipe_median = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    # Grupo 5: tipo_laboral → binario (Empleado=0, Independiente=1)
    pipe_tipo_laboral = Pipeline([
        ("encoder", OrdinalEncoder(
            categories=[["Empleado", "Independiente"]],
            handle_unknown="use_encoded_value",
            unknown_value=-1,  # si llega un valor nuevo en producción, no explota
        )),
    ])

    # Grupo 6: tendencia_ingresos → imputar "Sin_info" + OneHot
    pipe_tendencia = Pipeline([
        ("imputer", SimpleImputer(
            strategy="constant",
            fill_value="Sin_info",  # NaN y contaminados ya convertidos a NaN en DataCleaner
        )),
        ("onehot", OneHotEncoder(
            sparse_output=False,      # devuelve array denso, no matriz dispersa
            handle_unknown="ignore",  # categoría nueva en producción → columna de ceros
            drop="first",             # elimina una columna para evitar multicolinealidad
        )),
    ])

    # Grupo 7: tipo_credito → agrupar raros + OneHot
    pipe_tipo_credito = Pipeline([
        ("grouper", TipoCreditoGrouper()),
        ("onehot", OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop="first",
        )),
    ])

    # ----------------------------------------------------------
    # PASO 3B: ColumnTransformer — aplica cada sub-pipeline a sus columnas
    # remainder="drop" elimina todas las columnas no listadas explícitamente
    # (fecha_prestamo, cuota_pactada, puntaje, saldo_mora_codeudor mueren acá)
    # ----------------------------------------------------------

    preprocessor = ColumnTransformer(
        transformers=[
            ("log_cap",      pipe_log_cap,      COLS_LOG_CAP),
            ("log",          pipe_log,          COLS_LOG),
            ("zero_log",     pipe_zero_log,     COLS_ZERO_LOG),
            ("median",       pipe_median,       COLS_MEDIAN_ONLY),
            ("passthrough",  "passthrough",     COLS_PASSTHROUGH),
            ("tipo_laboral", pipe_tipo_laboral, [COL_TIPO_LABORAL]),
            ("tendencia",    pipe_tendencia,    [COL_TENDENCIA]),
            ("tipo_credito", pipe_tipo_credito, [COL_TIPO_CREDITO]),
        ],
        remainder="drop",   # todo lo que no está listado arriba se elimina
    )

    # ----------------------------------------------------------
    # PASO 3C: Pipeline principal — los 3 pasos en orden obligatorio
    # ----------------------------------------------------------

    pipeline = Pipeline([
        ("derived",       DerivedFeaturesCreator()),  # paso 1: crear features nuevas
        ("cleaner",       DataCleaner()),              # paso 2: limpiar imposibles
        ("preprocessor",  preprocessor),              # paso 3: todo lo demás
    ])

    return pipeline

# ============================================================
# FUNCIONES PÚBLICAS (importables desde otros módulos)
# ============================================================

def load_data(path: Path = RUTA_DATASET) -> pd.DataFrame:
    """
    Carga el dataset de créditos desde un archivo CSV.

    Parameters
    ----------
    path : Path, optional
        Ruta al archivo CSV. Por defecto RUTA_DATASET (../Base_de_datos.csv).

    Returns
    -------
    pd.DataFrame
        Dataset crudo, sin transformar.

    Raises
    ------
    FileNotFoundError
        Si el archivo no existe en la ruta indicada.
    """
    # Validamos existencia antes de leer para dar un error claro
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontró el dataset en {path}. "
            f"Verificá que ejecutás el script desde la carpeta src/."
        )

    # decimal="," porque puntaje viene con coma decimal (ej: 95,227787)
    # engine="python" + dtype_backend="numpy_nullable" evita que pyarrow
    # interprete columnas numéricas como texto al usar decimal=","
    df = pd.read_csv(path, decimal=",", engine="python")

    # Forzar conversión a numérico en las columnas que deben serlo.
    # errors="coerce": lo que no se pueda convertir queda como NaN
    # (útil porque tendencia_ingresos tiene contaminación que ya manejamos después)
    columnas_numericas = [
        "tipo_credito", "capital_prestado", "plazo_meses", "edad_cliente",
        "salario_cliente", "total_otros_prestamos", "cuota_pactada",
        "puntaje", "puntaje_datacredito", "cant_creditosvigentes",
        "huella_consulta", "saldo_mora", "saldo_total", "saldo_principal",
        "saldo_mora_codeudor", "creditos_sectorFinanciero",
        "creditos_sectorCooperativo", "creditos_sectorReal",
        "promedio_ingresos_datacredito", "Pago_atiempo",
    ]
    for col in columnas_numericas:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def prepare_train_test_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Separa features y target, y divide en train/test de forma estratificada.

    La estratificación (stratify=y) preserva el desbalance 95/5 de la
    variable objetivo tanto en train como en test. Sin esto, el split
    aleatorio podría dejar muy pocos morosos en el test.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset crudo con la columna objetivo incluida.
    test_size : float, optional
        Proporción del dataset asignada a test. Por defecto 0.2.
    random_state : int, optional
        Semilla para reproducibilidad. Por defecto 42.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test en ese orden.
    """
    # Separamos features (X) de la variable objetivo (y)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # stratify=y garantiza misma proporción de clases en train y test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,   # crítico por el desbalance 95/5
    )

    return X_train, X_test, y_train, y_test


# ============================================================
# BLOQUE PRINCIPAL (se ejecuta solo con: python ft_engineering.py)
# ============================================================

if __name__ == "__main__":

    # --- 1. Cargar datos ---
    print("Cargando dataset...")
    df = load_data()
    print(f"  Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")

    # --- 2. Split estratificado ---
    print("Dividiendo en train/test (estratificado)...")
    X_train, X_test, y_train, y_test = prepare_train_test_split(df)
    print(f"  Train: {X_train.shape[0]} filas | Test: {X_test.shape[0]} filas")
    print(f"  Proporción clase 0 en train: {(y_train == 0).mean():.3f}")
    print(f"  Proporción clase 0 en test:  {(y_test == 0).mean():.3f}")

    # --- 3. Construir pipeline ---
    print("Construyendo pipeline...")
    pipeline = build_pipeline()

    # --- 4. Fit SOLO en train + transform en train y test ---
    # fit_transform en train: aprende parámetros Y transforma
    # transform en test: usa los parámetros aprendidos del train (sin leakage)
    print("Fitteando pipeline en train y transformando...")
    X_train_proc = pipeline.fit_transform(X_train)
    X_test_proc = pipeline.transform(X_test)
    print(f"  Shape procesado train: {X_train_proc.shape}")
    print(f"  Shape procesado test:  {X_test_proc.shape}")

    # --- 5. Recuperar nombres de columnas de salida ---
    # El ColumnTransformer genera nombres automáticos; los recuperamos
    # para que los CSVs procesados sean legibles
    nombres_columnas = pipeline.named_steps["preprocessor"].get_feature_names_out()

    # Convertimos los arrays numpy de salida a DataFrames con nombres
    X_train_proc_df = pd.DataFrame(X_train_proc, columns=nombres_columnas)
    X_test_proc_df = pd.DataFrame(X_test_proc, columns=nombres_columnas)

    # --- 6. Guardar artefactos ---
    print("Guardando artefactos...")

    # Pipeline fitteado (lo consume la API en Fase 5)
    joblib.dump(pipeline, RUTA_PIPELINE_OUT)
    print(f"  Pipeline guardado en {RUTA_PIPELINE_OUT}")

    # Datasets procesados (los consume el modelado en Fase 4)
    X_train_proc_df.to_csv(RUTA_XTRAIN_OUT, index=False)
    X_test_proc_df.to_csv(RUTA_XTEST_OUT, index=False)
    y_train.to_csv(RUTA_YTRAIN_OUT, index=False)
    y_test.to_csv(RUTA_YTEST_OUT, index=False)
    print(f"  Datasets procesados guardados en la raíz del proyecto")

    # --- 7. Resumen ejecutivo ---
    print("\n" + "=" * 50)
    print("FEATURE ENGINEERING COMPLETADO")
    print("=" * 50)
    print(f"Columnas finales: {len(nombres_columnas)}")
    print(f"NaN restantes en train: {X_train_proc_df.isna().sum().sum()}")
    print(f"NaN restantes en test:  {X_test_proc_df.isna().sum().sum()}")
    print("=" * 50)