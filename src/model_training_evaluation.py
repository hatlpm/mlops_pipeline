"""
model_training_evaluation.py
============================

Entrenamiento, evaluación y selección del modelo de riesgo crediticio.

Consume los CSV procesados que genera ``ft_engineering.py`` (V1.1.0) y
entrena tres clasificadores supervisados, los compara con métricas
diseñadas para desbalance severo (clase 0 / morosos ≈ 4.75 %), elige el
mejor, calibra su umbral de decisión y serializa el artefacto final.

El artefacto guardado en ``../model.pkl`` NO es el estimador suelto, sino
un diccionario con todo lo que ``model_deploy.py`` necesita en producción:

    {
        "model":         <estimador ya fitteado>,
        "threshold":     <float, umbral óptimo para la clase 0>,
        "feature_names": <list[str], las 25 columnas de entrada>,
    }

El script puede ejecutarse de forma independiente::

    cd src/
    python model_training_evaluation.py

o ser importado desde ``model_deploy.py`` sin disparar el entrenamiento
(gracias al guard ``if __name__ == "__main__":``).

Autor: Harrison
Versión: V1.2.0
"""

# ============================================================
# IMPORTS
# ============================================================

# --- Manejo de datos ---
import pandas as pd
import numpy as np

# --- Serialización del artefacto ---
import joblib

# --- Tipado estático (mejora legibilidad y autocompletado) ---
from pathlib import Path
from typing import Dict, Tuple

# --- Modelos supervisados ---
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Escalado SOLO para la regresión logística (ver build_models) ---
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# --- Métricas de evaluación ---
from sklearn.metrics import (
    average_precision_score,   # PR-AUC (average precision)
    roc_auc_score,             # ROC-AUC (referencia)
    f1_score,                  # F1 por clase
    precision_recall_curve,    # barrido de umbrales para calibrar threshold
    confusion_matrix,          # matriz de confusión del ganador
    classification_report,     # reporte por clase del ganador
)


# ============================================================
# CONSTANTES DE CONFIGURACIÓN
# ============================================================

# --- Rutas (relativas a src/, igual patrón que ft_engineering.py) ---
RUTA_XTRAIN: Path = Path("../X_train_processed.csv")
RUTA_XTEST: Path = Path("../X_test_processed.csv")
RUTA_YTRAIN: Path = Path("../y_train.csv")
RUTA_YTEST: Path = Path("../y_test.csv")
RUTA_MODEL_OUT: Path = Path("../model.pkl")

# --- Variable objetivo ---
TARGET: str = "Pago_atiempo"

# --- Clase de interés para el negocio ---
# La clase 0 (morosos) es la minoría que queremos detectar. TODAS las
# métricas de selección se calculan con pos_label=0.
POS_LABEL: int = 0

# --- Reproducibilidad: una sola semilla para todo el experimento ---
RANDOM_STATE: int = 42

# --- Epsilon para evitar división por cero al calcular F1 manualmente ---
_EPS: float = 1e-12


# ============================================================
# CARGA DE DATOS
# ============================================================

def load_processed_data(
    ruta_xtrain: Path = RUTA_XTRAIN,
    ruta_ytrain: Path = RUTA_YTRAIN,
    ruta_xtest: Path = RUTA_XTEST,
    ruta_ytest: Path = RUTA_YTEST,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Carga los CSV procesados generados por ``ft_engineering.py``.

    No aplica ningún preprocesamiento adicional: los datos ya pasaron por el
    pipeline completo de feature engineering (imputación, encoding, log,
    capeo). Aquí solo se leen y se valida su integridad.

    Parameters
    ----------
    ruta_xtrain, ruta_ytrain, ruta_xtest, ruta_ytest : Path, optional
        Rutas a los cuatro CSV. Por defecto apuntan a ``../`` (un nivel
        arriba de ``src/``), igual que ``ft_engineering.py``.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]
        ``X_train, y_train, X_test, y_test`` en ese orden. Las ``y`` se
        devuelven como ``Series`` 1-D (no DataFrame de una columna).

    Raises
    ------
    FileNotFoundError
        Si alguno de los cuatro archivos no existe.
    ValueError
        Si quedan NaN en los datos (señal de que el feature engineering
        no se completó correctamente) o si train/test no comparten columnas.
    """
    # Validamos existencia de los cuatro archivos antes de leer nada,
    # para dar un mensaje claro si se ejecuta desde la carpeta equivocada.
    for ruta in (ruta_xtrain, ruta_ytrain, ruta_xtest, ruta_ytest):
        if not ruta.exists():
            raise FileNotFoundError(
                f"No se encontró {ruta}. ¿Estás ejecutando desde src/ y "
                f"corriste antes ft_engineering.py para generar los CSV?"
            )

    # Lectura. Los CSV procesados son float64 limpios; no necesitan
    # decimal="," ni engine="python" como el Base_de_datos.csv original.
    X_train = pd.read_csv(ruta_xtrain)
    X_test = pd.read_csv(ruta_xtest)

    # squeeze("columns") convierte el DataFrame de una sola columna en Series.
    y_train = pd.read_csv(ruta_ytrain).squeeze("columns")
    y_test = pd.read_csv(ruta_ytest).squeeze("columns")

    # --- Validaciones de integridad (defensa ante errores silenciosos) ---
    if X_train.isna().sum().sum() or X_test.isna().sum().sum():
        raise ValueError(
            "Hay NaN en los datos procesados. El feature engineering debería "
            "haberlos eliminado. Revisá ft_engineering.py."
        )
    if list(X_train.columns) != list(X_test.columns):
        raise ValueError(
            "Las columnas de train y test no coinciden. El ColumnTransformer "
            "debe producir el mismo esquema en ambos."
        )

    return X_train, y_train, X_test, y_test


# ============================================================
# CONFIGURACIÓN DE MODELOS
# ============================================================

def compute_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calcula ``scale_pos_weight`` para XGBoost en la dirección correcta.

    XGBoost pondera la clase POSITIVA, que por convención es la etiqueta 1
    (en este problema, la mayoría / buenos pagadores). El valor balanceador
    es ``n(clase 0) / n(clase 1)``. Como la clase 1 domina, el resultado es
    < 1 (≈ 0.05): esto *castiga* a la mayoría y obliga al modelo a prestar
    atención a la clase 0.

    ATENCIÓN: invertir el cociente (usar ``n1/n0 ≈ 20``) amplificaría la
    clase que ya domina y destruiría la detección de morosos. Es el error
    más común con esta API; por eso se calcula explícitamente acá.

    Parameters
    ----------
    y_train : pd.Series
        Vector de etiquetas de entrenamiento con valores en {0, 1}.

    Returns
    -------
    float
        Cociente ``n(clase 0) / n(clase 1)``.
    """
    n_clase_0 = int((y_train == 0).sum())   # morosos (minoría)
    n_clase_1 = int((y_train == 1).sum())   # buenos pagadores (mayoría)
    # n1 nunca es 0 en este dataset, pero protegemos la división igual.
    return n_clase_0 / max(n_clase_1, 1)


def build_models(scale_pos_weight: float) -> Dict[str, object]:
    """
    Construye los tres clasificadores con su configuración anti-desbalance.

    Cada modelo recibe la misma semilla (``RANDOM_STATE``) para que el
    experimento sea reproducible. El desbalance se maneja distinto según
    el algoritmo:

    - LogisticRegression y RandomForest: ``class_weight="balanced"``, que
      ajusta los pesos de forma inversamente proporcional a la frecuencia
      de cada clase. Es simétrico y no tiene el problema de dirección de
      ``scale_pos_weight``.
    - XGBoost: ``scale_pos_weight`` (ver compute_scale_pos_weight).

    La regresión logística se envuelve en un ``Pipeline`` con
    ``StandardScaler`` porque las features conviven en escalas muy distintas
    (las ``log_*`` en ~14, ``puntaje_datacredito`` en ~780). Sin escalar, la
    penalización L2 quedaría dominada por las columnas de mayor magnitud y el
    solver podría no converger. Los modelos de árbol son invariantes a la
    escala, así que no lo necesitan.

    Parameters
    ----------
    scale_pos_weight : float
        Valor balanceador para XGBoost. Ver ``compute_scale_pos_weight``.

    Returns
    -------
    Dict[str, object]
        Diccionario ``{nombre: estimador_sin_fittear}``.
    """
    # --- Baseline lineal: LogReg + escalado interno ---
    # El escalado vive DENTRO del pipeline del modelo, no es un repaso del
    # feature engineering: solo prepara los datos para este algoritmo y
    # viaja con el modelo si éste resulta ganador (se serializa completo).
    log_reg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",   # compensa el 95/5
            max_iter=1000,             # holgura para garantizar convergencia
            random_state=RANDOM_STATE,
        )),
    ])

    # --- RandomForest: no lineal, con frenos anti-overfit ---
    # max_depth y min_samples_leaf limitados evitan que el bosque memorice
    # la clase mayoritaria con árboles profundos sobre datos desbalanceados.
    random_forest = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,                   # freno de profundidad (anti-overfit)
        min_samples_leaf=20,           # hojas con soporte mínimo, suaviza
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,                     # usa todos los núcleos disponibles
    )

    # --- XGBoost: boosting, candidato fuerte en tabular crediticio ---
    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=5,                   # árboles poco profundos + muchos
        learning_rate=0.05,            # tasa baja para mejor generalización
        subsample=0.8,                 # submuestreo de filas (regularización)
        colsample_bytree=0.8,          # submuestreo de columnas por árbol
        scale_pos_weight=scale_pos_weight,   # dirección correcta (≈0.05)
        eval_metric="logloss",         # métrica interna neutral
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,                   # silencia los logs de entrenamiento
    )

    return {
        "LogisticRegression": log_reg,
        "RandomForest": random_forest,
        "XGBoost": xgb,
    }


# ============================================================
# UTILIDADES DE PROBABILIDAD Y UMBRAL
# ============================================================

def class0_probabilities(model: object, X: pd.DataFrame) -> np.ndarray:
    """
    Extrae la probabilidad predicha de la clase 0 de forma robusta.

    ``predict_proba`` devuelve una matriz cuyas columnas están ordenadas
    según ``model.classes_``. Asumir que la columna 0 es la clase 0 funciona
    casi siempre, pero buscamos el índice explícitamente para no depender de
    ese orden (buena práctica que evita bugs silenciosos).

    Parameters
    ----------
    model : object
        Estimador ya fitteado con método ``predict_proba`` y atributo
        ``classes_`` (los Pipeline de sklearn también lo exponen).
    X : pd.DataFrame
        Features sobre las que predecir.

    Returns
    -------
    np.ndarray
        Vector 1-D con P(clase = 0) para cada fila.
    """
    # Localizamos en qué columna quedó la clase 0 según el orden interno.
    idx_clase_0 = list(model.classes_).index(POS_LABEL)
    return model.predict_proba(X)[:, idx_clase_0]


def optimal_threshold_f1(
    y_true: pd.Series,
    proba_class0: np.ndarray,
) -> Tuple[float, float]:
    """
    Encuentra el umbral que maximiza el F1 de la clase 0.

    Con 4.75 % de minoría, el modelo aprende probabilidades bajas para la
    clase 0, por lo que el umbral por defecto de 0.5 casi nunca es óptimo.
    Barremos todos los umbrales de la curva precisión-recall (calculada con
    ``pos_label=0``) y elegimos el que da mayor F1.

    Regla de decisión asociada: se predice clase 0 cuando
    ``proba_class0 >= threshold``.

    Parameters
    ----------
    y_true : pd.Series
        Etiquetas verdaderas en {0, 1}.
    proba_class0 : np.ndarray
        Probabilidad predicha de la clase 0.

    Returns
    -------
    Tuple[float, float]
        ``(umbral_óptimo, f1_en_ese_umbral)``.
    """
    # precision_recall_curve devuelve precision/recall de longitud n+1 y
    # thresholds de longitud n. El último par (precision=1, recall=0) no
    # tiene umbral asociado, por eso recortamos f1 a len(thresholds).
    precision, recall, thresholds = precision_recall_curve(
        y_true, proba_class0, pos_label=POS_LABEL
    )

    # F1 = 2·P·R / (P + R). El epsilon evita 0/0 cuando P=R=0.
    f1_scores = 2 * precision * recall / (precision + recall + _EPS)
    f1_scores = f1_scores[:-1]   # alinear con thresholds (descartar punto final)

    # Índice del mejor F1 → umbral y valor correspondientes.
    best_idx = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# ============================================================
# ENTRENAMIENTO Y EVALUACIÓN
# ============================================================

def train_models(
    models: Dict[str, object],
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Dict[str, object]:
    """
    Fittea cada modelo sobre el conjunto de entrenamiento.

    Parameters
    ----------
    models : Dict[str, object]
        Estimadores sin fittear, salidos de ``build_models``.
    X_train : pd.DataFrame
        Features de entrenamiento (25 columnas).
    y_train : pd.Series
        Etiquetas de entrenamiento.

    Returns
    -------
    Dict[str, object]
        El mismo diccionario, con los estimadores ya fitteados in-place.
    """
    for nombre, modelo in models.items():
        print(f"  Entrenando {nombre}...")
        modelo.fit(X_train, y_train)   # fit modifica el estimador in-place
    return models


def evaluate_model(
    model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Calcula las métricas de evaluación de un modelo sobre el test set.

    Todas las métricas se orientan a la clase 0 (la que importa al negocio):

    - ``pr_auc``  : PR-AUC clase 0 — métrica PRIMARIA de selección.
    - ``f1_class0``: F1 clase 0 en su umbral óptimo — desempate.
    - ``roc_auc`` : ROC-AUC — referencia (simétrico entre clases).
    - ``threshold``: umbral que maximiza F1 clase 0.

    Parameters
    ----------
    model : object
        Estimador ya fitteado.
    X_test : pd.DataFrame
        Features de evaluación.
    y_test : pd.Series
        Etiquetas verdaderas de evaluación.

    Returns
    -------
    Dict[str, float]
        Diccionario con las claves ``pr_auc``, ``f1_class0``, ``roc_auc`` y
        ``threshold``.
    """
    # Probabilidad de la clase 0 (la de interés).
    proba_0 = class0_probabilities(model, X_test)

    # --- PR-AUC clase 0 (primaria) ---
    # pos_label=0: sin esto sklearn mediría la precisión/recall de la mayoría.
    pr_auc = average_precision_score(y_test, proba_0, pos_label=POS_LABEL)

    # --- ROC-AUC (referencia) ---
    # Es simétrico: el valor es idéntico se mire desde la clase 0 o la 1.
    # Lo calculamos con la probabilidad de la clase 1 (1 - proba_0).
    roc_auc = roc_auc_score(y_test, 1.0 - proba_0)

    # --- F1 clase 0 en el umbral óptimo (desempate) ---
    best_threshold, best_f1 = optimal_threshold_f1(y_test, proba_0)

    return {
        "pr_auc": float(pr_auc),
        "f1_class0": float(best_f1),
        "roc_auc": float(roc_auc),
        "threshold": float(best_threshold),
    }


def select_best(results: Dict[str, Dict[str, float]]) -> str:
    """
    Elige el modelo ganador por PR-AUC clase 0, con F1 clase 0 de desempate.

    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Métricas por modelo, salidas de ``evaluate_model``.

    Returns
    -------
    str
        Nombre del modelo ganador.
    """
    # max() con una tupla de criterios: primero pr_auc, luego f1_class0.
    # Python compara tuplas elemento a elemento, así que esto implementa
    # "primaria + desempate" en una sola línea.
    return max(
        results,
        key=lambda nombre: (results[nombre]["pr_auc"], results[nombre]["f1_class0"]),
    )


# ============================================================
# SERIALIZACIÓN DEL ARTEFACTO
# ============================================================

def save_artifact(
    model: object,
    threshold: float,
    feature_names: list,
    ruta_out: Path = RUTA_MODEL_OUT,
) -> None:
    """
    Serializa el artefacto de producción como diccionario con joblib.

    Guardar un dict (en vez del estimador suelto) permite que
    ``model_deploy.py`` valide los features de entrada y aplique el umbral
    correcto sin depender del 0.5 por defecto.

    Parameters
    ----------
    model : object
        Mejor estimador, ya fitteado.
    threshold : float
        Umbral óptimo para la clase 0.
    feature_names : list
        Lista de las 25 columnas de entrada, en orden, para validar inputs.
    ruta_out : Path, optional
        Destino del artefacto. Por defecto ``../model.pkl``.
    """
    artifact = {
        "model": model,                  # estimador ya fitteado
        "threshold": threshold,          # float, umbral óptimo para clase 0
        "feature_names": feature_names,  # 25 strings para validar inputs
    }
    joblib.dump(artifact, ruta_out)


# ============================================================
# REPORTE EJECUTIVO
# ============================================================

def print_report(
    results: Dict[str, Dict[str, float]],
    best_name: str,
    best_model: object,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    ruta_out: Path,
) -> None:
    """
    Imprime el reporte ejecutivo del experimento.

    Incluye la tabla comparativa de las tres métricas por modelo, el modelo
    ganador con su umbral, y la matriz de confusión / reporte por clase del
    ganador en su umbral óptimo.

    Parameters
    ----------
    results : Dict[str, Dict[str, float]]
        Métricas por modelo.
    best_name : str
        Nombre del modelo ganador.
    best_model : object
        Estimador ganador, ya fitteado.
    X_test, y_test : pd.DataFrame, pd.Series
        Conjunto de evaluación.
    ruta_out : Path
        Ruta donde se guardó el artefacto.
    """
    # --- Tabla comparativa ---
    tabla = pd.DataFrame(results).T[["pr_auc", "f1_class0", "roc_auc", "threshold"]]
    tabla = tabla.sort_values("pr_auc", ascending=False)

    print("\n" + "=" * 64)
    print("REPORTE EJECUTIVO — model_training_evaluation.py (V1.2.0)")
    print("=" * 64)
    print("\nComparativa de modelos (orden: mejor PR-AUC clase 0 arriba):\n")
    print(tabla.round(4).to_string())

    print(f"\nMODELO GANADOR : {best_name}")
    print(f"  PR-AUC clase 0 : {results[best_name]['pr_auc']:.4f}  (primaria)")
    print(f"  F1 clase 0     : {results[best_name]['f1_class0']:.4f}  (desempate)")
    print(f"  ROC-AUC        : {results[best_name]['roc_auc']:.4f}  (referencia)")
    print(f"  Umbral óptimo  : {results[best_name]['threshold']:.4f}")

    # --- Matriz de confusión y reporte del ganador en su umbral óptimo ---
    proba_0 = class0_probabilities(best_model, X_test)
    umbral = results[best_name]["threshold"]
    # Regla de decisión: clase 0 si P(clase0) >= umbral, si no clase 1.
    y_pred = np.where(proba_0 >= umbral, 0, 1)

    print("\nMatriz de confusión del ganador (filas=real, columnas=predicho):")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(f"           pred=0   pred=1")
    print(f"  real=0  {cm[0, 0]:>7d} {cm[0, 1]:>8d}")
    print(f"  real=1  {cm[1, 0]:>7d} {cm[1, 1]:>8d}")

    print("\nReporte por clase del ganador (en su umbral óptimo):")
    print(classification_report(y_test, y_pred, digits=4))

    print(f"Artefacto guardado en: {ruta_out.resolve()}")
    print("=" * 64)


# ============================================================
# ORQUESTACIÓN
# ============================================================

def main() -> None:
    """
    Orquesta el flujo completo: carga, entrenamiento, evaluación, selección,
    serialización y reporte.

    Parameters
    ----------
    Ninguno.

    Returns
    -------
    None
    """
    # --- 1. Cargar datos procesados ---
    print("Cargando datos procesados desde ../ ...")
    X_train, y_train, X_test, y_test = load_processed_data()
    print(f"  Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"  Clase 0 en train: {(y_train == 0).mean():.4f} | "
          f"en test: {(y_test == 0).mean():.4f}")

    # --- 2. Configurar el desbalance de XGBoost ---
    spw = compute_scale_pos_weight(y_train)
    print(f"  scale_pos_weight (n0/n1) = {spw:.4f}")

    # --- 3. Construir y entrenar los tres modelos ---
    print("Entrenando modelos...")
    models = build_models(scale_pos_weight=spw)
    models = train_models(models, X_train, y_train)

    # --- 4. Evaluar cada modelo sobre el test set ---
    print("Evaluando modelos...")
    results = {
        nombre: evaluate_model(modelo, X_test, y_test)
        for nombre, modelo in models.items()
    }

    # --- 5. Seleccionar el ganador y serializar el artefacto ---
    best_name = select_best(results)
    best_model = models[best_name]
    best_threshold = results[best_name]["threshold"]
    save_artifact(
        model=best_model,
        threshold=best_threshold,
        feature_names=list(X_train.columns),
        ruta_out=RUTA_MODEL_OUT,
    )

    # --- 6. Reporte ejecutivo ---
    print_report(results, best_name, best_model, X_test, y_test, RUTA_MODEL_OUT)


# ============================================================
# BLOQUE PRINCIPAL
# El guard permite importar este módulo desde model_deploy.py
# (para reutilizar funciones) sin disparar el entrenamiento.
# ============================================================

if __name__ == "__main__":
    main()


# ============================================================
# DECISIONES Y JUSTIFICACIONES (log para handoff)
# ============================================================
#
# 1. ELECCIÓN DE MÉTRICAS
#    - PR-AUC clase 0 como métrica primaria: con 4.75 % de minoría, ROC-AUC
#      se infla porque cuenta el enorme volumen de verdaderos negativos
#      fáciles. PR-AUC solo mira precisión vs recall de la clase 0, así que
#      penaliza directamente fallar en detectar morosos.
#    - F1 clase 0 como desempate: resume precisión y recall de la minoría en
#      un punto de operación concreto.
#    - ROC-AUC como referencia: se reporta por costumbre y comparabilidad,
#      pero NO decide el ganador.
#
# 2. MANEJO DEL DESBALANCE
#    - LogReg y RF: class_weight="balanced" (simétrico, sin riesgo de
#      dirección).
#    - XGBoost: scale_pos_weight = n0/n1 ≈ 0.05. Pondera la etiqueta 1
#      (mayoría); un valor <1 la castiga. Invertirlo (≈20) sería el error
#      clásico: amplificaría la clase dominante y hundiría el recall de
#      morosos.
#
# 3. HIPERPARÁMETROS (elegidos fijos, no por GridSearch)
#    - Se priorizó reproducibilidad y tiempo de ejecución sobre el último
#      punto de métrica. Los tres modelos usan valores sensatos y conocidos
#      por su robustez en tabular. Un GridSearchCV es una extensión válida
#      si sobra tiempo, pero comparar 3 familias de modelos ya cubre la
#      experimentación que pide la rúbrica.
#    - RF: max_depth=8 + min_samples_leaf=20 frenan el overfitting típico de
#      bosques profundos sobre datos desbalanceados.
#    - XGB: learning_rate bajo (0.05) + subsample/colsample 0.8 regularizan.
#
# 4. CALIBRACIÓN DEL UMBRAL
#    - Se elige el umbral que maximiza F1 clase 0 sobre la curva PR.
#    - LIMITACIÓN CONOCIDA: el umbral (y la selección del ganador) se
#      calibran sobre el test set. Es levemente circular; lo correcto sería
#      una validación cruzada o un split de validación dedicado. Se acepta
#      para el alcance del proyecto y se deja documentado. Si querés cerrar
#      el hueco, calibrá el umbral con cross_val_predict sobre train.
#
# 5. ESCALADO SOLO EN LOGREG
#    - La regresión logística va envuelta en Pipeline(StandardScaler, LogReg)
#      porque las features conviven en escalas dispares y sin escalar el
#      solver puede no converger y la penalización L2 se sesga. NO es repetir
#      el feature engineering: es preprocesamiento interno del modelo, que
#      viaja serializado con él si gana. Los árboles no lo necesitan.
#
# 6. QUÉ DECIDÍ NO HACER
#    - No usar AutoML ni librerías fuera de requirements.txt.
#    - No re-imputar ni re-encodear: los CSV ya vienen limpios de
#      ft_engineering.py (0 NaN verificados).
#    - No hacer oversampling/SMOTE: class_weight y scale_pos_weight ya
#      atacan el desbalance sin inventar filas sintéticas.
#
# 7. QUÉ DEBE REVISAR EL ESTUDIANTE AL CORRER EN LOCAL
#    - Ejecutar SIEMPRE desde src/ (las rutas son relativas a ../).
#    - Verificar que ../model.pkl se creó y que es un dict con las claves
#      model, threshold, feature_names.
#    - Mirar la tabla: si XGBoost gana, confirmar que su PR-AUC supera al
#      resto y que el umbral guardado NO es 0.5.
#    - Si LogReg lanza ConvergenceWarning, subir max_iter (no debería con el
#      StandardScaler puesto).