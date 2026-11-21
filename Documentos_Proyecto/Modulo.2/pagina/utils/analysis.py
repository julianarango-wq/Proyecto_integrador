"""Carga de datos, entrenamiento de modelos y métricas del dashboard.

Toda la lógica pesada vive aquí y se cachea para que la interfaz responda
inmediatamente. Los modelos se implementan con scikit-learn (en lugar de
TensorFlow) para que la aplicación se despliegue sin problemas en Streamlit
Cloud: la metodología y las métricas se calculan en tiempo de ejecución.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO_DATOS = RAIZ / "data" / "inventario_carga_.xlsx"

COLUMNAS_NUMERICAS = [
    "Potencia_W",
    "Horas_Uso_Dia",
    "Consumo_kWh",
    "H_activa",
    "Consumo_Diario_Total",
]
CARACTERISTICAS_REGRESION = [
    "Horas_Uso_Dia",
    "Potencia_W",
    "Consumo_kWh",
    "Consumo_Teorico",
]
LONGITUD_SECUENCIA = 5


@st.cache_data(show_spinner="Cargando dataset...")
def cargar_datos() -> pd.DataFrame:
    """Lee el Excel, deriva variables y calcula los clústeres K-Means."""
    df = pd.read_excel(ARCHIVO_DATOS)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df["Mes"] = pd.to_datetime(df["Mes"])
    df["Consumo_Teorico"] = (df["Potencia_W"] * df["Horas_Uso_Dia"]) / 1000.0

    mediana = df["Consumo_Diario_Total"].median()
    df["Alto_Consumo"] = (df["Consumo_Diario_Total"] > mediana).astype(int)

    escalador = StandardScaler()
    x_escalado = escalador.fit_transform(df[COLUMNAS_NUMERICAS])
    df["Cluster"] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(
        x_escalado
    )
    return df


@st.cache_data(show_spinner="Preparando serie diaria...")
def serie_diaria(df: pd.DataFrame) -> pd.Series:
    """Un único valor de consumo total por fecha, ordenado cronológicamente."""
    return (
        df.groupby("Fecha")["Consumo_Diario_Total"]
        .first()
        .sort_index()
        .rename("Consumo diario (kWh)")
    )


@st.cache_data(show_spinner="Calculando codo de K-Means...")
def curva_codo(df: pd.DataFrame) -> pd.DataFrame:
    """Inercia del clúster para k = 2..8."""
    x = StandardScaler().fit_transform(df[COLUMNAS_NUMERICAS])
    filas = []
    for k in range(2, 9):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10).fit(x)
        filas.append({"Clústeres": k, "Inercia": modelo.inertia_})
    return pd.DataFrame(filas)


def _crear_secuencias(datos: np.ndarray, longitud: int) -> tuple[np.ndarray, np.ndarray]:
    x, y = [], []
    for i in range(len(datos) - longitud):
        x.append(datos[i : i + longitud].ravel())
        y.append(datos[i + longitud])
    return np.asarray(x), np.asarray(y).ravel()


@st.cache_resource(show_spinner="Entrenando modelos predictivos...")
def entrenar_modelos(df: pd.DataFrame, serie: pd.Series) -> dict[str, Any]:
    """Entrena regresión, clasificación y serie temporal; devuelve métricas y modelos."""
    # ---------------------------------------------------------------- regresión
    x_reg = df[CARACTERISTICAS_REGRESION + ["Electrodomestico"]].copy()
    y_reg = df["Consumo_Diario_Total"].to_numpy()

    preprocesador = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), CARACTERISTICAS_REGRESION),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Electrodomestico"]),
        ]
    )
    x_proc = preprocesador.fit_transform(x_reg)
    if hasattr(x_proc, "toarray"):
        x_proc = x_proc.toarray()

    x_train, x_test, y_train, y_test = train_test_split(
        x_proc, y_reg, test_size=0.2, random_state=42
    )
    regresion = MLPRegressor(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        learning_rate_init=0.01,
        max_iter=250,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
    )
    regresion.fit(x_train, y_train)
    pred_reg = regresion.predict(x_test)

    error = y_test - pred_reg
    mse = mean_squared_error(y_test, pred_reg)
    metricas_reg = {
        "MSE": float(mse),
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_test, pred_reg)),
        "MAPE": float(mean_absolute_percentage_error(y_test, pred_reg)),
        "R2": float(r2_score(y_test, pred_reg)),
    }
    comparacion_reg = pd.DataFrame(
        {"Consumo real (kWh)": y_test, "Consumo predicho (kWh)": pred_reg}
    )
    comparacion_reg["Error (kWh)"] = comparacion_reg.iloc[:, 0] - comparacion_reg.iloc[:, 1]

    # ------------------------------------------------------------ clasificación
    x_clf = df[["Horas_Uso_Dia"]].to_numpy()
    y_clf = df["Alto_Consumo"].to_numpy()
    xc_train, xc_test, yc_train, yc_test = train_test_split(
        x_clf, y_clf, test_size=0.3, random_state=42
    )
    escalador_clf = StandardScaler().fit(xc_train)
    clasificador = MLPClassifier(
        hidden_layer_sizes=(16, 8),
        activation="relu",
        max_iter=300,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=42,
    )
    clasificador.fit(escalador_clf.transform(xc_train), yc_train)
    prob_clf = clasificador.predict_proba(escalador_clf.transform(xc_test))[:, 1]
    pred_clf = (prob_clf >= 0.5).astype(int)

    matriz = confusion_matrix(yc_test, pred_clf, labels=[0, 1])
    fpr, tpr, _ = roc_curve(yc_test, prob_clf)
    metricas_clf = {
        "Exactitud": float(accuracy_score(yc_test, pred_clf)),
        "Precisión": float(precision_score(yc_test, pred_clf, zero_division=0)),
        "Recall": float(recall_score(yc_test, pred_clf, zero_division=0)),
        "F1": float(f1_score(yc_test, pred_clf, zero_division=0)),
        "AUC": float(auc(fpr, tpr)),
    }
    curva_roc = pd.DataFrame({"Tasa de falsos positivos": fpr, "Tasa de verdaderos positivos": tpr})

    # ------------------------------------------------------------ serie temporal
    valores = serie.to_numpy(dtype=float).reshape(-1, 1)
    corte = int(len(valores) * 0.8)
    escalador_ts = StandardScaler().fit(valores[:corte])
    escalados = escalador_ts.transform(valores)

    x_train_ts, y_train_ts = _crear_secuencias(escalados[:corte], LONGITUD_SECUENCIA)
    contexto = np.vstack([escalados[corte - LONGITUD_SECUENCIA : corte], escalados[corte:]])
    x_test_ts, y_test_ts = _crear_secuencias(contexto, LONGITUD_SECUENCIA)

    serie_model = MLPRegressor(
        hidden_layer_sizes=(32,),
        activation="tanh",
        max_iter=400,
        early_stopping=True,
        n_iter_no_change=25,
        random_state=42,
    )
    serie_model.fit(x_train_ts, y_train_ts)

    pred_ts = escalador_ts.inverse_transform(
        serie_model.predict(x_test_ts).reshape(-1, 1)
    ).ravel()
    real_ts = escalador_ts.inverse_transform(y_test_ts.reshape(-1, 1)).ravel()
    fechas_ts = serie.index[corte:]

    persistencia = serie.to_numpy()[corte - 1 : -1]

    def _métricas(real: np.ndarray, predicho: np.ndarray) -> dict[str, float]:
        return {
            "MAE (kWh)": float(mean_absolute_error(real, predicho)),
            "RMSE (kWh)": float(np.sqrt(mean_squared_error(real, predicho))),
            "R²": float(r2_score(real, predicho)),
        }

    metricas_serie = {
        "Red neuronal": _métricas(real_ts, pred_ts),
        "Persistencia": _métricas(real_ts, persistencia),
    }
    predicciones_ts = pd.DataFrame(
        {
            "Fecha": fechas_ts,
            "Consumo real (kWh)": real_ts,
            "Predicción red (kWh)": pred_ts,
            "Predicción persistencia (kWh)": persistencia,
        }
    )

    return {
        "preprocesador": preprocesador,
        "regresion": regresion,
        "escalador_clf": escalador_clf,
        "clasificador": clasificador,
        "escalador_ts": escalador_ts,
        "serie_model": serie_model,
        "metricas_reg": metricas_reg,
        "error_reg": error,
        "comparacion_reg": comparacion_reg,
        "matriz": matriz,
        "metricas_clf": metricas_clf,
        "curva_roc": curva_roc,
        "metricas_serie": metricas_serie,
        "predicciones_ts": predicciones_ts,
    }


def predecir_consumo(modelos: dict[str, Any], horas: float, potencia: float,
                     consumo_kwh: float, electrodomestico: str) -> float:
    """Predicción puntual para el simulador de la interfaz."""
    entrada = pd.DataFrame(
        [
            {
                "Horas_Uso_Dia": horas,
                "Potencia_W": potencia,
                "Consumo_kWh": consumo_kwh,
                "Consumo_Teorico": (potencia * horas) / 1000.0,
                "Electrodomestico": electrodomestico,
            }
        ]
    )
    procesada = modelos["preprocesador"].transform(entrada)
    if hasattr(procesada, "toarray"):
        procesada = procesada.toarray()
    return float(modelos["regresion"].predict(procesada)[0])


def pronosticar(modelos: dict[str, Any], serie: pd.Series, pasos: int) -> pd.DataFrame:
    """Pronóstico recursivo a partir de los últimos días observados."""
    escalador = modelos["escalador_ts"]
    historial = escalador.transform(serie.to_numpy(dtype=float).reshape(-1, 1)).ravel()
    ventana = list(historial[-LONGITUD_SECUENCIA:])
    predicciones = []
    for _ in range(pasos):
        siguiente = float(modelos["serie_model"].predict(np.asarray(ventana).reshape(1, -1))[0])
        predicciones.append(float(escalador.inverse_transform([[siguiente]])[0, 0]))
        ventana = ventana[1:] + [siguiente]

    fechas = pd.date_range(serie.index[-1] + pd.Timedelta(days=1), periods=pasos, freq="D")
    return pd.DataFrame({"Fecha": fechas, "Consumo pronosticado (kWh)": predicciones})
