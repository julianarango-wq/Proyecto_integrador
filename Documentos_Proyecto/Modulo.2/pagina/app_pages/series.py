"""Análisis de la serie temporal de consumo diario."""

import pandas as pd
import streamlit as st

from utils.analysis import cargar_datos, entrenar_modelos, pronosticar, serie_diaria

df = cargar_datos()
serie = serie_diaria(df)
modelos = entrenar_modelos(df, serie)
metricas = modelos["metricas_serie"]
predicciones = modelos["predicciones_ts"]

st.title("Series temporales", icon=":material/timeline:")
st.caption(
    "La serie diaria tiene 1.200 días consecutivos. El modelo secuencial usa los últimos "
    "5 días para predecir el siguiente y se valida en el 20 % final, cronológicamente separado."
)

st.subheader("Serie completa", icon=":material/show_chart:")
diario = serie.reset_index()
diario["Fecha"] = pd.to_datetime(diario["Fecha"]).dt.strftime("%Y-%m-%d")
st.echarts_chart(
    {
        "tooltip": {"trigger": "axis"},
        "dataset": {"source": diario},
        "xAxis": {"type": "time"},
        "yAxis": {"type": "value", "name": "kWh"},
        "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 20}],
        "series": [{"type": "line", "showSymbol": False, "smooth": True, "name": "Consumo"}],
    },
    alt="Serie diaria de consumo total con 1.200 observaciones",
)

st.subheader("Validación en el periodo de prueba", icon=":material/verified:")
comparacion = predicciones.copy()
comparacion["Fecha"] = pd.to_datetime(comparacion["Fecha"]).dt.strftime("%Y-%m-%d")
st.echarts_chart(
    {
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 0},
        "dataset": {"source": comparacion},
        "xAxis": {"type": "time"},
        "yAxis": {"type": "value", "name": "kWh"},
        "dataZoom": [{"type": "inside"}],
        "series": [
            {"type": "line", "name": "Consumo real", "showSymbol": False},
            {"type": "line", "name": "Predicción red", "showSymbol": False},
            {"type": "line", "name": "Persistencia", "showSymbol": False, "lineStyle": {"type": "dashed"}},
        ],
    },
    alt="Comparación del consumo real con la predicción del modelo y de la persistencia",
)

col1, col2 = st.columns(2, gap="medium")
with col1:
    with st.container(border=True):
        st.subheader("Modelo secuencial", icon=":material/model_training:")
        m = metricas["Red neuronal"]
        st.metric("MAE", f"{m['MAE (kWh)']:.3f} kWh", border=True)
        st.metric("RMSE", f"{m['RMSE (kWh)']:.3f} kWh", border=True)
        st.metric("R²", f"{m['R²']:.3f}", border=True)

with col2:
    with st.container(border=True):
        st.subheader("Modelo base: persistencia", icon=":material/straighten:")
        b = metricas["Persistencia"]
        st.metric("MAE", f"{b['MAE (kWh)']:.3f} kWh", border=True)
        st.metric("RMSE", f"{b['RMSE (kWh)']:.3f} kWh", border=True)
        st.metric("R²", f"{b['R²']:.3f}", border=True)

st.caption(
    "La persistencia repite el valor del día anterior. El modelo secuencial la supera en las "
    "tres métricas, aunque su capacidad explicativa sigue siendo limitada."
)

st.subheader("Pronóstico", icon=":material/trending_up:")
dias = st.slider("Días a pronosticar", min_value=7, max_value=30, value=14, step=7)
pronostico = pronosticar(modelos, serie, dias)
historial = serie.tail(60).reset_index()
historial["Fecha"] = pd.to_datetime(historial["Fecha"]).dt.strftime("%Y-%m-%d")
historial = historial.rename(columns={"Consumo diario (kWh)": "Observado"})
union = pd.concat(
    [
        historial[["Fecha", "Observado"]],
        pronostico.assign(Fecha=pronostico["Fecha"].dt.strftime("%Y-%m-%d"))
        .rename(columns={"Consumo pronosticado (kWh)": "Pronóstico"})[["Fecha", "Pronóstico"]],
    ],
    ignore_index=True,
)
st.echarts_chart(
    {
        "tooltip": {"trigger": "axis"},
        "legend": {"top": 0},
        "dataset": {"source": union},
        "xAxis": {"type": "time"},
        "yAxis": {"type": "value", "name": "kWh"},
        "series": [
            {
                "type": "line",
                "name": "Observado",
                "showSymbol": False,
                "encode": {"x": "Fecha", "y": "Observado"},
            },
            {
                "type": "line",
                "name": "Pronóstico",
                "showSymbol": False,
                "lineStyle": {"type": "dashed"},
                "encode": {"x": "Fecha", "y": "Pronóstico"},
            },
        ],
    },
    alt=f"Pronóstico del consumo diario para los próximos {dias} días",
)
st.caption(
    f"Pronóstico recursivo a {dias} días a partir de los últimos valores observados. "
    "Las predicciones se generan una a una alimentando el modelo con la predicción anterior."
)
