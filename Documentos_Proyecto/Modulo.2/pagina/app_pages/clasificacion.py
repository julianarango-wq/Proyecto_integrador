"""Clasificación binaria de consumo alto y bajo."""

import altair as alt
import pandas as pd
import streamlit as st

from utils.analysis import cargar_datos, entrenar_modelos, serie_diaria

df = cargar_datos()
modelos = entrenar_modelos(df, serie_diaria(df))
metricas = modelos["metricas_clf"]

st.title("Clasificación de consumo", icon=":material/donut_large:")
st.caption(
    "La variable objetivo marca como «alto consumo» los registros cuyo consumo diario total "
    "supera la mediana. El clasificador usa las horas de uso del equipo como predictor."
)

with st.container(horizontal=True):
    st.metric("Exactitud", f"{metricas['Exactitud']:.3f}", border=True)
    st.metric("Precisión", f"{metricas['Precisión']:.3f}", border=True)
    st.metric("Recall", f"{metricas['Recall']:.3f}", border=True)
    st.metric("F1", f"{metricas['F1']:.3f}", border=True)
    st.metric("AUC", f"{metricas['AUC']:.3f}", border=True)

col1, col2 = st.columns([1, 1], gap="medium")
with col1:
    with st.container(border=True):
        st.subheader("Matriz de confusión", icon=":material/grid_on:")
        matriz = pd.DataFrame(
            modelos["matriz"],
            index=["Bajo", "Alto"],
            columns=["Bajo", "Alto"],
        )
        larga = matriz.reset_index().melt(id_vars="index", var_name="Predicción", value_name="Casos")
        mapa = (
            alt.Chart(larga)
            .mark_rect()
            .encode(
                x=alt.X("Predicción:N", title="Predicción", axis=alt.Axis(labelAngle=0)),
                y=alt.Y("index:N", title="Valor real"),
                color=alt.Color("Casos:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["index", "Predicción", "Casos"],
            )
        )
        texto = mapa.mark_text(fontSize=16).encode(text=alt.Text("Casos:Q", format="d"))
        st.altair_chart(mapa + texto, alt="Matriz de confusión del clasificador de alto consumo")

with col2:
    with st.container(border=True):
        st.subheader("Curva ROC", icon=":material/show_chart:")
        roc = modelos["curva_roc"]
        curva = (
            alt.Chart(roc)
            .mark_line(color="#1D4ED8", strokeWidth=2)
            .encode(
                x=alt.X("Tasa de falsos positivos:Q", title="Tasa de falsos positivos"),
                y=alt.Y("Tasa de verdaderos positivos:Q", title="Tasa de verdaderos positivos"),
                tooltip=["Tasa de falsos positivos", "Tasa de verdaderos positivos"],
            )
        )
        diagonal = (
            alt.Chart(pd.DataFrame({"x": [0, 1], "y": [0, 1]}))
            .mark_line(color="gray", strokeDash=[5, 5])
            .encode(
                x=alt.X("x:Q", title="Tasa de falsos positivos"),
                y=alt.Y("y:Q", title="Tasa de verdaderos positivos"),
            )
        )
        st.altair_chart(curva + diagonal, alt="Curva ROC con AUC cercano a 0,5")

st.subheader("Distribución de la clase objetivo", icon=":material/pie_chart:")
conteo = (
    df["Alto_Consumo"]
    .map({0: "Bajo consumo", 1: "Alto consumo"})
    .value_counts()
    .rename("Registros")
)
st.bar_chart(conteo, alt="Cantidad de registros por clase de consumo")
st.warning(
    "El AUC cercano a 0,5 indica que las horas de uso apenas separan ambas clases. "
    "El resultado debe interpretarse como exploratorio, no como un clasificador confiable.",
    icon=":material/info:",
)
