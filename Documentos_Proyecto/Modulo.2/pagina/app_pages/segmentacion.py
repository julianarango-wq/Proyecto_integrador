"""Segmentación de equipos con K-Means."""

import pandas as pd
import streamlit as st

from utils.analysis import COLUMNAS_NUMERICAS, cargar_datos, curva_codo

df = cargar_datos()

st.title("Segmentación de equipos", icon=":material/hub:")
st.caption(
    "K-Means agrupa los registros en tres clústeres usando potencia, horas de uso, "
    "consumo unitario, horas activas y consumo diario total (variables estandarizadas)."
)

perfil = df.groupby("Cluster")[COLUMNAS_NUMERICAS].mean().round(3)
conteos = df["Cluster"].value_counts().sort_index()

with st.container(horizontal=True):
    st.metric("Clústeres", "3", border=True)
    st.metric("Registros segmentados", f"{len(df):,}", border=True)
    st.metric("Grupo más grande", f"{conteos.max():,}", border=True)


def etiqueta(fila: pd.Series) -> str:
    if fila["Potencia_W"] > 1000 and fila["Horas_Uso_Dia"] < 2:
        return "Alta potencia, uso breve"
    if fila["Potencia_W"] < 300 and fila["Horas_Uso_Dia"] > 8:
        return "Baja potencia, uso continuo"
    return "Uso intensivo mixto"


st.subheader("Perfil de cada clúster", icon=":material/groups:")
columnas = st.columns(3, gap="medium")
for columna, (cluster, fila) in zip(columnas, perfil.iterrows()):
    with columna:
        with st.container(border=True, height="stretch"):
            st.subheader(f"Clúster {cluster}")
            st.badge(etiqueta(fila))
            st.metric("Registros", f"{conteos[cluster]:,}")
            st.metric("Potencia promedio", f"{fila['Potencia_W']:.0f} W")
            st.metric("Horas de uso", f"{fila['Horas_Uso_Dia']:.2f} h/día")
            st.metric("Consumo unitario", f"{fila['Consumo_kWh']:.2f} kWh")

col1, col2 = st.columns([2, 1], gap="medium")
with col1:
    with st.container(border=True):
        st.subheader("Distribución de los clústeres", icon=":material/scatter_plot:")
        muestra = df.sample(min(2500, len(df)), random_state=42)
        st.scatter_chart(
            muestra,
            x="Horas_Uso_Dia",
            y="Consumo_Teorico",
            color="Cluster",
            x_label="Horas de uso al día",
            y_label="Consumo teórico (kWh)",
            alt="Clústeres K-Means según horas de uso y consumo teórico",
        )

with col2:
    with st.container(border=True):
        st.subheader("Registros por clúster", icon=":material/bar_chart:")
        st.bar_chart(conteos, alt="Cantidad de registros en cada clúster")

with st.expander("Ver tabla completa de promedios y método del codo"):
    st.dataframe(perfil, alt="Promedios de variables numéricas por clúster")
    codo = curva_codo(df)
    st.line_chart(
        codo.set_index("Clústeres"),
        y="Inercia",
        alt="Inercia del modelo K-Means para distintos números de clústeres",
    )
    st.caption(
        "El codo ayuda a justificar el número de grupos. La inercia siempre baja al aumentar k; "
        "se busca el punto donde la mejora deja de ser significativa."
    )
