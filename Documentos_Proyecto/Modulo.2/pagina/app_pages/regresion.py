"""Modelo de regresión y simulador de consumo diario total."""

import altair as alt
import pandas as pd
import streamlit as st

from utils.analysis import cargar_datos, entrenar_modelos, predecir_consumo, serie_diaria

df = cargar_datos()
serie = serie_diaria(df)
modelos = entrenar_modelos(df, serie)
metricas = modelos["metricas_reg"]

st.title("Predicción de consumo", icon=":material/query_stats:")
st.caption(
    "Red neuronal densa entrenada con horas de uso, potencia, consumo unitario, "
    "consumo teórico y tipo de equipo. El objetivo es el consumo diario total."
)

with st.container(horizontal=True):
    st.metric("R²", f"{metricas['R2']:.3f}", border=True)
    st.metric("RMSE", f"{metricas['RMSE']:.3f} kWh", border=True)
    st.metric("MAE", f"{metricas['MAE']:.3f} kWh", border=True)
    st.metric("MAPE", f"{metricas['MAPE'] * 100:.2f} %", border=True)

col1, col2 = st.columns([1, 1], gap="medium")
with col1:
    with st.container(border=True):
        st.subheader("Real versus predicho", icon=":material/scatter_plot:")
        muestra = modelos["comparacion_reg"].sample(600, random_state=42)
        puntos = (
            alt.Chart(muestra)
            .mark_circle(size=35, opacity=0.55)
            .encode(
                x=alt.X("Consumo real (kWh):Q", title="Consumo real (kWh)"),
                y=alt.Y("Consumo predicho (kWh):Q", title="Consumo predicho (kWh)"),
                tooltip=["Consumo real (kWh)", "Consumo predicho (kWh)"],
            )
        )
        limite = alt.Chart(
            pd.DataFrame({"x": [13, 25], "y": [13, 25]})
        ).mark_line(color="gray", strokeDash=[5, 5]).encode(
            x=alt.X("x:Q", title="Consumo real (kWh)"),
            y=alt.Y("y:Q", title="Consumo predicho (kWh)"),
        )
        st.altair_chart(puntos + limite, alt="Dispersión de valores reales frente a predichos")

with col2:
    with st.container(border=True):
        st.subheader("Distribución de errores", icon=":material/bar_chart:")
        errores = pd.Series(modelos["error_reg"], name="error")
        bins = pd.cut(errores, bins=16)
        histograma = bins.value_counts().sort_index().rename("Registros")
        histograma.index = [f"{i.left:.1f}" for i in histograma.index]
        st.bar_chart(histograma, alt="Distribución de los errores de predicción")

st.subheader("Simulador de predicción", icon=":material/tune:")
st.caption("Ajuste los valores y calcule el consumo diario total estimado para un equipo.")

with st.form("simulador_consumo", border=True):
    fila1, fila2 = st.columns(2)
    horas = fila1.number_input("Horas de uso al día", min_value=0.0, max_value=24.0, value=5.0, step=0.1)
    potencia = fila2.number_input("Potencia (W)", min_value=0.0, max_value=5000.0, value=150.0, step=10.0)
    fila3, fila4 = st.columns(2)
    consumo = fila3.number_input("Consumo unitario (kWh)", min_value=0.0, max_value=50.0, value=0.75, step=0.05)
    electro = fila4.selectbox("Electrodoméstico", sorted(df["Electrodomestico"].unique()))
    calcular = st.form_submit_button("Calcular consumo", type="primary", icon=":material/calculate:")

if calcular:
    resultado = predecir_consumo(modelos, horas, potencia, consumo, electro)
    teorico = (potencia * horas) / 1000.0
    with st.container(border=True):
        with st.container(horizontal=True):
            st.metric("Consumo diario total estimado", f"{resultado:.3f} kWh")
            st.metric("Consumo teórico del equipo", f"{teorico:.3f} kWh")
        st.caption(
            "La predicción corresponde al total diario del hogar, no solo al equipo seleccionado. "
            f"R² de {metricas['R2']:.3f}: úsela como orientación, no como medición exacta."
        )
