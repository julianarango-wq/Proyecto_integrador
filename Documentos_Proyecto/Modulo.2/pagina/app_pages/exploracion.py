"""Exploración interactiva del dataset."""

import altair as alt
import pandas as pd
import streamlit as st

from utils.analysis import COLUMNAS_NUMERICAS, cargar_datos

df = cargar_datos()

st.title("Exploración de datos", icon=":material/table_chart:")
st.caption(
    "9.600 registros de ocho electrodomésticos entre 2021 y 2024. "
    "Filtre por equipo y cambie de vista para revisar la tabla, las estadísticas o las gráficas."
)

equipos = st.multiselect(
    "Electrodomésticos",
    options=sorted(df["Electrodomestico"].unique()),
    default=sorted(df["Electrodomestico"].unique()),
)
datos = df[df["Electrodomestico"].isin(equipos)]

vista = st.segmented_control(
    "Vista", ["Muestra de datos", "Estadísticas", "Gráficas"], default="Muestra de datos"
)

formato_numeros = {
    "Potencia_W": st.column_config.NumberColumn("Potencia (W)", format="%d"),
    "Horas_Uso_Dia": st.column_config.NumberColumn("Horas de uso", format="%.2f"),
    "Consumo_kWh": st.column_config.NumberColumn("Consumo (kWh)", format="%.3f"),
    "Consumo_Teorico": st.column_config.NumberColumn("Consumo teórico (kWh)", format="%.3f"),
    "H_activa": st.column_config.NumberColumn("Horas activas", format="%d"),
    "Consumo_Diario_Total": st.column_config.NumberColumn("Total diario (kWh)", format="%.3f"),
}

if vista == "Muestra de datos":
    st.subheader("Primeros registros", icon=":material/table_chart:")
    st.dataframe(
        datos.head(300),
        column_config=formato_numeros,
        hide_index=True,
        alt="Muestra de registros del inventario de carga eléctrica",
    )
    st.caption(f"Mostrando 300 de {len(datos):,} filas. Use la lupa de la tabla para buscar valores.")

elif vista == "Estadísticas":
    st.subheader("Estadísticas descriptivas", icon=":material/query_stats:")
    st.dataframe(
        datos[COLUMNAS_NUMERICAS].describe().round(3),
        alt="Estadísticas descriptivas de las variables numéricas",
    )

    st.subheader("Resumen por electrodoméstico", icon=":material/devices:")
    resumen = (
        datos.groupby("Electrodomestico")
        .agg(
            Potencia_W=("Potencia_W", "mean"),
            Horas_Uso_Dia=("Horas_Uso_Dia", "mean"),
            Consumo_kWh=("Consumo_kWh", "mean"),
        )
        .round(3)
        .sort_values("Consumo_kWh", ascending=False)
        .reset_index()
    )
    st.dataframe(
        resumen,
        column_config={
            "Electrodomestico": st.column_config.TextColumn("Electrodoméstico"),
            "Potencia_W": st.column_config.NumberColumn("Potencia (W)", format="%.0f"),
            "Horas_Uso_Dia": st.column_config.NumberColumn("Horas de uso", format="%.2f"),
            "Consumo_kWh": st.column_config.NumberColumn("Consumo (kWh)", format="%.3f"),
        },
        hide_index=True,
        alt="Promedios de potencia, horas de uso y consumo por electrodoméstico",
    )

else:
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        with st.container(border=True):
            st.subheader("Distribución del consumo diario", icon=":material/bar_chart:")
            bins = pd.cut(datos["Consumo_Diario_Total"], bins=14)
            distribucion = bins.value_counts().sort_index().rename("Registros")
            distribucion.index = [f"{inter.left:.1f}–{inter.right:.1f}" for inter in distribucion.index]
            st.bar_chart(distribucion, alt="Histograma del consumo diario total en kWh")

    with col2:
        with st.container(border=True):
            st.subheader("Correlación entre variables", icon=":material/grid_on:")
            matriz = (
                datos[COLUMNAS_NUMERICAS]
                .corr()
                .round(2)
                .reset_index()
                .melt(id_vars="index", var_name="Variable", value_name="Correlación")
            )
            mapa = (
                alt.Chart(matriz)
                .mark_rect()
                .encode(
                    x=alt.X("Variable:N", title=None),
                    y=alt.Y("index:N", title=None),
                    color=alt.Color("Correlación:Q", scale=alt.Scale(scheme="blueorange", domain=[-1, 1])),
                    tooltip=["index", "Variable", "Correlación"],
                )
            )
            etiquetas = mapa.mark_text(fontSize=10).encode(text=alt.Text("Correlación:Q", format=".2f"))
            st.altair_chart(mapa + etiquetas, alt="Mapa de calor de correlaciones entre variables numéricas")

    with st.container(border=True):
        st.subheader("Horas de uso frente al consumo", icon=":material/scatter_plot:")
        muestra = datos.sample(min(1500, len(datos)), random_state=42)
        grafica = (
            alt.Chart(muestra)
            .mark_circle(size=45, opacity=0.65)
            .encode(
                x=alt.X("Horas_Uso_Dia:Q", title="Horas de uso al día"),
                y=alt.Y("Consumo_kWh:Q", title="Consumo del equipo (kWh)"),
                color=alt.Color("Electrodomestico:N", title="Electrodoméstico"),
                tooltip=["Electrodomestico", "Horas_Uso_Dia", "Consumo_kWh"],
            )
        )
        st.altair_chart(grafica, alt="Relación entre horas de uso y consumo por electrodoméstico")
