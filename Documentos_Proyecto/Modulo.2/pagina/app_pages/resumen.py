"""Vista general del inventario de carga."""

import pandas as pd
import streamlit as st

from utils.analysis import cargar_datos, serie_diaria

df = cargar_datos()
serie = serie_diaria(df)

st.title("Analítica de carga eléctrica", icon=":material/bolt:")
st.caption(
    "Exploración, modelado y visualización del consumo energético de electrodomésticos. "
    "Use el menú lateral para recorrer el análisis."
)

with st.container(horizontal=True):
    st.metric("Registros", f"{len(df):,}", border=True)
    st.metric("Electrodomésticos", df["Electrodomestico"].nunique(), border=True)
    st.metric("Días analizados", f"{len(serie):,}", border=True)
    st.metric("Consumo diario promedio", f"{serie.mean():.2f} kWh", border=True)

st.subheader("Consumo diario total", icon=":material/timeline:")
st.caption("Arrastre sobre la gráfica o use el deslizador inferior para explorar un periodo.")

diario = serie.reset_index()
diario["Fecha"] = pd.to_datetime(diario["Fecha"]).dt.strftime("%Y-%m-%d")
st.echarts_chart(
    {
        "tooltip": {"trigger": "axis"},
        "dataset": {"source": diario},
        "xAxis": {"type": "time"},
        "yAxis": {"type": "value", "name": "kWh"},
        "dataZoom": [{"type": "inside"}, {"type": "slider", "height": 20}],
        "series": [
            {
                "type": "line",
                "name": "Consumo diario",
                "showSymbol": False,
                "smooth": True,
                "lineStyle": {"width": 2},
            }
        ],
    },
    alt="Evolución del consumo diario total entre 2021 y 2024, con valores alrededor de 20 kWh",
)

col1, col2 = st.columns([1, 1], gap="medium")
with col1:
    with st.container(border=True):
        st.subheader("Consumo por equipo", icon=":material/bar_chart:")
        promedios = (
            df.groupby("Electrodomestico")["Consumo_kWh"]
            .mean()
            .sort_values(ascending=False)
            .rename("Consumo promedio (kWh)")
        )
        st.bar_chart(promedios, horizontal=True, alt="Consumo promedio por electrodoméstico")

with col2:
    with st.container(border=True):
        st.subheader("Perfil del dataset", icon=":material/info:")
        st.table(
            {
                ":material/calendar_today: Periodo": f"{serie.index.min():%d %b %Y} – {serie.index.max():%d %b %Y}",
                ":material/bolt: Consumo mínimo": f"{serie.min():.2f} kWh",
                ":material/bolt: Consumo máximo": f"{serie.max():.2f} kWh",
                ":material/query_stats: Desviación estándar": f"{serie.std():.2f} kWh",
                ":material/devices: Equipos únicos": f"{df['Electrodomestico'].nunique()}",
            },
            border="horizontal",
            width="content",
        )
