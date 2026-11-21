"""Punto de entrada del dashboard de analítica de carga eléctrica."""

import streamlit as st

st.set_page_config(
    page_title="Analítica de carga eléctrica",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="expanded",
)

pagina = st.navigation(
    {
        "": [
            st.Page(
                "app_pages/resumen.py",
                title="Resumen",
                icon=":material/dashboard:",
                help="Indicadores clave del inventario de carga y evolución del consumo diario.",
            )
        ],
        "Análisis": [
            st.Page(
                "app_pages/exploracion.py",
                title="Exploración de datos",
                icon=":material/table_chart:",
                help="Muestra del dataset, estadísticas descriptivas y gráficas por electrodoméstico.",
            )
        ],
        "Modelos": [
            st.Page(
                "app_pages/regresion.py",
                title="Predicción de consumo",
                icon=":material/query_stats:",
                help="Métricas del modelo de regresión y simulador de consumo diario total.",
            ),
            st.Page(
                "app_pages/segmentacion.py",
                title="Segmentación",
                icon=":material/hub:",
                help="Clústeres K-Means de los equipos y sus perfiles de consumo.",
            ),
            st.Page(
                "app_pages/series.py",
                title="Series temporales",
                icon=":material/timeline:",
                help="Evolución diaria, validación del modelo secuencial y pronóstico futuro.",
            ),
            st.Page(
                "app_pages/clasificacion.py",
                title="Clasificación",
                icon=":material/donut_large:",
                help="Clasificador de consumo alto/bajo con matriz de confusión y curva ROC.",
            ),
        ],
        "Informe": [
            st.Page(
                "app_pages/conclusiones.py",
                title="Conclusiones",
                icon=":material/description:",
                help="Hallazgos del proyecto, alcance de los modelos y descarga del dataset.",
            )
        ],
    },
    position="sidebar",
    expanded=True,
)

with st.sidebar:
    st.caption("Inventario de carga · 2021–2024 · scikit-learn")

pagina.run()
