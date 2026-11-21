"""Hallazgos del proyecto, alcance de los modelos y descarga del dataset."""

import pandas as pd
import streamlit as st

from utils.analysis import ARCHIVO_DATOS, cargar_datos, entrenar_modelos, serie_diaria

df = cargar_datos()
serie = serie_diaria(df)
modelos = entrenar_modelos(df, serie)

reg = modelos["metricas_reg"]
clf = modelos["metricas_clf"]
serie_m = modelos["metricas_serie"]

st.title("Conclusiones", icon=":material/description:")
st.caption(
    "Síntesis del análisis, desempeño de los modelos y recomendaciones para uso futuro. "
    "Todas las cifras se recalculan en ejecución a partir del dataset incluido."
)

st.subheader("Hallazgos principales", icon=":material/lightbulb:")
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    with st.container(border=True, height="stretch"):
        st.subheader("1 · El volumen es estable")
        st.write(
            f"El inventario reúne **{len(df):,} registros** de "
            f"**{df['Electrodomestico'].nunique()} electrodomésticos** a lo largo de "
            f"**{len(serie):,} días consecutivos**. El consumo diario se mantiene en un rango "
            f"estrecho, con un promedio de **{serie.mean():.2f} kWh** y una desviación de "
            f"**{serie.std():.2f} kWh**, sin tendencias abruptas."
        )

with col2:
    with st.container(border=True, height="stretch"):
        st.subheader("2 · Los equipos se agrupan bien")
        st.write(
            "K-Means encuentra tres perfiles claros: equipos de **alta potencia y uso breve**, "
            "de **baja potencia y uso continuo**, y de **uso intensivo mixto**. "
            "Esta segmentación es útil para dirigir campañas de eficiencia por tipo de equipo "
            "y no por electrodoméstico individual."
        )

with col3:
    with st.container(border=True, height="stretch"):
        st.subheader("3 · Predecir es más difícil que describir")
        st.write(
            f"La regresión explica solo **{reg['R2'] * 100:.1f} %** de la varianza y el "
            f"clasificador de consumo alto roza el azar (**AUC {clf['AUC']:.3f}**). "
            "El modelo secuencial sí mejora a la persistencia, pero con un poder explicativo "
            "limitado: los datos son regulares y no aportan señales discriminatorias fuertes."
        )

st.subheader("Resumen de desempeño", icon=":material/monitoring:")
resultados = pd.DataFrame(
    [
        {
            "Modelo": "Regresión (MLP)",
            "Métrica principal": f"R² = {reg['R2']:.3f}",
            "Error": f"RMSE = {reg['RMSE']:.3f} kWh",
            "Interpretación": "Baja capacidad explicativa; orientativo.",
        },
        {
            "Modelo": "Clasificación (MLP)",
            "Métrica principal": f"AUC = {clf['AUC']:.3f}",
            "Error": f"Exactitud = {clf['Exactitud']:.3f}",
            "Interpretación": "Casi aleatorio con horas de uso como único predictor.",
        },
        {
            "Modelo": "Serie temporal (MLP)",
            "Métrica principal": f"R² = {serie_m['Red neuronal']['R²']:.3f}",
            "Error": f"MAE = {serie_m['Red neuronal']['MAE (kWh)']:.3f} kWh",
            "Interpretación": "Supera a la persistencia; útil para tendencias suaves.",
        },
    ]
)
st.dataframe(
    resultados,
    hide_index=True,
    width="stretch",
    column_config={
        "Modelo": st.column_config.TextColumn("Modelo"),
        "Métrica principal": st.column_config.TextColumn("Métrica principal"),
        "Error": st.column_config.TextColumn("Error"),
        "Interpretación": st.column_config.TextColumn("Interpretación", width="large"),
    },
    alt="Tabla comparativa del desempeño de los tres modelos",
)

st.subheader("Alcance y limitaciones", icon=":material/rule:")
with st.expander("Leer antes de usar los modelos", expanded=True):
    st.markdown(
        """
        - **Datos sintéticos y regulares.** El consumo diario total se construye a partir de
          pocas variables y repetimos una estructura estable durante 1.200 días. Esto reduce
          la variabilidad que normalmente aprovechan los modelos.
        - **El objetivo no es el equipo, es el total del hogar.** El simulador estima el
          consumo diario total, no el consumo aislado del electrodoméstico seleccionado.
        - **La clasificación es exploratoria.** Con un AUC cercano a 0,5, las horas de uso
          por sí solas no separan consumo alto de bajo. Se necesitan más variables (tipo de
          equipo, potencia, horario, temperatura, ocupación).
        - **Los resultados se calculan en caliente.** Las métricas se obtienen al ejecutar la
          aplicación con scikit-learn; no son cifras copiadas del cuaderno original con
          TensorFlow, por lo que pueden variar ligeramente de una versión a otra.
        """
    )

st.subheader("Recomendaciones", icon=":material/checklist:")
st.success(
    "Priorizar la segmentación para acciones de eficiencia, usar la serie temporal para "
    "pronósticos de corto plazo y tratar la regresión y la clasificación como apoyo "
    "exploratorio hasta incorporar nuevas variables y datos reales.",
    icon=":material/thumb_up:",
)

st.subheader("Descarga del dataset", icon=":material/download:")
with st.container(border=True):
    st.write(
        f"**{ARCHIVO_DATOS.name}** · {len(df):,} filas · "
        f"{ARCHIVO_DATOS.stat().st_size / 1024:.0f} KB · Excel (.xlsx)"
    )
    st.download_button(
        "Descargar dataset",
        data=ARCHIVO_DATOS.read_bytes(),
        file_name=ARCHIVO_DATOS.name,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        icon=":material/download:",
        type="primary",
    )

st.caption("Proyecto de analítica de carga eléctrica · Streamlit + scikit-learn")
