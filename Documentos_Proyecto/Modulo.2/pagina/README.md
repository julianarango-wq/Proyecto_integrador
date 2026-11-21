# Analítica de carga eléctrica

Dashboard web interactivo construido con **Streamlit** a partir del cuaderno de Gradio
`Copia_de_proyecto11.ipynb`. Incluye exploración de datos, regresión con simulador,
segmentación K-Means, series temporales con pronóstico, clasificación ROC y conclusiones.

La aplicación usa **scikit-learn** en lugar de TensorFlow para que el despliegue en la nube
sea ligero y confiable. Las métricas se calculan en tiempo de ejecución y se cachean.

## Estructura

```
.
├── streamlit_app.py          # Punto de entrada y navegación
├── app_pages/                # Resumen, exploración, modelos e informe
├── utils/analysis.py         # Carga de datos, modelos y métricas (cacheadas)
├── data/inventario_carga_.xlsx
├── .streamlit/config.toml    # Tema y configuración
└── requirements.txt
```

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Se abre automáticamente en `http://localhost:8501`.

## Desplegar en Streamlit Cloud

1. Sube esta carpeta a un repositorio de GitHub.
2. En [share.streamlit.io](https://share.streamlit.io) elige **New app**.
3. Selecciona el repositorio, la rama y `streamlit_app.py` como *Main file*.
4. Streamlit Cloud instalará `requirements.txt` automáticamente. No hacen falta secretos
   ni variables de entorno: el dataset viaja incluido en `data/`.

> El archivo `data/inventario_carga_.xlsx` debe permanecer en esa ruta: `utils/analysis.py`
> lo resuelve de forma relativa al proyecto, de modo que funciona igual en local y en la nube.

## Uso de la interfaz

- **Resumen**: indicadores clave y evolución del consumo diario (zoom y desplazamiento).
- **Exploración**: filtro por electrodoméstico, tabla, estadísticas, correlaciones y gráficas.
- **Predicción**: métricas del modelo de regresión y simulador de consumo.
- **Segmentación**: perfiles de los tres clústeres y método del codo.
- **Series temporales**: validación cronológica y pronóstico ajustable de 7 a 30 días.
- **Clasificación**: matriz de confusión, curva ROC y distribución de clases.
- **Conclusiones**: hallazgos, limitaciones y descarga del dataset.

## Notas

La interfaz usa componentes de Streamlit recientes (`st.navigation`, `st.echarts_chart`,
`st.segmented_control`, `st.badge`). Por eso `requirements.txt` fija `streamlit>=1.57`.
