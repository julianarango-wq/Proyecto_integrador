# Dashboard de Carga Eléctrica — Streamlit

Migración del notebook `proyecto11.ipynb` (interfaz original en Gradio) a una aplicación Streamlit.

**Interfaz:** paleta cálida (terracota/ámbar/crema), tipografías modernas (Plus Jakarta Sans y
Space Grotesk) y miniaturas descriptivas que aparecen al pasar el cursor —o al enfocar con
el teclado— sobre cada pestaña del menú.

## Ejecutar en local

```powershell
cd "D:\personal Julian\unicomfacauca\diplomado\Documentos\Modulo.2\dashboard"
pip install -r requirements.txt
streamlit run app.py
```

La app se abre en `http://localhost:8501`. La primera carga entrena los modelos (puede tardar
un par de minutos); las siguientes cargas son instantáneas gracias al caché de Streamlit.

## Estructura

```
dashboard/
├── app.py                       # Aplicación Streamlit (10 pestañas)
├── requirements.txt             # Dependencias
├── runtime.txt                  # Python 3.12 (requerido por TensorFlow)
├── data/
│   └── inventario_carga_.xlsx   # Dataset (9600 registros)
└── README.md
```

## Desplegar en Streamlit Community Cloud

1. Sube esta carpeta a un repositorio de GitHub (incluye `data/inventario_carga_.xlsx`).
2. En <https://share.streamlit.io> → **New app**, selecciona el repo, la rama y `app.py` como
   archivo principal.
3. En **Advanced settings → Python dependencies**, asegúrate de que `requirements.txt`
   se detecte automáticamente (o pega su contenido).
4. Clic en **Deploy**.

> Nota: `tensorflow-cpu` mantiene la imagen de despliegue más ligera que `tensorflow` y es
> suficiente para entrenar y predecir con las redes del proyecto.
