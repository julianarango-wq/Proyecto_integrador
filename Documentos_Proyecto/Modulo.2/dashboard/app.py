"""Dashboard de Análisis de Carga Eléctrica y Modelos Predictivos.

Versión Streamlit del notebook `proyecto11.ipynb` (interfaz original en Gradio).
Incluye: EDA, regresión con RNA, clustering K-Means, serie temporal con RNN,
clasificación binaria, matriz de confusión y curva ROC.
"""

from __future__ import annotations

import base64
import json
from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, SimpleRNN
from tensorflow.keras.models import Sequential

# ------------------------------------------------------------------ config
st.set_page_config(
    page_title="Dashboard Inteligente - Inventario de Carga",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta cálida y tipografía moderna
sns.set_theme(
    style="whitegrid",
    rc={
        "grid.color": "#EBDCC9",
        "axes.facecolor": "#FFFDF8",
        "figure.facecolor": "#FFFDF8",
        "text.color": "#4A2F24",
        "axes.labelcolor": "#4A2F24",
        "xtick.color": "#8A6A58",
        "ytick.color": "#8A6A58",
    },
)
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Plus Jakarta Sans", "DejaVu Sans"]

BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [
    BASE_DIR / "data" / "inventario_carga_.xlsx",
    BASE_DIR.parent / "inventario_carga_.xlsx",
    Path.home() / "Downloads" / "inventario_carga_.xlsx",
]

# -------------------------------------------------------------- tema cálido
THEME_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root{
  --w-bg:#FFF6EC; --w-card:#FFFDF9; --w-primary:#C85A3C; --w-primary-2:#E07A3F;
  --w-accent:#E9A23B; --w-ink:#4A2F24; --w-muted:#8A6A58; --w-border:#EBD9C8;
}
html, body, [class*="css"]{
  font-family:'Plus Jakarta Sans','Segoe UI Variable','Segoe UI',sans-serif !important;
}
h1,h2,h3,h4{
  font-family:'Space Grotesk','Segoe UI Variable Display','Segoe UI',sans-serif !important;
  color:var(--w-ink) !important; letter-spacing:-.02em;
}
.stApp{ background:linear-gradient(135deg,#FFF9F2 0%,#FDF0E2 55%,#FBE6D3 100%); }
header[data-testid="stHeader"]{ background:rgba(255,248,240,.78) !important; backdrop-filter:blur(8px); }
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#FCE9D6 0%,#F7DCC3 100%) !important;
  border-right:1px solid var(--w-border);
}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label{ color:var(--w-ink) !important; }
[data-testid="stSidebar"] code{
  background:rgba(255,255,255,.6) !important; color:var(--w-ink) !important;
  border:1px solid var(--w-border); border-radius:8px;
}

/* Pestañas tipo píldora */
[data-testid="stTabs"] [role="tablist"]{
  overflow:visible !important; flex-wrap:wrap !important; gap:6px !important; padding-bottom:8px !important;
}
[data-testid="stTab"]{
  border-radius:999px !important; padding:8px 14px !important;
  border:1px solid var(--w-border) !important;
  background:rgba(255,255,255,.62) !important;
  transition:all .2s ease !important;
}
[data-testid="stTab"] p{
  color:var(--w-ink) !important; font-size:12.5px !important;
  font-weight:600 !important; margin:0 !important; white-space:nowrap !important;
}
[data-testid="stTab"]:hover{
  background:#FFF0E0 !important; border-color:var(--w-accent) !important;
  transform:translateY(-2px); box-shadow:0 6px 18px rgba(170,80,40,.16) !important;
}
[data-testid="stTab"][aria-selected="true"]{
  background:linear-gradient(135deg,var(--w-primary),var(--w-primary-2)) !important;
  border-color:transparent !important; box-shadow:0 8px 22px rgba(200,90,60,.3) !important;
}
[data-testid="stTab"][aria-selected="true"] p{ color:#FFF !important; }
.react-aria-SelectionIndicator{ display:none !important; }

/* Miniatura flotante al pasar el cursor por el menú */
[data-testid="stTab"]::after{
  position:absolute; top:calc(100% + 12px); left:0;
  width:264px; box-sizing:border-box;
  background-color:var(--w-card);
  border:1px solid var(--w-border); border-radius:16px;
  box-shadow:0 18px 42px rgba(120,60,30,.22);
  color:var(--w-ink); font-family:'Plus Jakarta Sans',sans-serif;
  font-size:11.5px; font-weight:600; line-height:1.45; text-align:center;
  opacity:0; visibility:hidden; transform:translateY(-8px);
  transition:opacity .18s ease, transform .18s ease, visibility .18s;
  pointer-events:none; z-index:99999; content:none;
}
[data-testid="stTab"]:hover::after,
[data-testid="stTab"]:focus::after{ opacity:1; visibility:visible; transform:translateY(0); }

/* Tarjetas, métricas y controles */
[data-testid="stMetric"]{
  background:var(--w-card) !important; border:1px solid var(--w-border) !important;
  border-radius:18px !important; padding:16px 18px !important;
  box-shadow:0 8px 22px rgba(150,85,50,.08) !important;
}
[data-testid="stMetricLabel"]{ color:var(--w-muted) !important; font-size:12px !important; font-weight:600 !important; }
[data-testid="stMetricValue"]{ color:var(--w-primary) !important; font-family:'Space Grotesk',sans-serif !important; }
[data-testid="stButton"] button{
  background:linear-gradient(135deg,var(--w-primary),var(--w-primary-2)) !important;
  color:#fff !important; border:none !important; border-radius:12px !important;
  box-shadow:0 6px 16px rgba(200,90,60,.25) !important;
  font-family:'Plus Jakarta Sans',sans-serif !important; font-weight:600 !important;
}
[data-testid="stButton"] button:hover{ filter:brightness(1.06); transform:translateY(-1px); }
[data-testid="stAlert"]{
  border-radius:14px !important; background:#FFF1E0 !important;
  border:1px solid #F2C79E !important; color:var(--w-ink) !important;
}
[data-testid="stContainer"]{
  background:rgba(255,253,249,.78) !important; border:1px solid var(--w-border) !important;
  border-radius:18px !important;
}
[data-testid="stDataFrame"]{ border:1px solid var(--w-border) !important; border-radius:14px !important; overflow:hidden !important; }
[data-testid="stCaptionContainer"]{ color:var(--w-muted) !important; }
[data-testid="stDivider"]{ border-color:var(--w-border) !important; }
::-webkit-scrollbar{ width:10px; height:10px; }
::-webkit-scrollbar-thumb{ background:#DDB89A; border-radius:10px; }
::-webkit-scrollbar-track{ background:#FFF6EC; }
"""


def _fig_to_b64(fig) -> str:
    """Convierte una figura matplotlib en PNG base64 para usarla miniatura CSS."""
    fig.set_size_inches(3.7, 2.05)
    fig.tight_layout(pad=0.4)
    buf = BytesIO()
    fig.savefig(
        buf,
        format="png",
        dpi=110,
        facecolor="#FFFDF8",
        bbox_inches="tight",
        pad_inches=0.06,
    )
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@st.cache_data(show_spinner=False)
def build_hover_thumbnails(
    df: pd.DataFrame,
    num_cols: list[str],
    errores: np.ndarray,
    df_clust: pd.DataFrame,
    y_test_ts: np.ndarray,
    y_pred_ts: np.ndarray,
    cm: np.ndarray,
    fpr: np.ndarray,
    tpr: np.ndarray,
    metrics_reg: dict,
    roc_auc: float,
) -> dict:
    """Genera una miniatura y un texto corto para cada pestaña del menú."""
    thumbs: dict = {}

    # 0. Explicación (solo texto)
    thumbs[0] = {
        "img": None,
        "caption": "EDA · RNA · K-Means · RNN · Clasificación",
    }

    # 1. Dataframe (mini tabla)
    fig, ax = plt.subplots()
    ax.axis("off")
    mini = df.head(3)[
        ["Electrodomestico", "Potencia_W", "Horas_Uso_Dia", "Consumo_kWh"]
    ]
    tabla = ax.table(
        cellText=[[str(v) for v in row] for row in mini.values],
        colLabels=list(mini.columns),
        cellLoc="center",
        loc="center",
    )
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(6)
    tabla.scale(1.05, 1.4)
    thumbs[1] = {
        "img": _fig_to_b64(fig),
        "caption": f"{len(df):,} registros · {df['Electrodomestico'].nunique()} equipos",
    }

    # 2. EDA
    fig, ax = plt.subplots()
    sns.histplot(df["Consumo_Diario_Total"], bins=30, color="#C85A3C", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    thumbs[2] = {
        "img": _fig_to_b64(fig),
        "caption": "Distribución y relación uso–consumo",
    }

    # 3. Correlación
    fig, ax = plt.subplots()
    sns.heatmap(df[num_cols].corr(), cmap="YlOrRd", ax=ax, cbar=False)
    thumbs[3] = {
        "img": _fig_to_b64(fig),
        "caption": "Correlaciones entre variables numéricas",
    }

    # 4. Regresión
    fig, ax = plt.subplots()
    sns.histplot(errores, bins=35, color="#E07A3F", ax=ax)
    ax.set_xlabel("")
    ax.set_ylabel("")
    thumbs[4] = {
        "img": _fig_to_b64(fig),
        "caption": (
            f"R² {metrics_reg['R2']:.2f} · RMSE {metrics_reg['RMSE']:.2f} "
            f"· MAPE {metrics_reg['MAPE']:.1f}%"
        ),
    }

    # 5. Clustering
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df_clust,
        x="Horas_Uso_Dia",
        y="Consumo_Diario_Total",
        hue="Cluster",
        palette=["#C85A3C", "#E9A23B", "#8B5E3C"],
        s=12,
        alpha=0.75,
        legend=False,
        ax=ax,
    )
    ax.set_xlabel("")
    ax.set_ylabel("")
    thumbs[5] = {
        "img": _fig_to_b64(fig),
        "caption": "3 clústeres por consumo, potencia y horas",
    }

    # 6. Series temporales
    fig, ax = plt.subplots()
    n = min(250, len(y_test_ts))
    ax.plot(y_test_ts[:n], color="#8B5E3C", lw=1.4, label="Real")
    ax.plot(y_pred_ts[:n], color="#E07A3F", lw=1.4, label="RNN")
    ax.set_xlabel("")
    ax.set_ylabel("")
    thumbs[6] = {
        "img": _fig_to_b64(fig),
        "caption": "Real vs. predicho con SimpleRNN",
    }

    # 7. Clasificación
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlOrRd", cbar=False, ax=ax)
    thumbs[7] = {
        "img": _fig_to_b64(fig),
        "caption": "Matriz de confusión y reporte de clasificación",
    }

    # 8. ROC
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="#C85A3C", lw=2)
    ax.plot([0, 1], [0, 1], color="#8B5E3C", lw=1, linestyle="--")
    ax.set_xlabel("")
    ax.set_ylabel("")
    thumbs[8] = {
        "img": _fig_to_b64(fig),
        "caption": f"Curva ROC · AUC {roc_auc:.2f}",
    }

    # 9. Conclusiones (solo texto)
    thumbs[9] = {
        "img": None,
        "caption": "Patrones, modelo, segmentación y series",
    }

    return thumbs


def apply_theme(thumbs: dict) -> None:
    """Inyecta el CSS cálido y las reglas de miniaturas por pestaña."""
    reglas = []
    for idx, item in thumbs.items():
        contenido = json.dumps(item["caption"], ensure_ascii=False)
        if item["img"]:
            estilo = (
                f"content:{contenido}; padding:152px 14px 12px; "
                f'background-image:url("data:image/png;base64,{item["img"]}"); '
                "background-repeat:no-repeat; background-position:12px 10px; "
                "background-size:238px 132px;"
            )
        else:
            estilo = f"content:{contenido}; padding:18px 16px; width:238px;"
        reglas.append(
            f'[data-testid="stTab"][data-key="{idx}"]::after{{{estilo}}}'
        )
    st.html("<style>" + THEME_CSS + "".join(reglas) + "</style>")


# ------------------------------------------------------------------- datos
@st.cache_data(show_spinner="Cargando dataset...")
def load_data() -> pd.DataFrame:
    for path in DATA_CANDIDATES:
        if path.exists():
            return pd.read_excel(path)
    raise FileNotFoundError(
        "No se encontró 'inventario_carga_.xlsx'. Colócalo en la carpeta 'data/'."
    )


# ------------------------------------------------------------------ modelos
@st.cache_resource(show_spinner="Entrenando modelos de IA (primera carga)...")
def train_models(df: pd.DataFrame) -> dict:
    """Entrena todos los modelos del notebook y devuelve métricas y artefactos."""
    resultado: dict = {}

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    resultado["num_cols"] = num_cols

    # --- 1. Red neuronal de regresión -------------------------------------
    X_reg = df[["Horas_Uso_Dia", "Electrodomestico"]].copy()
    y_reg = df["Consumo_Diario_Total"].to_numpy()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), ["Horas_Uso_Dia"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Electrodomestico"]),
        ]
    )
    X_reg_proc = preprocessor.fit_transform(X_reg)
    if hasattr(X_reg_proc, "toarray"):
        X_reg_proc = X_reg_proc.toarray()

    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg_proc, y_reg, test_size=0.2, random_state=42
    )

    model_reg = Sequential(
        [
            Input(shape=(X_train_r.shape[1],)),
            Dense(128, activation="relu"),
            Dense(64, activation="relu"),
            Dense(32, activation="relu"),
            Dense(16, activation="relu"),
            Dense(1),
        ]
    )
    model_reg.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss="mse",
        metrics=["mae"],
    )
    model_reg.fit(
        X_train_r, y_train_r, epochs=150, batch_size=16, verbose=0, validation_split=0.1
    )

    y_pred_r = model_reg.predict(X_test_r, verbose=0).flatten()
    mse_r = mean_squared_error(y_test_r, y_pred_r)
    resultado["preprocessor"] = preprocessor
    resultado["model_reg"] = model_reg
    resultado["errores_reg"] = y_test_r - y_pred_r
    resultado["metrics_reg"] = {
        "MSE": mse_r,
        "RMSE": np.sqrt(mse_r),
        "MAE": mean_absolute_error(y_test_r, y_pred_r),
        "MAPE": mean_absolute_percentage_error(y_test_r, y_pred_r) * 100,
        "R2": r2_score(y_test_r, y_pred_r),
    }

    # --- 2. Clustering K-Means --------------------------------------------
    scaler_clust = StandardScaler()
    scaled_features = scaler_clust.fit_transform(df[num_cols])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clust = df.copy()
    df_clust["Cluster"] = kmeans.fit_predict(scaled_features)
    resultado["df_clust"] = df_clust

    # --- 3. Serie temporal con SimpleRNN ----------------------------------
    df_temp = df.copy()
    df_temp["Fecha"] = pd.date_range(start="2026-01-01", periods=len(df_temp), freq="D")
    sequence_length = 5

    ts_data = df_temp["Consumo_Diario_Total"].to_numpy().reshape(-1, 1)
    scaler_ts = StandardScaler()
    ts_scaled = scaler_ts.fit_transform(ts_data)

    xs, ys = [], []
    for i in range(len(ts_scaled) - sequence_length):
        xs.append(ts_scaled[i : i + sequence_length])
        ys.append(ts_scaled[i + sequence_length])
    X_ts, y_ts = np.array(xs), np.array(ys)

    split_idx = int(len(X_ts) * 0.8)
    X_train_ts, X_test_ts = X_ts[:split_idx], X_ts[split_idx:]
    y_train_ts, y_test_ts = y_ts[:split_idx], y_ts[split_idx:]

    model_rnn = Sequential(
        [
            Input(shape=(sequence_length, 1)),
            SimpleRNN(32, activation="tanh"),
            Dense(1),
        ]
    )
    model_rnn.compile(optimizer="adam", loss="mse")
    model_rnn.fit(X_train_ts, y_train_ts, epochs=50, batch_size=4, verbose=0)

    y_pred_ts_scaled = model_rnn.predict(X_test_ts, verbose=0)
    resultado["y_pred_ts"] = scaler_ts.inverse_transform(y_pred_ts_scaled).flatten()
    resultado["y_test_ts"] = scaler_ts.inverse_transform(y_test_ts).flatten()
    resultado["fechas_ts"] = df_temp["Fecha"].iloc[sequence_length + split_idx :].values

    # --- 4. Clasificación binaria -----------------------------------------
    mediana_consumo = df["Consumo_Diario_Total"].median()
    y_clf = (df["Consumo_Diario_Total"] > mediana_consumo).astype(int).to_numpy()
    X_clf = df[["Horas_Uso_Dia"]]

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.3, random_state=42
    )

    model_clf = Sequential(
        [
            Input(shape=(X_train_c.shape[1],)),
            Dense(16, activation="relu"),
            Dense(8, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model_clf.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    model_clf.fit(X_train_c, y_train_c, epochs=50, batch_size=4, verbose=0)

    y_pred_prob_c = model_clf.predict(X_test_c, verbose=0).ravel()
    y_pred_c = (y_pred_prob_c > 0.5).astype(int)

    cm = confusion_matrix(y_test_c, y_pred_c)
    fpr, tpr, _ = roc_curve(y_test_c, y_pred_prob_c)
    resultado["cm"] = cm
    resultado["roc_auc"] = auc(fpr, tpr)
    resultado["fpr"] = fpr
    resultado["tpr"] = tpr
    resultado["report"] = classification_report(
        y_test_c,
        y_pred_c,
        target_names=["Bajo Consumo", "Alto Consumo"],
        output_dict=True,
    )
    resultado["accuracy"] = float((y_pred_c == y_test_c).mean())
    resultado["mediana_consumo"] = mediana_consumo

    return resultado


# ------------------------------------------------------------------ gráficos
def fig_distribucion(df: pd.DataFrame):
    fig, ax = plt.subplots()
    sns.histplot(df["Consumo_Diario_Total"], kde=True, color="#C85A3C", ax=ax)
    ax.set_title("Distribución de Consumo Diario Total")
    ax.set_xlabel("Consumo diario total (kWh)")
    fig.tight_layout()
    return fig


def fig_scatter(df: pd.DataFrame):
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df,
        x="Horas_Uso_Dia",
        y="Consumo_Diario_Total",
        hue="Electrodomestico",
        palette="flare",
        ax=ax,
        s=35,
        alpha=0.8,
    )
    ax.set_title("Horas de Uso vs Consumo Diario Total")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


def fig_correlacion(df: pd.DataFrame, num_cols: list[str]):
    fig, ax = plt.subplots()
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="YlOrRd", fmt=".2f", ax=ax)
    ax.set_title("Matriz de Correlación")
    fig.tight_layout()
    return fig


def fig_errores(errores: np.ndarray):
    fig, ax = plt.subplots()
    sns.histplot(errores, kde=True, color="#E07A3F", ax=ax)
    ax.set_title("Histograma de Errores de Predicción")
    ax.set_xlabel("Error (Real - Predicho)")
    fig.tight_layout()
    return fig


def fig_clustering(df_clust: pd.DataFrame):
    fig, ax = plt.subplots()
    sns.scatterplot(
        data=df_clust,
        x="Horas_Uso_Dia",
        y="Consumo_Diario_Total",
        hue="Cluster",
        palette=["#C85A3C", "#E9A23B", "#8B5E3C"],
        ax=ax,
        s=35,
        alpha=0.8,
    )
    ax.set_title("Clustering de Equipos (K-Means)")
    fig.tight_layout()
    return fig


def fig_series(m: dict, n_points: int):
    y_test = m["y_test_ts"][:n_points]
    y_pred = m["y_pred_ts"][:n_points]
    fechas = m["fechas_ts"][:n_points]
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(y_test))
    ax.plot(x, y_test, color="#8B5E3C", label="Real", marker="o", markersize=3, alpha=0.9)
    ax.plot(x, y_pred, color="#E07A3F", label="Predicción RNN", marker="x", markersize=3, alpha=0.8)
    ax.set_title("Predicción de Consumo en el Último Tramo de Tiempo")
    ax.set_xlabel("Días")
    ax.set_ylabel("Consumo diario total (kWh)")
    if len(fechas):
        ticks = np.linspace(0, len(x) - 1, min(6, len(x)), dtype=int)
        ax.set_xticks(ticks)
        ax.set_xticklabels([pd.Timestamp(f).strftime("%Y-%m-%d") for f in fechas[ticks]], rotation=30)
    ax.legend()
    fig.tight_layout()
    return fig


def fig_matriz_confusion(cm: np.ndarray):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=["Bajo", "Alto"],
        yticklabels=["Bajo", "Alto"],
        ax=ax,
    )
    ax.set_ylabel("Real")
    ax.set_xlabel("Predicción")
    ax.set_title("Matriz de Confusión")
    fig.tight_layout()
    return fig


def fig_roc(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float):
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color="#C85A3C", lw=2, label=f"Curva ROC (AUC = {roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], color="#8B5E3C", lw=2, linestyle="--")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Tasa de Falsos Positivos")
    ax.set_ylabel("Tasa de Verdaderos Positivos")
    ax.set_title("Curva ROC")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------- main
def main() -> None:
    st.title("⚡ Dashboard de Análisis de Carga Eléctrica y Modelos Predictivos")
    st.caption("Proyecto 11 · Notebook `proyecto11.ipynb` migrado de Gradio a Streamlit")

    df = load_data()
    m = train_models(df)

    thumbs = build_hover_thumbnails(
        df,
        m["num_cols"],
        m["errores_reg"],
        m["df_clust"],
        m["y_test_ts"],
        m["y_pred_ts"],
        m["cm"],
        m["fpr"],
        m["tpr"],
        m["metrics_reg"],
        m["roc_auc"],
    )
    apply_theme(thumbs)

    # --- Barra lateral ----------------------------------------------------
    with st.sidebar:
        st.header("📊 Dataset")
        st.metric("Registros", f"{len(df):,}")
        st.metric("Electrodomésticos", df["Electrodomestico"].nunique())
        st.metric("Consumo diario medio", f"{df['Consumo_Diario_Total'].mean():.2f} kWh")
        st.metric("Clústeres K-Means", m["df_clust"]["Cluster"].nunique())
        st.divider()
        st.markdown("**Columnas del dataset**")
        st.code(", ".join(df.columns), language="text")
        st.divider()
        st.caption("Modelos entrenados con TensorFlow · Interfaz Streamlit")

    # --- Indicadores principales ------------------------------------------
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² (Regresión)", f"{m['metrics_reg']['R2']:.4f}")
    col2.metric("RMSE (Regresión)", f"{m['metrics_reg']['RMSE']:.4f}")
    col3.metric("Accuracy (Clasificación)", f"{m['accuracy']:.2%}")
    col4.metric("AUC (ROC)", f"{m['roc_auc']:.2f}")

    # --- Pestañas ---------------------------------------------------------
    tabs = st.tabs(
        [
            "📘 Explicación",
            "📋 Dataframe & Estadísticas",
            "📈 Gráficas del EDA",
            "🔗 Correlación",
            "🧠 Red Neuronal (Regresión)",
            "🎯 Clustering ML",
            "⏳ Series Temporales",
            "✅ Clasificación",
            "📉 Curva ROC",
            "📌 Conclusiones",
        ]
    )

    # 0. Explicación
    with tabs[0]:
        st.markdown(
            """
            ### Descripción del Ejercicio

            En este proyecto se analiza el dataset de **inventario de cargas eléctricas**.

            - Se realiza un **análisis exploratorio de datos (EDA)** para entender los patrones de uso.
            - Se entrena una **Red Neuronal Artificial (RNA)** para predecir el **Consumo Diario Total**
              a partir de `Horas_Uso_Dia` y `Electrodomestico`.
            - Se aplica **K-Means** para segmentar el equipamiento según su patrón de consumo.
            - Se analiza la tendencia temporal del consumo con una **red recurrente (SimpleRNN)**.
            - Se diseña un **clasificador binario** para identificar equipos de consumo alto y bajo.

            Usa el menú de pestañas para recorrer cada una de las etapas del análisis.
            """
        )

    # 1. Dataframe y estadísticas
    with tabs[1]:
        st.markdown("### Muestra del Dataframe")
        st.dataframe(df.head(10), use_container_width=True)
        st.markdown("### Estadísticas Descriptivas")
        st.dataframe(df.describe(include="all"), use_container_width=True)

    # 2. Gráficas EDA
    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1:
            st.pyplot(fig_distribucion(df))
            st.caption("Distribución del consumo diario total.")
        with c2:
            st.pyplot(fig_scatter(df))
            st.caption("Relación entre horas de uso y consumo diario.")
        plt.close("all")

    # 3. Correlación
    with tabs[3]:
        st.pyplot(fig_correlacion(df, m["num_cols"]))
        plt.close("all")

    # 4. Regresión + simulador
    with tabs[4]:
        st.markdown("### Métricas de Evaluación del Modelo")
        mcols = st.columns(5)
        nombres = {
            "MSE": "MSE",
            "RMSE": "RMSE",
            "MAE": "MAE",
            "MAPE": "MAPE",
            "R2": "R²",
        }
        for c, (key, label) in zip(mcols, nombres.items()):
            value = m["metrics_reg"][key]
            c.metric(label, f"{value:.4f}%" if key == "MAPE" else f"{value:.4f}")

        st.pyplot(fig_errores(m["errores_reg"]))
        plt.close("all")

        st.markdown("### Simulador de Predicción de Consumo")
        with st.container(border=True):
            c1, c2 = st.columns(2)
            horas = c1.number_input(
                "Horas de uso al día", min_value=0.0, max_value=24.0, value=5.0, step=0.1
            )
            electro = c2.selectbox(
                "Electrodoméstico", options=sorted(df["Electrodomestico"].unique().tolist())
            )
            if st.button("Calcular consumo predicho", type="primary"):
                entrada = pd.DataFrame(
                    [[horas, electro]], columns=["Horas_Uso_Dia", "Electrodomestico"]
                )
                proc = m["preprocessor"].transform(entrada)
                if hasattr(proc, "toarray"):
                    proc = proc.toarray()
                pred = m["model_reg"].predict(proc, verbose=0)[0][0]
                st.success(f"**Consumo diario total predicho: {pred:.4f} kWh**")

    # 5. Clustering
    with tabs[5]:
        st.markdown(
            "### Agrupación por K-Means\n"
            "Se agruparon las cargas en 3 clústeres según su consumo, potencia y horas de uso."
        )
        st.pyplot(fig_clustering(m["df_clust"]))
        plt.close("all")
        resumen = (
            m["df_clust"]
            .groupby("Cluster")
            .agg(
                Registros=("Cluster", "size"),
                Consumo_medio=("Consumo_Diario_Total", "mean"),
                Horas_medias=("Horas_Uso_Dia", "mean"),
            )
            .round(2)
        )
        st.dataframe(resumen, use_container_width=True)

    # 6. Series temporales
    with tabs[6]:
        st.markdown(
            "### Análisis y Predicción con RNN\n"
            "Visualización del último tramo de la serie temporal comparado con las "
            "predicciones de la red neuronal recurrente."
        )
        max_points = len(m["y_test_ts"])
        n_points = st.slider(
            "Días a visualizar",
            min_value=min(50, max_points),
            max_value=max_points,
            value=min(200, max_points),
            step=50,
        )
        st.pyplot(fig_series(m, n_points))
        plt.close("all")

    # 7. Clasificación
    with tabs[7]:
        st.markdown("### Clasificador de Consumo Alto/Bajo (Red Neuronal)")
        st.caption(
            f"Umbral: consumo diario total > mediana ({m['mediana_consumo']:.2f} kWh) → Alto consumo"
        )
        c1, c2 = st.columns([1, 2])
        with c1:
            st.pyplot(fig_matriz_confusion(m["cm"]))
            plt.close("all")
        with c2:
            st.markdown("**Reporte de clasificación**")
            report_df = pd.DataFrame(m["report"]).T.loc[
                ["Bajo Consumo", "Alto Consumo", "macro avg", "weighted avg"]
            ]
            st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)
            st.metric("Accuracy global", f"{m['accuracy']:.2%}")

    # 8. ROC
    with tabs[8]:
        st.pyplot(fig_roc(m["fpr"], m["tpr"], m["roc_auc"]))
        plt.close("all")

    # 9. Conclusiones
    with tabs[9]:
        st.markdown(
            """
            ### Conclusiones Clave del Ejercicio

            1. **Patrón de consumo**: las `Horas_Uso_Dia` y la naturaleza del `Electrodomestico`
               determinan de forma lineal y no lineal el consumo diario total.
            2. **Precisión del modelo**: la red neuronal de regresión captura la tendencia general,
               aunque el R² obtenido indica que existen factores adicionales no incluidos en el modelo.
            3. **Segmentación efectiva**: K-Means separa los equipos de alta demanda de los de
               consumo esporádico o eficiente.
            4. **Series temporales**: las redes recurrentes permiten estimar picos de demanda
               futuros a partir de secuencias históricas.
            5. **Clasificación**: el modelo distingue entre consumo alto y bajo a partir de las
               horas de uso, con desempeño reflejado en la matriz de confusión y la curva ROC.
            """
        )


if __name__ == "__main__":
    main()
