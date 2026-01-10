# app.py
import warnings

warnings.filterwarnings("ignore")

import io
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scipy import stats

import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    RandomForestRegressor,
    GradientBoostingRegressor,
)
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose


# =========================
# App config
# =========================
st.set_page_config(
    page_title="Ultra Analytics Pro",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"Get Help": "https://docs.streamlit.io"},
)

# =========================
# Styling (compact + safe)
# =========================
st.markdown(
    """
<style>
/* 1. Global Animations */
@keyframes neuralEntrance {
    0% { opacity: 0; filter: blur(10px); transform: translateY(10px); }
    100% { opacity: 1; filter: blur(0px); transform: translateY(0); }
}

@keyframes bluePulse {
    0% { border-color: rgba(79, 70, 229, 0.3); box-shadow: 0 0 10px rgba(79, 70, 229, 0.1); }
    100% { border-color: rgba(124, 58, 237, 0.6); box-shadow: 0 0 20px rgba(124, 58, 237, 0.3); }
}

/* 2. Reset Graphs to Normal Size */
[data-testid="stImage"] img {
    border-radius: 0px !important;
    box-shadow: none !important;
    width: 100% !important;
    height: auto !important;
}

/* 3. Targeted Styling for Profile Pic Only */
.profile-container img {
    border-radius: 50% !important;
    border: 3px solid #4f46e5 !important;
    box-shadow: 0 0 25px rgba(79, 70, 229, 0.7) !important;
    object-fit: cover !important;
    width: 200px !important;
    height: 200px !important;
}

.profile-container {
    display: flex;
    justify-content: center;
    padding-bottom: 20px;
}

/* 4. Global Card & Tab Beautifier */
.card, div[data-testid="metric-container"], .stTabs [data-baseweb="tab"] {
    background: rgba(255, 255, 255, 0.04) !important;
    backdrop-filter: blur(15px);
    border: 1px solid rgba(167, 139, 250, 0.2) !important;
    border-radius: 18px !important;
    animation: bluePulse 4s infinite alternate;
}

.h1 {
    font-size: 2.5rem; font-weight: 950;
    background: linear-gradient(90deg, #fff, #4f46e5);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
</style>
""",
    unsafe_allow_html=True,
)


# =========================
# Session state
# =========================
def ss_init():
    defaults = {
        "df_raw": None,
        "df": None,
        "uploaded_name": None,
        "uploaded_time": None,
        "plotly_template": "plotly_white",
        "random_state": 42,
        "nav_query": "",
        "nav_section": "Core",
        "page_nav": "🏠 Dashboard Home",
        # model / report
        "last_model_bundle": None,  # dict with pipeline + metadata
        "last_run_report": None,  # dict with lightweight report data
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


ss_init()


# =========================
# Utilities
# =========================
@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file is None:
        return None
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file type. Upload CSV or Excel.")


def df_memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum() / 1024**2)


def download_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def to_joblib_bytes(obj) -> bytes:
    buf = io.BytesIO()
    joblib.dump(obj, buf)
    return buf.getvalue()


def require_df():
    if st.session_state.df is None:
        st.warning("⚠️ Upload a dataset from the sidebar first.")
        st.stop()


def get_numeric_cols(df: pd.DataFrame):
    return df.select_dtypes(include=[np.number]).columns.tolist()


def get_categorical_cols(df: pd.DataFrame):
    return df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()


def get_datetime_like_cols(df: pd.DataFrame):
    dt = df.select_dtypes(
        include=["datetime64[ns]", "datetime64[ns, UTC]"]
    ).columns.tolist()
    for c in df.columns:
        cl = str(c).lower()
        if c not in dt and any(
            k in cl for k in ["date", "time", "datetime", "timestamp"]
        ):
            dt.append(c)
    seen, out = set(), []
    for c in dt:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def make_preprocessor(df: pd.DataFrame, feature_cols):
    X = df[feature_cols].copy()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in feature_cols if c not in num_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    pre = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


def plot_feature_importance_from_pipeline(pipe, top_k=20):
    model = pipe.named_steps.get("model", None)
    pre = pipe.named_steps.get("pre", None)
    if model is None or pre is None:
        return None
    if not hasattr(model, "feature_importances_"):
        return None

    try:
        feat_names = pre.get_feature_names_out()
    except Exception:
        feat_names = None

    importances = model.feature_importances_
    if feat_names is None or len(feat_names) != len(importances):
        feat_names = [f"f{i}" for i in range(len(importances))]

    imp = (
        pd.DataFrame({"feature": feat_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(top_k)
    )
    fig = px.bar(
        imp[::-1],
        x="importance",
        y="feature",
        orientation="h",
        title="Top feature importances",
    )
    fig.update_layout(template=st.session_state.plotly_template, height=520)
    return fig


def build_report(task, target, features, metrics: dict):
    df = st.session_state.df
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": st.session_state.uploaded_name,
        "rows": int(len(df)) if df is not None else None,
        "cols": int(len(df.columns)) if df is not None else None,
        "missing": int(df.isnull().sum().sum()) if df is not None else None,
        "task": task,
        "target": target,
        "features": list(features),
        "metrics": metrics,
    }


def report_to_html(report: dict) -> bytes:
    html = f"""
    <html>
    <head><meta charset="utf-8"><title>Ultra Analytics Report</title></head>
    <body style="font-family:Arial; padding:24px;">
      <h2>Ultra Analytics Report</h2>
      <p><b>Created:</b> {report.get("created_at")}</p>

      <h3>Dataset</h3>
      <ul>
        <li><b>Name:</b> {report.get("dataset")}</li>
        <li><b>Rows:</b> {report.get("rows")}</li>
        <li><b>Cols:</b> {report.get("cols")}</li>
        <li><b>Missing:</b> {report.get("missing")}</li>
      </ul>

      <h3>Model</h3>
      <ul>
        <li><b>Task:</b> {report.get("task")}</li>
        <li><b>Target:</b> {report.get("target")}</li>
        <li><b>Features:</b> {", ".join(report.get("features", []))}</li>
      </ul>

      <h3>Metrics</h3>
      <pre>{report.get("metrics")}</pre>
    </body>
    </html>
    """.encode(
        "utf-8"
    )
    return html


# =========================
# Navigation
# =========================
NAV = {
    "Core": [
        "🏠 Dashboard Home",
        "📊 Data Explorer",
        "🧰 Data Transform",
        "📊 Matplotlib Lab",
        "📈 Exploratory Analysis",
        "📉 Advanced Statistics",
    ],
    "Modeling": [
        "🎯 Predictive Modeling",
        "🧠 Machine Learning",
        "🧪 Model Playground",
        "🔮 Time Series Forecast",
    ],
    "Discovery": [
        "🎨 Pattern Discovery",
        "📐 Statistical Testing",
    ],
    "Utilities": [
        "💾 Data Management",
        "⚙️ Settings",
        "👤 Profile",
        "👤 My CV",
    ],
}


# =========================
# Sidebar
# =========================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-card">
          <div class="sidebar-title">🚀 Ultra Analytics Pro</div>
          <div class="sidebar-sub">Single-file improved</div>
          <div class="badge">v5.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.session_state.nav_query = st.text_input(
        "Search pages",
        value=st.session_state.nav_query,
        placeholder="Type to filter…",
        label_visibility="collapsed",
    )

    all_pages = [(section, p) for section, pages in NAV.items() for p in pages]
    if st.session_state.nav_query.strip():
        q = st.session_state.nav_query.strip().lower()
        all_pages = [(s, p) for (s, p) in all_pages if q in p.lower()]
    if not all_pages:
        st.warning("No pages match your search.")
        all_pages = [(section, p) for section, pages in NAV.items() for p in pages]

    # Quick jump across all pages
    all_flat = [p for _, p in [(s, p) for s, ps in NAV.items() for p in ps]]
    jump = st.selectbox(
        "Quick jump",
        all_flat,
        index=(
            all_flat.index(st.session_state.page_nav)
            if st.session_state.page_nav in all_flat
            else 0
        ),
    )
    if jump != st.session_state.page_nav:
        st.session_state.page_nav = jump
        for s, ps in NAV.items():
            if jump in ps:
                st.session_state.nav_section = s
                break
        st.rerun()

    sections = list(dict.fromkeys([s for s, _ in all_pages]))
    if st.session_state.nav_section not in sections:
        st.session_state.nav_section = sections[0]

    st.session_state.nav_section = st.selectbox(
        "Section",
        sections,
        index=sections.index(st.session_state.nav_section),
    )
    section_pages = [p for s, p in all_pages if s == st.session_state.nav_section]

    if st.session_state.page_nav not in section_pages:
        st.session_state.page_nav = section_pages[0]

    st.session_state.page_nav = st.radio(
        "Page",
        section_pages,
        index=section_pages.index(st.session_state.page_nav),
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### 📦 Dataset")
    uploaded = st.file_uploader(
        "Upload CSV/Excel", type=["csv", "xlsx", "xls"], key="uploader"
    )

    if uploaded is not None:
        try:
            df_loaded = load_data(uploaded)
            st.session_state.df_raw = df_loaded
            st.session_state.df = df_loaded.copy()
            st.session_state.uploaded_name = uploaded.name
            st.session_state.uploaded_time = datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            st.success("Loaded successfully.")
        except Exception as e:
            st.error(f"Load error: {e}")

    if st.session_state.df is not None:
        df = st.session_state.df
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(df):,}")
        c2.metric("Cols", f"{len(df.columns)}")
        c3, c4 = st.columns(2)
        c3.metric("Missing", f"{int(df.isnull().sum().sum()):,}")
        c4.metric("MB", f"{df_memory_mb(df):.1f}")

        if st.button("🔁 Restore original", use_container_width=True):
            st.session_state.df = st.session_state.df_raw.copy()
            st.success("Restored.")
            st.rerun()

        if st.button("🧹 Clear dataset", use_container_width=True):
            st.session_state.df_raw = None
            st.session_state.df = None
            st.session_state.uploaded_name = None
            st.session_state.uploaded_time = None
            st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### 🤖 Model bundle")
    model_file = st.file_uploader(
        "Load .joblib model bundle", type=["joblib"], key="model_uploader"
    )
    if model_file is not None:
        try:
            bundle = joblib.load(model_file)
            if not isinstance(bundle, dict) or "pipeline" not in bundle:
                st.error("Invalid bundle. Expected dict with at least 'pipeline'.")
            else:
                st.session_state.last_model_bundle = bundle
                st.success("Model bundle loaded.")
        except Exception as e:
            st.error(f"Model load failed: {e}")

    if st.session_state.last_model_bundle is not None:
        b = st.session_state.last_model_bundle
        st.caption(
            f"Loaded task: {b.get('task')} • target: {b.get('target')} • trained: {b.get('trained_at')}"
        )


# =========================
# Pages
# =========================
def page_cv():
    st.markdown(
        '<div class="h1">👤 Professional Curriculum Vitae</div>', unsafe_allow_html=True
    )

    # CSS to force the Streamlit image into a circle with a glow
    st.markdown(
        """
        <style>
        [data-testid="stImage"] img {
            border-radius: 50%;
            border: 4px solid #7c3aed;
            box-shadow: 0 0 20px rgba(124, 58, 237, 0.6);
            object-fit: cover;
            width: 200px !important;
            height: 200px !important;
        }
        /* Center the image container */
        [data-testid="stImage"] {
            display: flex;
            justify-content: center;
        }
        </style>
    """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns([1, 2.2])

    with c1:
        st.markdown('<div class="profile-container">', unsafe_allow_html=True)

        img_url = "profile.jpg"
        st.image("profile.jpg")

        st.markdown(
            "<h2 style='text-align:center; margin-top:10px;'>Ahmad Fayyaz</h2>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center; color:#a78bfa; font-weight:bold;'>BS Statistics | Data Scientist</p>",
            unsafe_allow_html=True,
        )
        st.write("🎂 **Age:** 21")
        st.write("📖 **Hafiz-e-Quran**")
        st.write("🏓 **Interests:** Table Tennis")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎓 Education")
        st.write("**Bachelor of Science in Statistics**")
        st.caption("Specialization: Data Science")
        st.write("• **Secondary:** Government College Township")
        st.write("• **Primary:** The Educators")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("💼 Experience")
        st.write("**Sales Representative** | Geek Tech")
        st.write("**Property Management** | Coordination & Logistics")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🚀 Technical Skills")
        st.write("• **Python:** Pandas, NumPy, Scikit-Learn")
        st.write("• **Visualization:** Plotly, Streamlit, Shiny")
        st.write("• **ML:** Decision Trees, Regression, Classification")
        st.markdown("</div>", unsafe_allow_html=True)


def page_home():
    st.markdown('<div class="h1">🏠 Dashboard Home</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p">Global system overview, dataset health diagnostics, and raw data stream.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.df is None:
        st.markdown(
            '<div class="card">⚠️ System Idle. Please ingest a datastream via the sidebar.</div>',
            unsafe_allow_html=True,
        )
        return

    df = st.session_state.df

    # 🔵 1. TOP PERFORMANCE METRICS
    num = get_numeric_cols(df)
    cat = get_categorical_cols(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Numeric Channels", len(num))
    c2.metric("Total Rows", f"{len(df):,}")
    c3.metric("Data Duplicates", int(df.duplicated().sum()))
    c4.metric("Memory Usage", f"{df_memory_mb(df):.1f} MB")

    # 🔵 2. NEURAL HEALTH SCAN (Matplotlib Charts)
    st.markdown("### 🛠️ Neural Health Scan")
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Data Integrity (Missing Values)**")
        null_counts = df.isnull().sum()
        null_counts = null_counts[null_counts > 0].sort_values(ascending=False)

        if not null_counts.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            plt.style.use("dark_background")  # cite: 1.2
            null_counts.plot(kind="bar", color="#4f46e5", ax=ax)  # cite: 1.2
            st.pyplot(fig)  # cite: 1.4
        else:
            st.success("✨ Pulse Clear: 0 Missing Values Detected.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Feature Skewness Overview**")
        if len(num) > 0:
            skew = df[num].skew().head(5)
            fig2, ax2 = plt.subplots(figsize=(8, 5))
            plt.style.use("dark_background")  # cite: 1.2
            skew.plot(kind="barh", color="#7c3aed", ax=ax2)  # cite: 1.2
            st.pyplot(fig2)  # cite: 1.4
        else:
            st.info("No numeric features available for skew analysis.")
        st.markdown("</div>", unsafe_allow_html=True)

    # 🔵 3. NEURAL PULSE (Quick Clean Action)
    st.markdown("### ⚡ System Actions")
    if st.button("🚀 INITIATE NEURAL PULSE (Auto-Clean)"):
        with st.spinner("🧠 Optimizing Neural Pathway..."):
            # Remove duplicates and drop rows with any missing values
            df_cleaned = df.drop_duplicates().dropna()
            st.session_state.df = df_cleaned
            st.success(
                f"Pulse Complete: System now running at peak efficiency ({len(df_cleaned)} rows remain)."
            )
            st.rerun()

    # 🔵 4. DATASET SNAPSHOT (Restored & Beautified)
    st.markdown("### 📡 Active Data Stream Snapshot")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if st.session_state.uploaded_name:
        st.caption(
            f"Source: {st.session_state.uploaded_name} | Trained by: Ahmad Fayyaz"
        )

    st.dataframe(df.head(50), use_container_width=True, height=400)
    st.markdown("</div>", unsafe_allow_html=True)


def page_data_explorer():
    require_df()
    df = st.session_state.df

    st.markdown('<div class="h1">📊 Data Explorer</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p">Browse, clean, and inspect schema.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(["🔎 Browse", "🧼 Clean", "🧾 Schema"])

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            n = st.slider("Rows to show", 10, min(500, len(df)), 50, key="de_rows")
        with c2:
            sort_col = st.selectbox(
                "Sort by", ["(none)"] + list(df.columns), key="de_sort"
            )
        with c3:
            desc = st.toggle("Descending", value=False, key="de_desc")

        view = df.copy()
        if sort_col != "(none)":
            view = view.sort_values(by=sort_col, ascending=not desc)

        st.dataframe(view.head(n), use_container_width=True, height=520)
        st.download_button(
            "⬇️ Download current view (CSV)",
            data=download_csv_bytes(view.head(n)),
            file_name="data_view.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        missing = df.isnull().sum().sort_values(ascending=False)
        miss_df = pd.DataFrame({"column": missing.index, "missing": missing.values})
        miss_df["missing_pct"] = (miss_df["missing"] / len(df) * 100).round(2)

        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Missing values")
            st.dataframe(miss_df, use_container_width=True, height=380)
        with c2:
            top = miss_df.head(15)
            fig = px.bar(
                top,
                x="missing",
                y="column",
                orientation="h",
                title="Top missing columns",
            )
            fig.update_layout(template=st.session_state.plotly_template, height=420)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader("Duplicates")
        dup = int(df.duplicated().sum())
        st.metric("Duplicate rows", f"{dup:,}")
        if dup > 0 and st.button("Remove duplicates", key="de_rmdup"):
            st.session_state.df = df.drop_duplicates().reset_index(drop=True)
            st.success("Duplicates removed.")
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        schema = pd.DataFrame(
            {
                "column": df.columns,
                "dtype": [str(t) for t in df.dtypes],
                "non_null": df.count().values,
                "null": df.isnull().sum().values,
                "unique": df.nunique().values,
            }
        )
        st.dataframe(schema, use_container_width=True, height=520)
        st.markdown("</div>", unsafe_allow_html=True)


def page_transform():
    require_df()
    df = st.session_state.df

    st.markdown('<div class="h1">🧰 Data Transform</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p">Type casting + simple feature engineering + safe export.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)

    num = get_numeric_cols(df)
    tab1, tab2, tab3 = st.tabs(["🔢 Type casting", "🧪 Features", "💾 Export"])

    with tab1:
        st.subheader("Convert columns to numeric (safe)")
        cols = st.multiselect(
            "Columns",
            list(df.columns),
            default=num[: min(6, len(num))],
            key="tf_num_cols",
        )
        if st.button("Convert selected to numeric", key="tf_num_go"):
            new = df.copy()
            for c in cols:
                new[c] = pd.to_numeric(new[c], errors="coerce")
            st.session_state.df = new
            st.success("Converted. Non-numeric values became NaN.")
            st.rerun()

        st.subheader("Parse datetime column")
        dt_col = st.selectbox("Column", list(df.columns), key="tf_dt_col")
        if st.button("Parse as datetime", key="tf_dt_go"):
            new = df.copy()
            new[dt_col] = pd.to_datetime(new[dt_col], errors="coerce")
            st.session_state.df = new
            st.success("Parsed datetime. Invalid values became NaT.")
            st.rerun()

    with tab2:
        st.subheader("Create ratio feature")
        if len(num) >= 2:
            a = st.selectbox("Numerator", num, key="tf_ratio_a")
            b = st.selectbox(
                "Denominator", [c for c in num if c != a], key="tf_ratio_b"
            )
            name = st.text_input(
                "New feature name", value=f"{a}_per_{b}", key="tf_ratio_name"
            )
            if st.button("Add ratio", key="tf_ratio_go"):
                new = df.copy()
                A = pd.to_numeric(new[a], errors="coerce")
                B = pd.to_numeric(new[b], errors="coerce").replace(0, np.nan)
                new[name] = A / B
                st.session_state.df = new
                st.success("Ratio feature added.")
                st.rerun()
        else:
            st.info("Need at least 2 numeric columns.")

    with tab3:
        st.subheader("Download current working dataset")
        st.download_button(
            "⬇️ Download (CSV)",
            data=download_csv_bytes(df),
            file_name="dataset_processed.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def page_matplotlib_lab():
    require_df()
    df = st.session_state.df
    num_cols = get_numeric_cols(df)

    st.markdown(
        '<div class="h1">📊 Statistical Insight Lab</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="p">Visualize and export publication-quality charts using Matplotlib.</div>',
        unsafe_allow_html=True,
    )

    if len(num_cols) < 1:
        st.warning("This lab requires numeric data.")
        return

    # Helper function to convert Matplotlib figure to downloadable bytes
    def get_plot_bytes(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        return buf.getvalue()

    # --- ROW 1: Distribution & Outliers ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        st.write("**1. Data Density**")
        col_dist = st.selectbox("Select variable", num_cols, key="plt_dist")
        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.hist(
            df[col_dist].dropna(),
            bins=30,
            color="#7c3aed",
            edgecolor="white",
            alpha=0.7,
        )
        ax1.set_title(f"Density of {col_dist}")
        st.pyplot(fig1)
        st.download_button(
            "💾 Save Density Plot",
            get_plot_bytes(fig1),
            f"{col_dist}_density.png",
            "image/png",
        )

    with c2:
        st.write("**2. Anomaly Detection**")
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        ax2.boxplot(df[col_dist].dropna())
        ax2.set_title(f"Outliers: {col_dist}")
        st.pyplot(fig2)
        st.download_button(
            "💾 Save Boxplot",
            get_plot_bytes(fig2),
            f"{col_dist}_boxplot.png",
            "image/png",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- ROW 2: Relationships & Correlations ---
    st.markdown('<div class="card">', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        if len(num_cols) >= 2:
            st.write("**3. Variable Interaction**")
            x_c = st.selectbox("X Axis", num_cols, index=0)
            y_c = st.selectbox("Y Axis", num_cols, index=1 if len(num_cols) > 1 else 0)
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            ax3.scatter(df[x_c], df[y_c], alpha=0.5, color="#f472b6")
            ax3.set_xlabel(x_c)
            ax3.set_ylabel(y_c)
            st.pyplot(fig3)
            st.download_button(
                "💾 Save Scatter Plot",
                get_plot_bytes(fig3),
                "interaction.png",
                "image/png",
            )
        else:
            st.info("Requires 2+ numeric columns.")

    with c4:
        st.write("**4. Network Correlation**")
        if len(num_cols) > 1:
            fig4, ax4 = plt.subplots(figsize=(8, 5))
            corr = df[num_cols].corr()
            im = ax4.imshow(corr, cmap="Purples")
            plt.colorbar(im, ax=ax4)
            ax4.set_xticks(range(len(num_cols)))
            ax4.set_yticks(range(len(num_cols)))
            ax4.set_xticklabels(num_cols, rotation=45)
            ax4.set_yticklabels(num_cols)
            st.pyplot(fig4)
            st.download_button(
                "💾 Save Heatmap", get_plot_bytes(fig4), "correlation.png", "image/png"
            )
    st.markdown("</div>", unsafe_allow_html=True)


def page_eda():
    require_df()
    df = st.session_state.df
    num = get_numeric_cols(df)
    cat = get_categorical_cols(df)

    st.markdown('<div class="h1">📈 Exploratory Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p">Distributions, outliers, correlations, and relationships.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3 = st.tabs(
        ["📊 Distributions", "🧊 Correlations", "🧷 Relationships"]
    )

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if not num:
            st.info("No numeric columns found.")
        else:
            c1, c2 = st.columns([1, 1])
            with c1:
                col = st.selectbox("Numeric column", num, key="eda_dist_col")
                bins = st.slider("Bins", 10, 120, 40, key="eda_bins")
                fig = px.histogram(
                    df, x=col, nbins=bins, marginal="box", title=f"Distribution: {col}"
                )
                fig.update_layout(template=st.session_state.plotly_template, height=460)
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                x = pd.to_numeric(df[col], errors="coerce").dropna()
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Mean", f"{x.mean():.3f}" if len(x) else "NA")
                m2.metric("Median", f"{x.median():.3f}" if len(x) else "NA")
                m3.metric("Skew", f"{x.skew():.3f}" if len(x) else "NA")
                m4.metric("Kurt", f"{x.kurtosis():.3f}" if len(x) else "NA")

                st.subheader("Outliers (IQR)")
                if len(x) >= 8:
                    q1, q3 = x.quantile(0.25), x.quantile(0.75)
                    iqr = q3 - q1
                    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
                    out = x[(x < lo) | (x > hi)]
                    st.metric(
                        "Outliers", f"{len(out):,}", f"{(len(out)/len(x)*100):.2f}%"
                    )
                else:
                    st.info("Need more data for outlier detection.")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if len(num) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            method = st.radio(
                "Correlation method",
                ["pearson", "spearman", "kendall"],
                horizontal=True,
                key="eda_corr_m",
            )
            corr = df[num].corr(method=method, numeric_only=True)
            fig = px.imshow(
                corr,
                text_auto=".2f",
                title=f"{method.capitalize()} correlation heatmap",
                color_continuous_scale="RdBu_r",
                zmin=-1,
                zmax=1,
            )
            fig.update_layout(template=st.session_state.plotly_template, height=620)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if len(num) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            c1, c2 = st.columns(2)
            with c1:
                xcol = st.selectbox("X", num, key="eda_rel_x")
            with c2:
                ycol = st.selectbox("Y", [c for c in num if c != xcol], key="eda_rel_y")

            color = None
            if cat:
                color = st.selectbox(
                    "Color by (optional)", ["(none)"] + cat, key="eda_rel_color"
                )
                if color == "(none)":
                    color = None

            fig = px.scatter(
                df,
                x=xcol,
                y=ycol,
                color=color,
                trendline="ols",
                title=f"{xcol} vs {ycol}",
            )
            fig.update_layout(template=st.session_state.plotly_template, height=560)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


def page_adv_stats():
    require_df()
    df = st.session_state.df
    num = get_numeric_cols(df)
    cat = get_categorical_cols(df)

    st.markdown('<div class="h1">📉 Advanced Statistics</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p">Normality checks, t-test, ANOVA, and correlation tests.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📐 Normality", "🧪 T-test", "🧩 ANOVA", "🔗 Correlation test"]
    )

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if not num:
            st.info("No numeric columns found.")
        else:
            col = st.selectbox("Numeric column", num, key="as_norm_col")
            x = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(x) < 8:
                st.error("Need at least 8 observations for Shapiro test.")
            else:
                xs = x.sample(
                    min(5000, len(x)), random_state=st.session_state.random_state
                )
                stat, p = stats.shapiro(xs)
                c1, c2 = st.columns(2)
                c1.metric("Shapiro stat", f"{stat:.4f}")
                c2.metric("p-value", f"{p:.6f}")
                st.caption(
                    "Note: for large samples, tiny deviations can produce small p-values."
                )
        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if len(num) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            a = st.selectbox("Group A", num, key="as_t_a")
            b = st.selectbox("Group B", [c for c in num if c != a], key="as_t_b")
            if st.button("Run t-test"):
                xa = pd.to_numeric(df[a], errors="coerce").dropna()
                xb = pd.to_numeric(df[b], errors="coerce").dropna()
                t, p = stats.ttest_ind(xa, xb, equal_var=False)
                c1, c2 = st.columns(2)
                c1.metric("t-stat", f"{t:.4f}")
                c2.metric("p-value", f"{p:.6f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if not (num and cat):
            st.info("Need at least 1 numeric and 1 categorical column.")
        else:
            y = st.selectbox("Numeric outcome", num, key="as_aov_y")
            g = st.selectbox("Group (categorical)", cat, key="as_aov_g")
            if st.button("Run ANOVA"):
                groups = [
                    pd.to_numeric(v, errors="coerce").dropna().values
                    for _, v in df.groupby(g)[y]
                ]
                groups = [x for x in groups if len(x) >= 2]
                if len(groups) < 2:
                    st.error("Need at least 2 groups with 2+ observations each.")
                else:
                    f, p = stats.f_oneway(*groups)
                    c1, c2 = st.columns(2)
                    c1.metric("F-stat", f"{f:.4f}")
                    c2.metric("p-value", f"{p:.6f}")
                    fig = px.box(df, x=g, y=y, color=g, title=f"{y} by {g}")
                    fig.update_layout(
                        template=st.session_state.plotly_template, height=520
                    )
                    st.plotly_chart(fig, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with tab4:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if len(num) < 2:
            st.info("Need at least 2 numeric columns.")
        else:
            a = st.selectbox("X", num, key="as_cor_a")
            b = st.selectbox("Y", [c for c in num if c != a], key="as_cor_b")
            method = st.selectbox(
                "Method", ["pearson", "spearman", "kendall"], key="as_cor_m"
            )
            if st.button("Run correlation test"):
                xa = pd.to_numeric(df[a], errors="coerce")
                xb = pd.to_numeric(df[b], errors="coerce")
                mask = xa.notna() & xb.notna()
                xa, xb = xa[mask], xb[mask]
                if len(xa) < 6:
                    st.error("Not enough paired observations.")
                else:
                    r, p = (
                        stats.pearsonr(xa, xb)
                        if method == "pearson"
                        else (
                            stats.spearmanr(xa, xb)
                            if method == "spearman"
                            else stats.kendalltau(xa, xb)
                        )
                    )
                    c1, c2 = st.columns(2)
                    c1.metric("Correlation", f"{float(r):.4f}")
                    c2.metric("p-value", f"{float(p):.6f}")
        st.markdown("</div>", unsafe_allow_html=True)


def page_predictive_modeling():
    require_df()
    df = st.session_state.df
    num = get_numeric_cols(df)
    cat = get_categorical_cols(df)

    st.markdown('<div class="h1">🎯 Predictive Modeling</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p">Train and compare multiple models for classification and regression.</div>',
        unsafe_allow_html=True,
    )

    tab1, tab2 = st.tabs(
        ["🧾 Classification (multi-model)", "📈 Regression (multi-model)"]
    )

    with tab1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if not (num and cat):
            st.info("Need numeric features and a categorical target column.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            target = st.selectbox("Target (categorical)", cat, key="pm_clf_target")
            feats = st.multiselect(
                "Features", num, default=num[: min(6, len(num))], key="pm_clf_feats"
            )
            test_size = st.slider("Test split", 0.1, 0.5, 0.2, key="pm_clf_ts")
            if feats and st.button("Train & compare", key="pm_clf_go"):
                X = df[feats].apply(pd.to_numeric, errors="coerce")
                y = df[target].astype(str)
                mask = X.notna().all(axis=1) & y.notna()
                X, y = X.loc[mask], y.loc[mask]
                if len(X) < 20:
                    st.error(
                        "Not enough clean rows after removing missing values (need ~20+)."
                    )
                    st.stop()

                le = LabelEncoder()
                y_enc = le.fit_transform(y)
                class_counts = pd.Series(y_enc).value_counts()
                strat = y_enc if class_counts.min() >= 2 else None

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y_enc,
                    test_size=test_size,
                    random_state=st.session_state.random_state,
                    stratify=strat,
                )

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                models = {
                    "Logistic Regression": LogisticRegression(
                        max_iter=2000, random_state=st.session_state.random_state
                    ),
                    "Random Forest": RandomForestClassifier(
                        n_estimators=300, random_state=st.session_state.random_state
                    ),
                    "Gradient Boosting": GradientBoostingClassifier(
                        random_state=st.session_state.random_state
                    ),
                    "SVM (RBF)": SVC(
                        kernel="rbf", random_state=st.session_state.random_state
                    ),
                    "Decision Tree": DecisionTreeClassifier(
                        random_state=st.session_state.random_state
                    ),
                    "KNN": KNeighborsClassifier(n_neighbors=5),
                    "Naive Bayes": GaussianNB(),
                }

                rows, best = [], None
                for name, model in models.items():
                    try:
                        model.fit(X_train_s, y_train)
                        pred = model.predict(X_test_s)
                        acc = float(accuracy_score(y_test, pred))
                        rows.append({"Model": name, "Accuracy": acc})
                        if best is None or acc > best[1]:
                            best = (name, acc, pred)
                    except Exception as e:
                        rows.append(
                            {"Model": name, "Accuracy": np.nan, "Error": str(e)[:140]}
                        )

                res = pd.DataFrame(rows).sort_values(
                    "Accuracy", ascending=False, na_position="last"
                )
                st.dataframe(res, use_container_width=True)

                if best is not None:
                    best_name, best_acc, best_pred = best
                    st.success(f"Best model: {best_name} (Accuracy={best_acc:.3f})")
                    cm = confusion_matrix(y_test, best_pred)
                    fig = px.imshow(
                        cm, text_auto=True, title=f"Confusion matrix: {best_name}"
                    )
                    fig.update_layout(
                        template=st.session_state.plotly_template, height=520
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    st.text("Classification report:")
                    st.code(
                        classification_report(
                            y_test, best_pred, target_names=le.classes_
                        ),
                        language="text",
                    )

            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if len(num) < 2:
            st.info("Need at least 2 numeric columns.")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            target = st.selectbox("Target (numeric)", num, key="pm_reg_target")
            feats = st.multiselect(
                "Features",
                [c for c in num if c != target],
                default=[c for c in num if c != target][: min(6, len(num) - 1)],
                key="pm_reg_feats",
            )
            test_size = st.slider("Test split", 0.1, 0.5, 0.2, key="pm_reg_ts")

            if feats and st.button("Train & compare", key="pm_reg_go"):
                X = df[feats].apply(pd.to_numeric, errors="coerce")
                y = pd.to_numeric(df[target], errors="coerce")

                mask = X.notna().all(axis=1) & y.notna()
                X, y = X.loc[mask], y.loc[mask]

                if len(X) < 30:
                    st.error(
                        "Not enough clean rows after removing missing values (need ~30+)."
                    )
                    st.stop()
                if float(np.nanstd(y.values)) == 0.0:
                    st.error("Target has zero variance.")
                    st.stop()

                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_size,
                    random_state=st.session_state.random_state,
                )

                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                models = {
                    "Linear Regression": LinearRegression(),
                    "Ridge": Ridge(
                        alpha=1.0, random_state=st.session_state.random_state
                    ),
                    "Lasso": Lasso(
                        alpha=0.1, random_state=st.session_state.random_state
                    ),
                    "Random Forest": RandomForestRegressor(
                        n_estimators=300, random_state=st.session_state.random_state
                    ),
                    "Gradient Boosting": GradientBoostingRegressor(
                        random_state=st.session_state.random_state
                    ),
                    "SVR (RBF)": SVR(kernel="rbf"),
                    "Decision Tree": DecisionTreeRegressor(
                        random_state=st.session_state.random_state
                    ),
                    "KNN": KNeighborsRegressor(n_neighbors=5),
                }

                rows, best = [], None
                preds = {}

                for name, model in models.items():
                    try:
                        model.fit(X_train_s, y_train)
                        pred = model.predict(X_test_s)
                        r2 = float(r2_score(y_test, pred))
                        rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
                        mae = float(mean_absolute_error(y_test, pred))
                        rows.append({"Model": name, "R2": r2, "RMSE": rmse, "MAE": mae})
                        preds[name] = pred
                        if best is None or r2 > best[1]:
                            best = (name, r2)
                    except Exception as e:
                        rows.append(
                            {
                                "Model": name,
                                "R2": np.nan,
                                "RMSE": np.nan,
                                "MAE": np.nan,
                                "Error": str(e)[:140],
                            }
                        )

                res = pd.DataFrame(rows).sort_values(
                    "R2", ascending=False, na_position="last"
                )
                st.dataframe(res, use_container_width=True)

                if best is not None and best[0] in preds:
                    name = best[0]
                    pred = preds[name]
                    fig = go.Figure()
                    fig.add_trace(
                        go.Scatter(
                            x=y_test, y=pred, mode="markers", name="Pred vs actual"
                        )
                    )
                    lo, hi = float(np.min(y_test)), float(np.max(y_test))
                    fig.add_trace(
                        go.Scatter(
                            x=[lo, hi],
                            y=[lo, hi],
                            mode="lines",
                            name="Perfect fit",
                            line=dict(dash="dash"),
                        )
                    )
                    fig.update_layout(
                        template=st.session_state.plotly_template,
                        height=520,
                        title=f"Actual vs Predicted: {name}",
                    )
                    st.plotly_chart(fig, use_container_width=True)

            st.markdown("</div>", unsafe_allow_html=True)


def page_machine_learning():
    require_df()
    df = st.session_state.df

    st.markdown('<div class="h1">🧠 Machine Learning</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p">Train one model using a preprocessing pipeline and export it as a reusable bundle.</div>',
        unsafe_allow_html=True,
    )

    mode = st.radio("Task", ["Classification", "Regression"], horizontal=True)

    if mode == "Classification":
        num = get_numeric_cols(df)
        cat = get_categorical_cols(df)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        if not (num and cat):
            st.info("Need numeric and categorical columns.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        target = st.selectbox("Target", cat, key="ml_clf_target")
        feat_cols = st.multiselect(
            "Features",
            list(df.columns),
            default=num[: min(6, len(num))],
            key="ml_clf_feats",
        )
        model_name = st.selectbox(
            "Model",
            [
                "LogisticRegression",
                "RandomForest",
                "GradientBoosting",
                "SVC",
                "DecisionTree",
                "KNN",
                "GaussianNB",
            ],
            key="ml_clf_model",
        )
        test_size = st.slider("Test split", 0.1, 0.5, 0.2, key="ml_clf_ts")

        if feat_cols and st.button("Train model", key="ml_clf_go"):
            X = df[feat_cols].copy()
            y = df[target].astype(str)
            mask = y.notna()
            X, y = X.loc[mask], y.loc[mask]
            if len(X) < 20:
                st.error("Not enough rows.")
                st.stop()

            pre = make_preprocessor(df, feat_cols)

            model = {
                "LogisticRegression": LogisticRegression(
                    max_iter=2000, random_state=st.session_state.random_state
                ),
                "RandomForest": RandomForestClassifier(
                    n_estimators=300, random_state=st.session_state.random_state
                ),
                "GradientBoosting": GradientBoostingClassifier(
                    random_state=st.session_state.random_state
                ),
                "SVC": SVC(kernel="rbf", random_state=st.session_state.random_state),
                "DecisionTree": DecisionTreeClassifier(
                    random_state=st.session_state.random_state
                ),
                "KNN": KNeighborsClassifier(n_neighbors=5),
                "GaussianNB": GaussianNB(),
            }[model_name]

            pipe = Pipeline(steps=[("pre", pre), ("model", model)])

            le = LabelEncoder()
            y_enc = le.fit_transform(y)

            class_counts = pd.Series(y_enc).value_counts()
            strat = y_enc if class_counts.min() >= 2 else None

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y_enc,
                test_size=test_size,
                random_state=st.session_state.random_state,
                stratify=strat,
            )

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            acc = float(accuracy_score(y_test, pred))

            st.metric("Accuracy", f"{acc:.3f}")
            cm = confusion_matrix(y_test, pred)
            fig = px.imshow(cm, text_auto=True, title="Confusion matrix")
            fig.update_layout(template=st.session_state.plotly_template, height=520)
            st.plotly_chart(fig, use_container_width=True)

            st.text("Classification report:")
            st.code(
                classification_report(y_test, pred, target_names=le.classes_),
                language="text",
            )

            fig_imp = plot_feature_importance_from_pipeline(pipe)
            if fig_imp is not None:
                st.plotly_chart(fig_imp, use_container_width=True)

            bundle = {
                "pipeline": pipe,
                "task": "Classification",
                "target": target,
                "features": feat_cols,
                "label_encoder": le,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
            }
            st.session_state.last_model_bundle = bundle
            st.session_state.last_run_report = build_report(
                task="Classification",
                target=target,
                features=feat_cols,
                metrics={"accuracy": acc},
            )

            st.download_button(
                "⬇️ Download trained model bundle (.joblib)",
                data=to_joblib_bytes(bundle),
                file_name="trained_model_bundle.joblib",
                mime="application/octet-stream",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    else:
        num = get_numeric_cols(df)

        st.markdown('<div class="card">', unsafe_allow_html=True)
        if len(num) < 2:
            st.info("Need at least 2 numeric columns.")
            st.markdown("</div>", unsafe_allow_html=True)
            return

        target = st.selectbox("Target", num, key="ml_reg_target")
        feat_cols = st.multiselect(
            "Features",
            list(df.columns),
            default=[c for c in num if c != target][: min(6, len(num) - 1)],
            key="ml_reg_feats",
        )
        model_name = st.selectbox(
            "Model",
            [
                "LinearRegression",
                "Ridge",
                "Lasso",
                "RandomForest",
                "GradientBoosting",
                "SVR",
                "DecisionTree",
                "KNN",
            ],
            key="ml_reg_model",
        )
        test_size = st.slider("Test split", 0.1, 0.5, 0.2, key="ml_reg_ts")

        if feat_cols and st.button("Train model", key="ml_reg_go"):
            X = df[feat_cols].copy()
            y = pd.to_numeric(df[target], errors="coerce")

            mask = y.notna()
            X, y = X.loc[mask], y.loc[mask]
            if len(X) < 30:
                st.error("Not enough rows.")
                st.stop()

            pre = make_preprocessor(df, feat_cols)

            model = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(alpha=1.0, random_state=st.session_state.random_state),
                "Lasso": Lasso(alpha=0.1, random_state=st.session_state.random_state),
                "RandomForest": RandomForestRegressor(
                    n_estimators=300, random_state=st.session_state.random_state
                ),
                "GradientBoosting": GradientBoostingRegressor(
                    random_state=st.session_state.random_state
                ),
                "SVR": SVR(kernel="rbf"),
                "DecisionTree": DecisionTreeRegressor(
                    random_state=st.session_state.random_state
                ),
                "KNN": KNeighborsRegressor(n_neighbors=5),
            }[model_name]

            pipe = Pipeline(steps=[("pre", pre), ("model", model)])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=st.session_state.random_state
            )

            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)

            r2 = float(r2_score(y_test, pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
            mae = float(mean_absolute_error(y_test, pred))

            c1, c2, c3 = st.columns(3)
            c1.metric("R2", f"{r2:.3f}")
            c2.metric("RMSE", f"{rmse:.3f}")
            c3.metric("MAE", f"{mae:.3f}")

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=y_test, y=pred, mode="markers", name="Pred vs actual")
            )
            lo, hi = float(np.min(y_test)), float(np.max(y_test))
            fig.add_trace(
                go.Scatter(
                    x=[lo, hi],
                    y=[lo, hi],
                    mode="lines",
                    name="Perfect fit",
                    line=dict(dash="dash"),
                )
            )
            fig.update_layout(
                template=st.session_state.plotly_template,
                height=520,
                title="Actual vs Predicted",
            )
            st.plotly_chart(fig, use_container_width=True)

            fig_imp = plot_feature_importance_from_pipeline(pipe)
            if fig_imp is not None:
                st.plotly_chart(fig_imp, use_container_width=True)

            bundle = {
                "pipeline": pipe,
                "task": "Regression",
                "target": target,
                "features": feat_cols,
                "trained_at": datetime.now().isoformat(timespec="seconds"),
            }
            st.session_state.last_model_bundle = bundle
            st.session_state.last_run_report = build_report(
                task="Regression",
                target=target,
                features=feat_cols,
                metrics={"r2": r2, "rmse": rmse, "mae": mae},
            )

            st.download_button(
                "⬇️ Download trained model bundle (.joblib)",
                data=to_joblib_bytes(bundle),
                file_name="trained_model_bundle.joblib",
                mime="application/octet-stream",
                use_container_width=True,
            )

        st.markdown("</div>", unsafe_allow_html=True)

    # Optional SHAP (safe gate; only runs if installed)
    with st.expander("Explain with SHAP (optional)"):
        st.info(
            "This section runs only if the 'shap' package is installed, and works best for tree models."
        )
        try:
            import shap  # optional dependency

            st.success(
                "SHAP is installed. Train a tree model (RandomForest/GradientBoosting) to use it."
            )
        except Exception as e:
            st.warning(f"SHAP not available: {e}")


def page_model_playground():
    st.markdown('<div class="h1">🧪 Model Playground</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p">Use the last trained/loaded model bundle to generate predictions on the current dataset.</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.last_model_bundle is None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.info(
            "No model bundle loaded. Train a model in 🧠 Machine Learning or upload a .joblib bundle in the sidebar."
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    require_df()
    df = st.session_state.df
    bundle = st.session_state.last_model_bundle
    pipe = bundle["pipeline"]
    task = bundle.get("task", "Unknown")
    features = bundle.get("features", [])
    target = bundle.get("target", None)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write(
        {
            "task": task,
            "target": target,
            "features": features,
            "trained_at": bundle.get("trained_at"),
        }
    )

    missing_feats = [c for c in features if c not in df.columns]
    if missing_feats:
        st.error(
            f"Current dataset is missing required feature columns: {missing_feats}"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    X = df[features].copy()

    if st.button("Generate predictions", use_container_width=True):
        try:
            pred = pipe.predict(X)

            out = df.copy()
            if task == "Classification" and "label_encoder" in bundle:
                le = bundle["label_encoder"]
                pred_labels = le.inverse_transform(np.array(pred, dtype=int))
                out["prediction"] = pred_labels
            else:
                out["prediction"] = pred

            st.success("Predictions generated.")
            st.dataframe(out.head(50), use_container_width=True, height=420)

            st.download_button(
                "⬇️ Download predictions (CSV)",
                data=download_csv_bytes(out),
                file_name="predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


def page_time_series():
    require_df()
    df = st.session_state.df
    dt_cols = get_datetime_like_cols(df)
    num_cols = get_numeric_cols(df)

    st.markdown('<div class="h1">🔮 Time Series Forecast</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p">ARIMA + Holt-Winters + optional decomposition (safe-guarded).</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    if not dt_cols or not num_cols:
        st.info("Need at least 1 datetime-like column and 1 numeric column.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    dt = st.selectbox("Datetime column", dt_cols, key="ts_dt")
    ycol = st.selectbox("Value column", num_cols, key="ts_y")

    df2 = df[[dt, ycol]].copy()
    df2[dt] = pd.to_datetime(df2[dt], errors="coerce")
    df2[ycol] = pd.to_numeric(df2[ycol], errors="coerce")
    df2 = df2.dropna().sort_values(dt)

    if len(df2) < 30:
        st.error("Need ~30+ clean rows for forecasting.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    freq = st.selectbox("Assumed frequency", ["D", "W", "M"], index=0, key="ts_freq")
    horizon = st.slider("Forecast horizon", 5, 120, 30, key="ts_h")
    model_type = st.selectbox("Model", ["ARIMA(1,1,1)", "Holt-Winters"], key="ts_model")

    df2 = df2.set_index(dt).asfreq(freq)
    df2[ycol] = df2[ycol].interpolate(limit_direction="both")

    fig = px.line(df2, y=ycol, title="Time series")
    fig.update_layout(template=st.session_state.plotly_template, height=420)
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Run forecast", key="ts_run"):
        try:
            if model_type.startswith("ARIMA"):
                model = ARIMA(df2[ycol], order=(1, 1, 1))
                fit = model.fit()
                fc = fit.forecast(steps=horizon)
            else:
                sp = 7 if freq == "D" else (4 if freq == "W" else 12)
                model = ExponentialSmoothing(
                    df2[ycol], trend="add", seasonal="add", seasonal_periods=sp
                )
                fit = model.fit(optimized=True)
                fc = fit.forecast(horizon)

            fc = pd.Series(
                fc,
                index=pd.date_range(df2.index[-1], periods=horizon + 1, freq=freq)[1:],
            )

            out = pd.DataFrame({"actual": df2[ycol], "forecast": fc})
            fig2 = go.Figure()
            fig2.add_trace(
                go.Scatter(x=out.index, y=out["actual"], mode="lines", name="Actual")
            )
            fig2.add_trace(
                go.Scatter(
                    x=out.index, y=out["forecast"], mode="lines", name="Forecast"
                )
            )
            fig2.update_layout(
                template=st.session_state.plotly_template, height=520, title="Forecast"
            )
            st.plotly_chart(fig2, use_container_width=True)

            with st.expander("Optional: Seasonal decomposition"):
                try:
                    sp = 7 if freq == "D" else (4 if freq == "W" else 12)
                    decomp = seasonal_decompose(df2[ycol], model="additive", period=sp)
                    figd = make_subplots(
                        rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.03
                    )
                    figd.add_trace(
                        go.Scatter(x=df2.index, y=decomp.observed, name="Observed"),
                        row=1,
                        col=1,
                    )
                    figd.add_trace(
                        go.Scatter(x=df2.index, y=decomp.trend, name="Trend"),
                        row=2,
                        col=1,
                    )
                    figd.add_trace(
                        go.Scatter(x=df2.index, y=decomp.seasonal, name="Seasonal"),
                        row=3,
                        col=1,
                    )
                    figd.add_trace(
                        go.Scatter(x=df2.index, y=decomp.resid, name="Residual"),
                        row=4,
                        col=1,
                    )
                    figd.update_layout(
                        template=st.session_state.plotly_template,
                        height=800,
                        title="Decomposition",
                    )
                    st.plotly_chart(figd, use_container_width=True)
                except Exception as e:
                    st.error(f"Decomposition failed: {e}")

        except Exception as e:
            st.error(f"Forecast failed: {e}")

    st.markdown("</div>", unsafe_allow_html=True)


def page_pattern_discovery():
    require_df()
    df = st.session_state.df
    num = get_numeric_cols(df)

    st.markdown('<div class="h1">🎨 Pattern Discovery</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p">Clustering + PCA visualization + optional silhouette.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    if len(num) < 2:
        st.info("Need at least 2 numeric columns for clustering.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    feats = st.multiselect(
        "Features", num, default=num[: min(6, len(num))], key="pd_feats"
    )
    algo = st.selectbox(
        "Algorithm", ["KMeans", "DBSCAN", "Agglomerative"], key="pd_algo"
    )

    if feats and st.button("Run clustering", key="pd_run"):
        X = df[feats].apply(pd.to_numeric, errors="coerce").dropna()
        if len(X) < 20:
            st.error("Not enough clean rows.")
            st.stop()

        scaler = StandardScaler()
        Z = scaler.fit_transform(X)

        if algo == "KMeans":
            k = st.slider("k", 2, 12, 3, key="pd_k")
            model = KMeans(
                n_clusters=k, random_state=st.session_state.random_state, n_init=10
            )
            labels = model.fit_predict(Z)
        elif algo == "DBSCAN":
            eps = st.slider("eps", 0.1, 3.0, 0.5, key="pd_eps")
            ms = st.slider("min_samples", 3, 20, 5, key="pd_ms")
            model = DBSCAN(eps=eps, min_samples=ms)
            labels = model.fit_predict(Z)
        else:
            k = st.slider("clusters", 2, 12, 3, key="pd_ag_k")
            model = AgglomerativeClustering(n_clusters=k)
            labels = model.fit_predict(Z)

        pca = PCA(n_components=2, random_state=st.session_state.random_state)
        P = pca.fit_transform(Z)
        out = pd.DataFrame(
            {"PC1": P[:, 0], "PC2": P[:, 1], "cluster": labels.astype(str)}
        )

        fig = px.scatter(
            out, x="PC1", y="PC2", color="cluster", title="Clusters (PCA 2D)"
        )
        fig.update_layout(template=st.session_state.plotly_template, height=560)
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Cluster quality"):
            uniq = sorted(set(labels))
            if len(uniq) >= 2 and not (len(uniq) == 1 and uniq[0] == -1):
                try:
                    s = silhouette_score(Z, labels)
                    st.metric("Silhouette score", f"{float(s):.3f}")
                except Exception as e:
                    st.error(f"Silhouette failed: {e}")
            else:
                st.info(
                    "Silhouette not meaningful for a single cluster (or all-noise DBSCAN)."
                )

    st.markdown("</div>", unsafe_allow_html=True)


def page_stat_testing():
    require_df()
    df = st.session_state.df
    num = get_numeric_cols(df)

    st.markdown('<div class="h1">📐 Statistical Testing</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p">Confidence interval for mean + z-score outlier scan.</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    if not num:
        st.info("No numeric columns.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    col = st.selectbox("Numeric column", num, key="st_col")
    x = pd.to_numeric(df[col], errors="coerce").dropna()
    if len(x) < 10:
        st.error("Need at least 10 observations.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    conf = st.slider("Confidence", 0.80, 0.99, 0.95, 0.01, key="st_conf")
    alpha = 1 - conf

    mean = float(x.mean())
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    tcrit = float(stats.t.ppf(1 - alpha / 2, df=len(x) - 1))
    lo, hi = mean - tcrit * se, mean + tcrit * se

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean", f"{mean:.4f}")
    c2.metric("CI low", f"{lo:.4f}")
    c3.metric("CI high", f"{hi:.4f}")

    z = (x - x.mean()) / x.std(ddof=1)
    out = int((np.abs(z) > 3).sum())
    st.metric("Z>|3| outliers", f"{out:,}")

    st.markdown("</div>", unsafe_allow_html=True)


def page_data_management():
    require_df()
    df = st.session_state.df

    st.markdown('<div class="h1">💾 Data Management</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p">Export dataset + export last model report (HTML).</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.download_button(
        "⬇️ Download current dataset (CSV)",
        data=download_csv_bytes(df),
        file_name="dataset_current.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.write("Diagnostics:")
    st.write(
        {
            "rows": int(len(df)),
            "cols": int(len(df.columns)),
            "missing": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "memory_mb": round(df_memory_mb(df), 2),
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Last run report")
    report = st.session_state.get("last_run_report")
    if report:
        st.json(report)
        st.download_button(
            "⬇️ Download last report (HTML)",
            data=report_to_html(report),
            file_name="ultra_analytics_report.html",
            mime="text/html",
            use_container_width=True,
        )
    else:
        st.info("Train a model first (🧠 Machine Learning) to generate a report.")
    st.markdown("</div>", unsafe_allow_html=True)


def page_settings():
    st.markdown('<div class="h1">⚙️ Settings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="p">Plotly theme and random seed.</div>', unsafe_allow_html=True
    )

    st.markdown('<div class="card">', unsafe_allow_html=True)
    templates = ["plotly_white", "plotly_dark", "ggplot2", "seaborn", "simple_white"]
    st.session_state.plotly_template = st.selectbox(
        "Plotly template",
        templates,
        index=(
            templates.index(st.session_state.plotly_template)
            if st.session_state.plotly_template in templates
            else 0
        ),
    )
    st.session_state.random_state = st.number_input(
        "Random seed", value=int(st.session_state.random_state), step=1
    )
    st.markdown("</div>", unsafe_allow_html=True)


def page_profile():
    st.markdown('<div class="h1">👤 Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="p">Project metadata.</div>', unsafe_allow_html=True)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.write("Ultra Analytics Pro (Single-File Improved)")
    st.write(
        "EDA • Stats • ML • Forecasting • Clustering • Model export/import • HTML report export"
    )
    st.markdown("</div>", unsafe_allow_html=True)


def add_particles():
    st.markdown(
        """
        <div id="particles-js" style="position: fixed; top: 0; left: 0; width: 100%; height: 100%; z-index: -1; background: #070312;"></div>
        <script src="https://cdn.jsdelivr.net/particles.js/2.0.0/particles.min.js"></script>
        <script>
        particlesJS("particles-js", {
          "particles": {
            "number": { "value": 80, "density": { "enable": true, "value_area": 800 } },
            "color": { "value": "#4f46e5" },
            "shape": { "type": "circle" },
            "opacity": { "value": 0.5, "random": false },
            "size": { "value": 3, "random": true },
            "line_linked": { "enable": true, "distance": 150, "color": "#7c3aed", "opacity": 0.4, "width": 1 },
            "move": { "enable": true, "speed": 2, "direction": "none", "random": false, "straight": false, "out_mode": "out", "bounce": false }
          },
          "interactivity": {
            "detect_on": "canvas",
            "events": { "onhover": { "enable": true, "mode": "grab" }, "onclick": { "enable": true, "mode": "push" }, "resize": true }
          },
          "retina_detect": true
        });
        </script>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Router
# =========================
ROUTES = {
    "🏠 Dashboard Home": page_home,
    "📊 Matplotlib Lab": page_matplotlib_lab,
    "📊 Data Explorer": page_data_explorer,
    "🧰 Data Transform": page_transform,
    "📈 Exploratory Analysis": page_eda,
    "📉 Advanced Statistics": page_adv_stats,
    "🎯 Predictive Modeling": page_predictive_modeling,
    "🧠 Machine Learning": page_machine_learning,
    "🧪 Model Playground": page_model_playground,
    "🔮 Time Series Forecast": page_time_series,
    "🎨 Pattern Discovery": page_pattern_discovery,
    "📐 Statistical Testing": page_stat_testing,
    "💾 Data Management": page_data_management,
    "⚙️ Settings": page_settings,
    "👤 Profile": page_profile,
    "👤 My CV": page_cv,
}
add_particles()
ROUTES.get(st.session_state.page_nav, page_home)()
