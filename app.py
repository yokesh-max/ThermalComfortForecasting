import streamlit as st
import pandas as pd
import os
import numpy as np
import io
import base64

# Suppress TensorFlow internal C++ logging (fixes _audio_microfrontend_op.so not found warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from dotenv import load_dotenv
import agentic as ag
from MODELS import Pcdl as pc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

from MODELS import lstm
from MODELS import pcel

# ── SHARED CONFIGURATION ──────────────────────────────────────────
HVAC_FEATURES = [
    'Cooling_Power',              # primary AC control — 75% importance
    'Flowrate',                   # cold water volume
    'CHWR-CHWS',                  # cooling effort
    'Offcoil_Temperature',        # supply air coldness
    'Return_air_Co2',             # occupancy proxy
    'Return_air_static_pressure', # airflow state
    'Return_air_RH',              # room humidity
]
HVAC_TARGET   = 'PMV'
WINDOW        = 12
TRAIN_RATIO   = 0.7
EPOCHS        = 100
BATCH_SIZE    = 16
PATIENCE      = 15

# ── SHARED PHYSICS & UTILITIES ────────────────────────────────────

def get_comfort_descriptor(pmv):
    """Map numerical PMV to human-readable comfort status."""
    if pmv >= 3.0: return "🔥 Very Hot"
    if pmv >= 1.0: return "☀️ Warm"
    if pmv > -1.0: return "✅ Comfortable"
    if pmv > -3.0: return "🥶 Cool"
    return "🧊 Very Cold"

def make_windows(X, y, win=WINDOW):
    """Slide a window of `win` rows along the data."""
    Xw, yw = [], []
    for i in range(len(X) - win):
        Xw.append(X[i : i+win])
        yw.append(y[i + win])
    return np.array(Xw), np.array(yw)

def prepare_hvac_data(df, train_ratio=TRAIN_RATIO):
    """Full pipeline: map columns → split → scale → window."""
    current_cols = df.columns.tolist()
    mapped_features = []
    norm = lambda s: s.lower().replace(' ','').replace('_','').replace('-','')
    for feat in HVAC_FEATURES:
        if feat in current_cols: mapped_features.append(feat)
        else:
            match = next((c for c in current_cols if norm(c) == norm(feat)), None)
            if match: mapped_features.append(match)
            else: return None, None, None, None, None, None, None, None, f"Missing: {feat}"
    
    target_col = next((c for c in current_cols if c.lower() == HVAC_TARGET.lower()), None)
    if not target_col: return None, None, None, None, None, None, None, None, f"Missing: {HVAC_TARGET}"
    
    data = df[mapped_features + [target_col]].copy()
    data = data.interpolate(method='linear', limit=6, limit_direction='both').dropna().reset_index(drop=True)
    X_raw, y_raw = data[mapped_features].values, data[target_col].values
    split = int(len(X_raw) * train_ratio)
    X_train_raw, X_test_raw = X_raw[:split], X_raw[split:]
    y_train_raw, y_test_raw = y_raw[:split], y_raw[split:]
    
    feat_scaler, pmv_scaler = MinMaxScaler(), MinMaxScaler()
    X_train_sc = feat_scaler.fit_transform(X_train_raw)
    X_test_sc = feat_scaler.transform(X_test_raw) if len(X_test_raw) > 0 else np.empty((0, X_train_raw.shape[1]))
    y_train_sc = pmv_scaler.fit_transform(y_train_raw.reshape(-1,1)).ravel()
    y_test_sc = pmv_scaler.transform(y_test_raw.reshape(-1,1).astype('float32')).ravel() if len(y_test_raw) > 0 else np.array([])
    
    X_train_w, y_train_w = make_windows(X_train_sc, y_train_sc)
    X_test_w, y_test_w = make_windows(X_test_sc, y_test_sc)
    _, y_train_raw_w = make_windows(X_train_raw, y_train_raw)
    _, y_test_raw_w = make_windows(X_test_raw, y_test_raw)
    
    return X_train_w, y_train_w, X_test_w, y_test_w, y_train_raw_w, y_test_raw_w, feat_scaler, pmv_scaler, None

def evaluate_model(y_true_raw, preds_scaled, pmv_scaler):
    """Compute performance metrics."""
    if len(y_true_raw) == 0: return np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0
    preds_raw = pmv_scaler.inverse_transform(preds_scaled.reshape(-1,1)).ravel()
    mae = mean_absolute_error(y_true_raw, preds_raw)
    rmse = np.sqrt(mean_squared_error(y_true_raw, preds_raw))
    r2 = r2_score(y_true_raw, preds_raw)
    bias = np.sum(y_true_raw - preds_raw)
    mape = np.mean(np.abs((y_true_raw - preds_raw) / (np.abs(y_true_raw) + 1e-10))) * 100
    return preds_raw, mae, rmse, r2, bias, mape

def estimate_pmv_from_sensors(row_dict):
    """Placeholder estimator — Fanger formula is not used."""
    return 0.0

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title  = "Thermal Comfort Forecasting",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────
style_path = os.path.join("STYLE", "style.css")
if os.path.exists(style_path):
    with open(style_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning(f"⚠️ {style_path} not found.")

# ── SESSION STATE INIT ────────────────────────────────────────────
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

# Progressive disclosure flags
if "data_preprocessed" not in st.session_state:
    st.session_state.data_preprocessed = False
if "insights_generated" not in st.session_state:
    st.session_state.insights_generated = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "test_data_loaded" not in st.session_state:
    st.session_state.test_data_loaded = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "model_mae" not in st.session_state:
    st.session_state.model_mae = 0.0
if "model_rmse" not in st.session_state:
    st.session_state.model_rmse = 0.0
if "model_r2" not in st.session_state:
    st.session_state.model_r2 = 0.0
if "model_bias" not in st.session_state:
    st.session_state.model_bias = 0.0
if "model_mape" not in st.session_state:
    st.session_state.model_mape = 0.0
if "model_ci_width" not in st.session_state:
    st.session_state.model_ci_width = 0.0
if "main_df" not in st.session_state:
    st.session_state.main_df = None
if "main_df_name" not in st.session_state:
    st.session_state.main_df_name = ""
if "test_df_results" not in st.session_state:
    st.session_state.test_df_results = None
if "test_metrics" not in st.session_state:
    st.session_state.test_metrics = {}
if "agentic_df" not in st.session_state:
    st.session_state.agentic_df = None
if "agentic_df_name" not in st.session_state:
    st.session_state.agentic_df_name = ""

def reset_app_state():
    new_counter = st.session_state.get("reset_counter", 0) + 1
    st.session_state.clear()
    st.session_state.reset_counter = new_counter
    st.session_state.main_df = None
    st.session_state.main_df_name = ""
    st.session_state.test_df_results = None
    st.session_state.test_metrics = {}
    st.session_state.agentic_df = None
    st.session_state.agentic_df_name = ""
    st.session_state.last_main_func = "Select"

def clear_model_results():
    """Clear previous model training and test results when switching models."""
    st.session_state.model_trained = False
    st.session_state.test_data_loaded = False
    st.session_state.test_df_results = None
    st.session_state.test_metrics = {}
    st.session_state.model_mae = 0.0
    st.session_state.model_rmse = 0.0
    st.session_state.model_r2 = 0.0
    st.session_state.pcel_variant_metrics = None

# ── SLIDER CONFIGURATION ──────────────────────────────────────────
SLIDER_CONFIG = {
    'Cooling_Power':              (0.0,   75000.0, 35000.0, "Cooling Power (W)"),
    'Flowrate':                   (0.0,   2.5,     0.9,     "Flowrate (L/s)"),
    'CHWR-CHWS':                  (0.0,   20.0,    8.0,     "CHWR-CHWS (°C)"),
    'Offcoil_Temperature':        (8.0,   40.0,    22.0,    "Offcoil Temperature (°C)"),
    'Return_air_Co2':             (300.0, 1500.0,  450.0,   "Return Air CO₂ (ppm)"),
    'Return_air_static_pressure': (-15.0, 120.0,   20.0,    "Return Air Pressure (Pa)"),
    'Return_air_RH':              (50.0,  95.0,    65.0,    "Return Air Humidity (%)"),
}

@st.cache_data
def get_image_as_base64(file_path):
    """
    Reads a local file and returns its base64 encoded version for st.markdown usage.
    """
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode()
            return f"data:image/png;base64,{encoded_string}"
    except Exception:
        return ""

def perform_hvac_eda(df):
    """Generate visualisations for Exploratory Data Analysis with educational explanations."""
    import seaborn as sns
    
    st.markdown("#### 📈 Exploratory Data Analysis Results")
    st.info("EDA helps us understand the relationships, trends, and quality of our sensor data before training models.")
    
    st.write("**1. Feature Correlation Heatmap**")
    st.caption("How do different sensors relate to each other and PMV? Values close to 1 or -1 indicate strong relationships.")
    numeric_df = df[HVAC_FEATURES + [HVAC_TARGET]].select_dtypes(include=[np.number])
    if not numeric_df.empty:
        fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
        st.pyplot(fig_corr)
    
    # 2. Time Series of key variables
    st.write("**2. Sensor Trends Over Time**")
    st.caption("This chart shows how key features move together. We normalise them (0 to 1) to compare their patterns on the same scale.")
    if 'DateTime' in df.columns:
        fig_ts, ax_ts = plt.subplots(figsize=(12, 5))
        # Plot PMV and top 2 correlated features
        top_cols = [HVAC_TARGET]
        if 'Cooling_Power' in df.columns: top_cols.append('Cooling_Power')
        if 'Offcoil_Temperature' in df.columns: top_cols.append('Offcoil_Temperature')
        
        # Scale for visualization
        df_plot = df.copy()
        for col in top_cols:
            df_plot[col] = (df_plot[col] - df_plot[col].min()) / (df_plot[col].max() - df_plot[col].min())
            ax_ts.plot(df_plot['DateTime'], df_plot[col], label=col)
        
        ax_ts.set_title("Normalised Trends (PMV vs Key Drivers)")
        ax_ts.legend()
        st.pyplot(fig_ts)
    
    # 3. Distributions
    st.write("**3. Target Variable (PMV) Distribution**")
    st.caption("Is your comfort data balanced? A 'Bell Curve' around 0 indicates a good mix of comfortable and slightly uncomfortable points.")
    fig_dist, ax_dist = plt.subplots(figsize=(8, 4))
    sns.histplot(df[HVAC_TARGET], kde=True, color='blue', ax=ax_dist)
    st.pyplot(fig_dist)

    # 4. Outlier Detection
    st.write("**4. Outlier Detection (Box Plots)**")
    st.caption("The 'whiskers' show the normal range. Points outside them (dots) are outliers—potentially faulty sensor readings.")
    
    # We'll plot the top 4 features for outliers
    box_cols = HVAC_FEATURES[:4]
    if box_cols:
        fig_box, axes = plt.subplots(1, len(box_cols), figsize=(15, 5))
        if len(box_cols) == 1: axes = [axes]
        
        for i, col in enumerate(box_cols):
            sns.boxplot(y=df[col], ax=axes[i], color='lightgreen')
            axes[i].set_title(col.replace('_', ' '))
            axes[i].set_ylabel("")
        
        plt.setp(axes, ylim=None)
        plt.tight_layout()
        st.pyplot(fig_box)

def train_selected_model(df, model_type="LSTM"):
    """
    Train the selected model (LSTM or PCDL) on uploaded data.
    """
    df = df.sort_values('DateTime').reset_index(drop=True) if 'DateTime' in df.columns else df
    
    # Prepare data
    X_train_w, y_train_w, X_test_w, y_test_w, \
    y_train_raw_w, y_test_raw_w, \
    feat_scaler, pmv_scaler, error_msg = prepare_hvac_data(df)
    
    if error_msg:
        return {'error': error_msg}
    
    try:
        if model_type == "PCDL":
            model, history = pc.train_pcdl(X_train_w, y_train_w, X_test_w, y_test_w)
        elif model_type == "PCEL":
            # PCEL trains 5 variants internally and returns a wrapper
            model, history, feat_scaler, pmv_scaler, pcel_metrics, variant_metrics = pcel.train_pcel(df)
            # Override metrics with ensemble-specific results
            mae, rmse, r2, bias, mape, ci_width = (
                pcel_metrics['mae'], pcel_metrics['rmse'], pcel_metrics['r2'], 
                pcel_metrics['violations'], pcel_metrics['mape'], 0.0
            )
            preds_raw = pcel_metrics['preds_real']
        else:
            model, history = lstm.train_lstm(X_train_w, y_train_w, X_test_w, y_test_w)
        
        # Guard: Only predict if we have test data in this file (Skip for PCEL as it already evaluated)
        if model_type != "PCEL":
            if X_test_w is not None and getattr(X_test_w, "size", 0) > 0:
                preds_sc = model.predict(X_test_w, verbose=0)
                preds_raw, mae, rmse, r2, bias, mape = evaluate_model(y_test_raw_w, preds_sc, pmv_scaler)
                
                # Confidence Interval via MC Dropout
                try:
                    mc_preds = []
                    for _ in range(30):
                        p = model(X_test_w, training=True)
                        mc_preds.append(p.numpy().flatten())
                    mc_sc = np.array(mc_preds)
                    lower_sc = np.percentile(mc_sc, 2.5, axis=0)
                    upper_sc = np.percentile(mc_sc, 97.5, axis=0)
                    lower_raw = pmv_scaler.inverse_transform(lower_sc.reshape(-1, 1)).flatten()
                    upper_raw = pmv_scaler.inverse_transform(upper_sc.reshape(-1, 1)).flatten()
                    ci_width = float(np.mean(upper_raw - lower_raw))
                except Exception:
                    ci_width = 0.0
            else:
                preds_raw, mae, rmse, r2, bias, mape, ci_width = np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
            
        return {
            'model': model,
            'type': model_type,
            'feat_scaler': feat_scaler,
            'pmv_scaler': pmv_scaler,
            'mae_test': mae,
            'rmse_test': rmse,
            'r2_test': r2,
            'bias_test': bias,
            'mape_test': mape,
            'ci_width_test': ci_width,
            'history': history,
            'preds_raw': preds_raw,
            'y_test_raw': y_test_raw_w,
            'error': None,
            'has_test_results': len(preds_raw) > 0,
            'variant_metrics': variant_metrics if model_type == "PCEL" else None
        }
    
    except Exception as e:
        return {'error': f'Training error: {str(e)}'}

# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Menu Options</div>',
                unsafe_allow_html=True)

    nav_choice = st.selectbox(
        label   = "Navigation",
        options = ["Home", "Reports"],
        index   = 0, # Default to Home
        label_visibility = "collapsed",
        key     = f"home_select_{st.session_state.reset_counter}"
    )
    func_choice = st.selectbox(
        label   = "Select Functionality",
        options = ["Select Functionality", "Forecasting"],
        label_visibility = "collapsed",
        key     = f"func_select_{st.session_state.reset_counter}"
    )

    service_choice = st.selectbox(
        label   = "Select the Service",
        options = ["Select the Service", "Google Cloud Service"],
        label_visibility = "collapsed",
        key     = f"gcp_select_{st.session_state.reset_counter}"
    )
    
    model_choice = st.selectbox(
        label   = "Select Model",
        options = ["Select Model", "Claude Sonnet 4.5", "Claude Sonnet 4.6", "Claude Opus 4.6"],
        label_visibility = "collapsed",
        key     = f"model_select_{st.session_state.reset_counter}"
    )
    
    # Mapping requested labels to latest available Claude 4.x model IDs
    if "Opus" in model_choice:
        model_id = "claude-opus-4-6"
    else:
        model_id = "claude-sonnet-4-6" if "4.6" in model_choice else "claude-sonnet-4-5"

    st.markdown("<div style='margin-top: -5px;'></div>", unsafe_allow_html=True)
    st.button("Clear/Reset", key="reset_btn", on_click=reset_app_state)

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="text-align:center;color:rgba(255,255,255,0.7);
                    font-size:0.8rem;font-weight:600;margin-bottom:10px;">
            BUILD AND DEPLOYED ON
        </div>
        <div style="display:flex;justify-content:space-between;
                    align-items:center;gap:10px;padding:0 10px;">
            <img src="{get_image_as_base64('LOGO/oie_png.png')}"
                 style="width:45%;height:80px;background:white;
                        border-radius:4px;padding:2px;">
            <img src="{get_image_as_base64('LOGO/image.png')}"
                 style="width:45%;height:80px;background:white;
                        border-radius:4px;padding:2px;">
        </div>
    """, unsafe_allow_html=True)

# ── MAIN DATA CONTEXT (Global) ────────────────────────────────────
if "last_main_func" not in st.session_state:
    st.session_state.last_main_func = "Select"

main_func = st.session_state.last_main_func

# ── MAIN NAVIGATION ROUTING ──────────────────────────────────────
# ── LOGO & HEADER ────────────────────────────────────────────────
logo_src = get_image_as_base64("LOGO/image.png")
col_l1, col_l2, col_l3 = st.columns([0.1, 3.8, 0.1])
with col_l2:
    st.markdown(f"""
        <div class="main-header">
            <img src="{logo_src}" class="header-logo">
            <div class="header-title">
               Thermal Comfort Forecasting AI Model
            </div>
            <div class="header-line"></div>
        </div>
        <div class="header-spacer"></div>
    """, unsafe_allow_html=True)

if nav_choice == "Reports":
    st.markdown("<h2 style='text-align:center;'>📈 Model Performance Reports</h2>", unsafe_allow_html=True)
    if st.session_state.get("model_trained", False):
        tm = st.session_state.get("test_metrics", {})
        mae  = tm.get('mae', st.session_state.get('model_mae', 0.0))
        rmse = tm.get('rmse', st.session_state.get('model_rmse', 0.0))
        r2   = tm.get('r2', st.session_state.get('model_r2', 0.0))
        bias = tm.get('residual_sum', st.session_state.get('model_bias', 0.0))
        mape = tm.get('mape', st.session_state.get('model_mape', 0.0))
        ci   = tm.get('ci_width', st.session_state.get('model_ci_width', 0.0))
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MAE", f"{mae:.4f}", help="Mean Absolute Error")
        m2.metric("RMSE", f"{rmse:.4f}", help="Root Mean Squared Error")
        m3.metric("R² Score", f"{r2:.4f}", help="Coefficient of Determination")
        m4.metric("Bias (Sum)", f"{bias:.4f}", help="Total Sum of Errors")
        m5.metric("MAPE (%)", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
        
        ci_val = ci / 2
        df = st.session_state.get("main_df")
        if df is not None:
            test_size = len(df) - int(len(df) * 0.7)
            row_str = f"your **{test_size} rows** of evaluated test data"
        else:
            row_str = "the evaluated test data"
        
        if ci_val < 0.20:
            st.info(f"**95% CI Width (±{ci_val:.4f}) - High Confidence:** The model is highly confident in its predictions because it found strong, consistent thermodynamic patterns in {row_str}.")
        else:
            st.warning(f"**95% CI Width (±{ci_val:.4f}) - Uncertainty Detected:** The model is somewhat uncertain. It is struggling to find clear patterns in {row_str}.\n\n**Solutions to improve confidence:**\n- **Increase Training Data:** Upload a file with more historical records (currently {len(df) if df is not None else 0} total rows).\n- **Check for Anomalies:** Look for extreme sensor spikes or broken sensor readings in your dataset.\n- **Data Consistency:** Ensure HVAC states (like Cooling Power) aren't fluctuating wildly without corresponding changes in the environment.")
        
        st.info("💡 Model reporting data derived from combined training and test sequence evaluation.")
    else:
        st.warning("⚠️ No model has been trained yet. Please go to the **Home**, upload data, and train the LSTM model to generate a report.")

else: # nav_choice == "Home"
    uploaded_file = None
    if func_choice == "Forecasting":
        sf_col1, sf_col2 = st.columns([1, 2])
        with sf_col1:
            st.markdown(
                "<div style='margin-top:6px; font-weight:600; font-size:1.25rem; color:#1A237E;'>"
                "Select Function</div>", 
                unsafe_allow_html=True
            )
        with sf_col2:
            options = ["Select", "Thermal Comfort Forecasting"]
            default_index = options.index(st.session_state.last_main_func) if st.session_state.last_main_func in options else 0
            
            main_func = st.selectbox(
                label = "",
                options = options,
                index = default_index,
                label_visibility = "collapsed",
                key = f"main_func_select_{st.session_state.reset_counter}"
            )
            st.session_state.last_main_func = main_func

        if main_func == "Thermal Comfort Forecasting":
            st.markdown('<div class="dashboard-spacer"></div>', unsafe_allow_html=True)
            uploaded_file = None
            up_col1, up_col2 = st.columns([1, 2])
            with up_col1:
                st.markdown(
                "<div style='margin-top:6px; font-weight:600; font-size:1.25rem; color:#1A237E;'>"
                "Upload Training Data</div>", 
                unsafe_allow_html=True
            )
            with up_col2:
                if st.session_state.get("main_df") is not None:
                    st.success(f"📂 **Current Training File:** {st.session_state.main_df_name}")
                    if st.button("Upload New File", key=f"clear_train_btn_{st.session_state.reset_counter}"):
                        st.session_state.main_df = None
                        st.session_state.main_df_name = ""
                        st.session_state.data_preprocessed = False
                        st.session_state.insights_generated = False
                        st.session_state.show_eda = False
                        st.session_state.model_trained = False
                        if "last_file_key" in st.session_state:
                            del st.session_state.last_file_key
                        st.rerun()
                else:
                    uploaded_file = st.file_uploader(
                        label = "",
                        type  = ["csv", "xlsx", "xls"],
                        key   = f"uploaded_file_{st.session_state.reset_counter}",
                        label_visibility = "collapsed",
                    )
    elif func_choice == "Select Functionality":
        st.info("👈 **Please select a Functionality from the sidebar to begin.**")
    elif service_choice == "Select the Service":
        st.info("👈 **Please select a Service from the sidebar to proceed.**")
    elif main_func == "Select":
        st.info("⬆️ **Please select 'Thermal Comfort Forecasting' to upload your data.**")

    if uploaded_file is not None:
        file_key = f"file_{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_file_key") != file_key:
            st.session_state.last_file_key = file_key
            st.session_state.data_preprocessed = False
            st.session_state.insights_generated = False
            st.session_state.show_eda = False
            st.session_state.main_df = None
            if "hvac_model" in st.session_state:
                del st.session_state.hvac_model

        if st.session_state.main_df is None:
            file_ext   = os.path.splitext(uploaded_file.name)[1].lower()
            file_bytes = uploaded_file.read()
            try:
                df = ag.load_dataframe(file_bytes, file_ext)
                if df is not None:
                    st.session_state.main_df = df
                    st.session_state.main_df_name = uploaded_file.name
                else:
                    st.error("Unsupported file format.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.main_df = None

    df = st.session_state.get("main_df")
    if df is not None:
        st.success(f"**✅ Active Dataset**: {st.session_state.main_df_name}  "
                    f"({df.shape[0]} rows × {df.shape[1]} columns)")

        with st.expander("📄 **Detailed Data Preview**", expanded=False):
            st.dataframe(df, use_container_width=True)
            st.caption(f"**Dimensions**: {df.shape[0]} rows × "
                        f"{df.shape[1]} columns")

        st.markdown("<div style='margin-bottom:30px;'></div>", unsafe_allow_html=True)
        eda_col1, eda_col2 = st.columns([2, 1])
        with eda_col1:
            st.markdown(
                "<div style='margin-top:6px; font-weight:600; font-size:1.25rem; color:#1A237E;'>"
                "Data EDA</div>", 
                unsafe_allow_html=True
            )
        with eda_col2:
            st.markdown('<div class="eda-btn-container">', unsafe_allow_html=True)
            run_eda = st.button("📈 EDA", 
                                key=f"eda_btn_{st.session_state.reset_counter}",
                                use_container_width=True,
                                type="primary")
            st.markdown('</div>', unsafe_allow_html=True)
        
        if run_eda:
            st.session_state.show_eda = not st.session_state.get("show_eda", False)
            
        if st.session_state.get("show_eda", False):
            with st.expander("📊 Exploratory Data Analysis", expanded=True):
                perform_hvac_eda(df)

            st.markdown("<div style='margin-bottom:30px;'></div>", unsafe_allow_html=True)
            pp_col1, pp_col2 = st.columns([2, 1])
            with pp_col1:
                st.markdown("<h3 style='margin-top:0px;'>Data Preprocessing</h3>", unsafe_allow_html=True)
            with pp_col2:
                st.markdown('<div class="preprocess-btn-container">', unsafe_allow_html=True)
                start_pp = st.button("🛠️ Start Preprocessing",
                                     key=f"preprocess_btn_{st.session_state.reset_counter}",
                                     use_container_width=True,
                                     type="primary")
                st.markdown('</div>', unsafe_allow_html=True)
            
            if start_pp:
                with st.status("Running Preprocessing...", expanded=True) as status:
                    st.write("Validating data structure...")
                    norm = lambda s: s.lower().replace(' ','').replace('_','').replace('-','')
                    col_map = {norm(c): c for c in df.columns}
                    missing_features = [f for f in HVAC_FEATURES if norm(f) not in col_map]
                    target_found = norm(HVAC_TARGET) in col_map
                    
                    if missing_features or not target_found:
                        errs = []
                        if missing_features: errs.append(f"Features: {', '.join(missing_features)}")
                        if not target_found: errs.append(f"Target: {HVAC_TARGET}")
                        st.error(f"❌ Missing required columns: {' | '.join(errs)}")
                        status.update(label="Preprocessing: Not Complete ❌",
                                      state="error", expanded=False)
                    else:
                        st.write("Cleaning records...")
                        actual_features = [col_map[norm(f)] for f in HVAC_FEATURES]
                        actual_target = col_map[norm(HVAC_TARGET)]
                        initial_len = len(df)
                        df = df.dropna(subset=actual_features + [actual_target])
                        dropped = initial_len - len(df)
                        if dropped > 0:
                            st.info(f"Cleaned {dropped} rows with missing values.")
                        st.write("Finalising preparation...")
                        status.update(label="Preprocessing: Complete ✅",
                                      state="complete", expanded=False)
                        st.session_state.data_preprocessed = True
                        st.success("**✅ Dataset validated. Ready for modelling.**")
                
            st.markdown("<div style='margin-bottom:30px;'></div>", unsafe_allow_html=True)
            if st.session_state.data_preprocessed:
                ai_col1, ai_col2 = st.columns([2, 1])
                with ai_col1:
                    st.markdown("<h3 style='margin-top:0px;'>🤖 AI Data-Driven Insights</h3>", unsafe_allow_html=True)
                    st.markdown("""
                        **What our AI does with your data:**
                        *   **Finds Inefficiencies**: Detects sensor readings that waste energy.
                        *   **Comfort Optimization**: Calculates target settings to reach **PMV = 0** (Ideal).
                        *   **Physics Reasoning**: Ensures all advice respects the laws of thermodynamics.
                    """)
                with ai_col2:
                    st.markdown('<div class="ai-btn-container">', unsafe_allow_html=True)
                    gen_ai = st.button("✨ Generate AI Analysis",
                                       key=f"ai_btn_{st.session_state.reset_counter}",
                                       use_container_width=True,
                                       type="primary")
                    st.markdown('</div>', unsafe_allow_html=True)

                if gen_ai:
                    with st.status(f" Claude ({model_choice}) is planning your analysis...", expanded=True) as status:
                        st.markdown('<div class="thinking-status">📑 Drafting analysis protocol... <div class="dot-flashing"></div></div>', unsafe_allow_html=True)
                        ag.stream_text_animation("", delay=0.01)
                        plan_code = f"""
**📋 ANALYSIS EXECUTION PLAN**
1. SCAN_HISTORY: Reading latest {len(df)} data points
2. CALCULATE_PHYSICS: Analyzing thermodynamic correlations
3. INFERENCE: Calling {model_choice} (Model ID: {model_id})
4. TOOLS: Code Execution enabled (version: 20250825)
5. GENERATE_INSIGHTS: Extracting building efficiency optimizations
6. OPTIMIZE_VALUES: Determining PMV=0 settings
                        """
                        ag.stream_text_animation(plan_code, delay=0.005, is_code=True, language="markdown")
                        st.markdown('<div class="thinking-status">📊 Calculating thermodynamic correlations... <div class="dot-flashing"></div></div>', unsafe_allow_html=True)
                        ag.stream_text_animation("", delay=0.01)
                        st.markdown('<div class="thinking-status">🧠 Finalizing actionable insights... <div class="dot-flashing"></div></div>', unsafe_allow_html=True)
                        ag.stream_text_animation("", delay=0.01)
                        latest_reading = df.sort_values('DateTime')[HVAC_FEATURES].iloc[-1]
                        insights, opt_values = ag.get_ai_insights(df, latest_reading, model_id=model_id)
                        status.update(label="✅ Analysis Complete!", state="complete", expanded=False)
                    st.session_state.insights_generated = True
                    st.session_state.ai_insights_text = insights
                    st.session_state.ai_recommendations = opt_values
                    with st.expander("🔍 **Detailed AI Analysis**", expanded=True):
                        ag.stream_text_animation(insights.split('OPTIMAL_VALUES:')[0], delay=0.005)

                elif st.session_state.get("insights_generated"):
                    with st.expander("🔍 **Detailed AI Analysis**", expanded=True):
                        st.markdown(st.session_state.ai_insights_text.split('OPTIMAL_VALUES:')[0])

            if st.session_state.get("insights_generated"):
                st.markdown("---")
                st.markdown('<div class="dashboard-spacer"></div>', unsafe_allow_html=True)
                st.markdown("<h4>🔬 HVAC Thermal Comfort Modelling</h4>", unsafe_allow_html=True)
                ms_col1, ms_col2 = st.columns([1, 2])
                with ms_col1:
                    st.markdown(
                        "<div style='margin-top:10px; font-weight:600; font-size:1.1rem; color:#1A237E;'>"
                        "Model Selection</div>", 
                        unsafe_allow_html=True
                    )
                with ms_col2:
                    if "last_hvac_model_choice" not in st.session_state:
                        st.session_state.last_hvac_model_choice = "Select Model"
                    model_options = ["Select Model", "LSTM", "PCEL", "PCDL", "Agentic Forecast"]
                    default_mod_idx = model_options.index(st.session_state.last_hvac_model_choice) if st.session_state.last_hvac_model_choice in model_options else 0
                    hvac_model_choice = st.selectbox(
                        label = "Select Model",
                        options = model_options,
                        index = default_mod_idx,
                        label_visibility = "collapsed",
                        key = "hvac_model_choice_select",
                        on_change = clear_model_results
                    )
                    st.session_state.last_hvac_model_choice = hvac_model_choice

                if hvac_model_choice == "Agentic Forecast":
                    st.markdown("---")
                    st.markdown("##### 🤖 Agentic Forecast (AI Assistant)")
                    ag.display_chatbot()

                if hvac_model_choice in ["LSTM", "PCDL", "PCEL"]:
                    st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)
                    tr_col1, tr_col2 = st.columns([1, 2])
                    with tr_col1:
                        st.markdown(
                            "<div style='margin-top:10px; font-weight:600; font-size:1.1rem; color:#1A237E;'>"
                            "Train Model</div>", 
                            unsafe_allow_html=True
                        )
                    with tr_col2:
                        st.markdown('<div class="train-btn-container">', unsafe_allow_html=True)
                        train_clicked = st.button("Train Model",
                                                 key=f"train_btn_{st.session_state.reset_counter}",
                                                 use_container_width=True,
                                                 type="primary")
                        st.markdown('</div>', unsafe_allow_html=True)

                    if train_clicked:
                        if 'DateTime' in df.columns:
                            df['DateTime'] = pd.to_datetime(df['DateTime'])
                            df = df.sort_values('DateTime').reset_index(drop=True)
                        with st.spinner(f"⏳ Training {hvac_model_choice} model..."):
                            result = train_selected_model(df, hvac_model_choice)
                            if 'error' in result and result['error']:
                                st.error(f"❌ {result['error']}")
                            else:
                                st.session_state.hvac_model       = result['model']
                                st.session_state.hvac_feat_scaler = result['feat_scaler']
                                st.session_state.hvac_pmv_scaler  = result['pmv_scaler']
                                st.session_state.hvac_type        = result['type']
                                st.session_state.model_trained    = True
                                st.session_state.model_mae        = result['mae_test']
                                st.session_state.model_rmse       = result['rmse_test']
                                st.session_state.model_r2         = result['r2_test']
                                st.session_state.model_bias       = result['bias_test']
                                st.session_state.model_mape       = result['mape_test']
                                st.session_state.model_ci_width   = result['ci_width_test']
                                st.session_state.pcel_variant_metrics = result.get('variant_metrics')
                                X_raw = df[HVAC_FEATURES].values
                                st.session_state.last_12_raw = X_raw[-WINDOW:] if len(X_raw) >= WINDOW else X_raw
                                if result.get('has_test_results'):
                                    st.success(f"✅ {st.session_state.hvac_type} model trained successfully!")
                                else:
                                    st.success(f"✅ {st.session_state.hvac_type} model trained successfully! (Evaluation held for Test Section)")

            if st.session_state.get("model_trained", False) and hvac_model_choice != "Agentic Forecast":
                st.markdown("---")
                st.markdown("<h4>📉 Verify Model with Test Data</h4>", unsafe_allow_html=True)
                td_col1, td_col2 = st.columns([1, 2])
                with td_col1:
                    st.markdown(
                        "<div style='margin-top:6px; font-weight:600; font-size:1.1rem; color:#1A237E;'>"
                        "Test Data</div>", 
                        unsafe_allow_html=True
                    )
                with td_col2:
                    test_file = None
                    if st.session_state.get("test_data_loaded"):
                        st.success("📂 **Test Data Loaded & Evaluated**")
                        if st.button("Upload New Test File", key=f"clear_test_btn_{st.session_state.reset_counter}"):
                            st.session_state.test_data_loaded = False
                            st.session_state.test_df_results = None
                            st.session_state.test_metrics = {}
                            st.rerun()
                    else:
                        test_file = st.file_uploader(
                            label = "Upload Test Data",
                            type  = ["csv", "xlsx", "xls"],
                            key   = f"test_file_{st.session_state.reset_counter}",
                            label_visibility = "collapsed",
                        )
                    test_df = None
                    if test_file is not None:
                        test_ext = os.path.splitext(test_file.name)[1].lower()
                        test_bytes = test_file.read()
                        test_df = ag.load_dataframe(test_bytes, test_ext)
                    if test_df is not None:
                        st.session_state.test_data_loaded = True
                        norm = lambda s: s.lower().replace(' ','').replace('_','').replace('-','')
                        col_map = {norm(c): c for c in test_df.columns if norm(c)}
                        actual_features = [col_map[norm(f)] for f in HVAC_FEATURES if norm(f) in col_map]
                        test_clean = test_df.dropna(subset=actual_features).copy()
                        if not test_clean.empty:
                            with st.spinner("Calculating PMV predictions..."):
                                actual_pmv_list = []
                                forecast_pmv_list = []
                                current_history = st.session_state.last_12_raw.copy()
                                model = st.session_state.hvac_model
                                feat_scaler = st.session_state.hvac_feat_scaler
                                pmv_scaler = st.session_state.hvac_pmv_scaler
                                target_col_test = next((c for c in test_df.columns if c.lower() == HVAC_TARGET.lower()), None)
                                for idx, row in test_clean.iterrows():
                                    row_dict = {f: row[col_map[norm(f)]] for f in HVAC_FEATURES}
                                    actual_val = float(row[target_col_test]) if target_col_test else estimate_pmv_from_sensors(row_dict)
                                    actual_pmv_list.append(actual_val)
                                    X_history_scaled = feat_scaler.transform(current_history).reshape(1, WINDOW, len(HVAC_FEATURES)).astype('float32')
                                    pred_scaled = model(X_history_scaled, training=False).numpy()[0][0] if hasattr(model, '__call__') else model.predict(X_history_scaled, verbose=0)[0][0]
                                    forecast_pmv = float(pmv_scaler.inverse_transform([[pred_scaled]])[0][0])
                                    forecast_pmv_list.append(forecast_pmv)
                                    new_input = np.array([row_dict[f] for f in HVAC_FEATURES], dtype='float32')
                                    current_history = np.vstack([current_history[1:], new_input])
                                test_clean['Actual PMV (Data)'] = actual_pmv_list
                                test_clean['AI Forecast PMV'] = forecast_pmv_list
                                test_clean['Residual (Difference)'] = test_clean['Actual PMV (Data)'] - test_clean['AI Forecast PMV']
                                test_clean['Comfort Status (Data)'] = test_clean['Actual PMV (Data)'].apply(get_comfort_descriptor)
                                test_clean['Comfort Status (AI)'] = test_clean['AI Forecast PMV'].apply(get_comfort_descriptor)
                                st.session_state.test_df_results = test_clean
                                actuals = np.array(actual_pmv_list)
                                forecasts = np.array(forecast_pmv_list)
                                _, mae, rmse, r2, bias, mape = evaluate_model(
                                    actuals, 
                                    pmv_scaler.transform(forecasts.reshape(-1,1)).ravel(), 
                                    pmv_scaler
                                )
                                # Confidence Interval via MC Dropout for test data
                                try:
                                    X_test_all = []
                                    curr = st.session_state.last_12_raw.copy()
                                    for idx, row in test_clean.iterrows():
                                        x_w = feat_scaler.transform(curr).reshape(WINDOW, len(HVAC_FEATURES))
                                        X_test_all.append(x_w)
                                        # slide
                                        row_dict = {f: row[col_map[norm(f)]] for f in HVAC_FEATURES}
                                        new_inp = np.array([row_dict[f] for f in HVAC_FEATURES], dtype='float32')
                                        curr = np.vstack([curr[1:], new_inp])
                                    
                                    X_test_all = np.array(X_test_all).astype('float32')
                                    mc_preds = []
                                    for _ in range(30):
                                        p = model(X_test_all, training=True)
                                        mc_preds.append(p.numpy().flatten())
                                    mc_sc = np.array(mc_preds)
                                    lower_sc = np.percentile(mc_sc, 2.5, axis=0)
                                    upper_sc = np.percentile(mc_sc, 97.5, axis=0)
                                    lower_raw = pmv_scaler.inverse_transform(lower_sc.reshape(-1, 1)).flatten()
                                    upper_raw = pmv_scaler.inverse_transform(upper_sc.reshape(-1, 1)).flatten()
                                    ci_width = float(np.mean(upper_raw - lower_raw))
                                except Exception:
                                    ci_width = 0.0
                                
                                st.session_state.test_metrics = {
                                    'mae': mae,
                                    'rmse': rmse,
                                    'r2': r2,
                                    'residual_sum': bias,
                                    'mape': mape,
                                    'ci_width': ci_width,
                                    'no_target': target_col_test is None
                                }

                # Use persisted results if available
                if st.session_state.test_df_results is not None:
                    test_clean = st.session_state.test_df_results
                    metrics = st.session_state.test_metrics
                    
                    st.success("✅ Test Evaluation Results Loaded!")
                    if metrics.get('no_target'):
                        st.warning("⚠️ No 'PMV' column found. Using Physical Formula as ground truth.")

                    st.markdown("""
                        **What the colors mean:**
                        *   🟦 **Blue/Cool**: Feeling Cold | ⬜ **Comfortable** | 🟥 **Warm**: Feeling Hot
                    """)
                    display_cols = ['Actual PMV (Data)', 'Comfort Status (Data)', 'AI Forecast PMV', 'Comfort Status (AI)', 'Residual (Difference)']
                    st.dataframe(test_clean[display_cols].style.format({
                        'Actual PMV (Data)': "{:.2f}",
                        'AI Forecast PMV': "{:.2f}",
                        'Residual (Difference)': "{:.2f}"
                    }).background_gradient(cmap="coolwarm", subset=['Actual PMV (Data)', 'AI Forecast PMV', 'Residual (Difference)']), 
                    use_container_width=True)
                    
                    # --- NEW INTERACTIVE PLOTLY CHART ---
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=test_clean['Actual PMV (Data)'].values,
                        mode='lines',
                        name='Actual PMV (Ground Truth)',
                        line=dict(color='#1A73E8', width=3)
                    ))
                    fig.add_trace(go.Scatter(
                        y=test_clean['AI Forecast PMV'].values,
                        mode='lines',
                        name='AI Forecast PMV',
                        line=dict(color='#D81B60', width=3, dash='dash')
                    ))
                    
                    fig.update_layout(
                        title="📊 Actual vs Forecasted PMV (Interactive Window)",
                        xaxis_title="Time Step (Row)",
                        yaxis_title="PMV Value",
                        height=450,
                        hovermode="x unified",
                        template="plotly_white",
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    # -----------------------------------

                    st.markdown("---")
                    st.markdown("#### 📏 Performance Metrics Consistency Check")
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric(label="MAE", value=f"{metrics['mae']:.4f}", help="Mean Absolute Error (Lower is better)")
                    m2.metric(label="RMSE", value=f"{metrics['rmse']:.4f}", help="Root Mean Squared Error (Lower is better)")
                    m3.metric(label="R² Score", value=f"{metrics['r2']:.4f}", help="Coefficient of Determination (Higher is better)")
                    m4.metric(label="Bias (Sum)", value=f"{metrics['residual_sum']:.4f}", help="Total Sum of Errors (Actual - Predicted)")
                    m5.metric(label="MAPE (%)", value=f"{metrics['mape']:.2f}%", help="Mean Absolute Percentage Error")
                    
                    # PCEL Variant Comparison Table
                    if st.session_state.get('pcel_variant_metrics'):
                        st.markdown("---")
                        st.markdown("#### 📊 Final Comparison Table (PCEL Ensemble vs Variants)")
                        comp_data = []
                        for name, m in st.session_state.pcel_variant_metrics.items():
                            comp_data.append({
                                "Model Version": name,
                                "MAE": m['mae'],
                                "RMSE": m['rmse'],
                                "R²": m['r2'],
                                "MAPE (%)": m['mape'],
                                "Violations": m.get('violations', 0)
                            })
                        comp_df = pd.DataFrame(comp_data)
                        st.dataframe(comp_df.style.format({
                            'MAE': "{:.4f}",
                            'RMSE': "{:.4f}",
                            'R²': "{:.4f}",
                            'MAPE (%)': "{:.2f}%"
                        }), use_container_width=True)
                    
                    ci_val = metrics.get('ci_width', 0.0) / 2
                    df = st.session_state.get("main_df")
                    row_str = f"your **{len(df)} rows** of active data" if df is not None else "the uploaded data"
                    
                    if ci_val < 0.20:
                        st.info(f"**95% CI Width (±{ci_val:.4f}) - High Confidence:** The model is highly confident in its predictions because it found strong, consistent thermodynamic patterns in {row_str}.")
                    else:
                        st.warning(f"**95% CI Width (±{ci_val:.4f}) - Uncertainty Detected:** The model is somewhat uncertain. It is struggling to find clear patterns in {row_str}.\n\n**Solutions to improve confidence:**\n- **Increase Training Data:** Upload a file with more historical records (currently {len(df) if df is not None else 0} rows).\n- **Check for Anomalies:** Look for extreme sensor spikes or broken sensor readings in your dataset.\n- **Data Consistency:** Ensure HVAC states (like Cooling Power) aren't fluctuating wildly without corresponding changes in the environment.")
