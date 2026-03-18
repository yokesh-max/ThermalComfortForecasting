import streamlit as st
import pandas as pd
import os
import json
import joblib
import numpy as np
import io

from dotenv import load_dotenv
from groq import Groq
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

import hvac_models as hm

# ── ENV & GROQ ────────────────────────────────────────────────────
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    st.error("❌ Groq API Key not found in .env file.")
client = Groq(
    api_key     = api_key,
    timeout     = 60.0, # 60 seconds timeout
    max_retries = 3     # retry 3 times
) if api_key else None

# ── PAGE CONFIG ───────────────────────────────────────────────────
st.set_page_config(
    page_title  = "LLM at Scale — Dashboard",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
else:
    st.warning("⚠️ style.css not found.")

# ── SESSION STATE INIT ────────────────────────────────────────────
if "reset_counter" not in st.session_state:
    st.session_state.reset_counter = 0

# Progressive disclosure flags
if "data_preprocessed" not in st.session_state:
    st.session_state.data_preprocessed = False
if "insights_generated" not in st.session_state:
    st.session_state.insights_generated = False

def reset_app_state():
    new_counter = st.session_state.get("reset_counter", 0) + 1
    st.session_state.clear()
    st.session_state.reset_counter = new_counter

# ── CONFIG FROM hvac_models ───────────────────────────────────────
HVAC_FEATURES = hm.FEATURES    # single source of truth — never redefine
HVAC_TARGET   = hm.TARGET
WINDOW        = hm.WINDOW
TRAIN_RATIO   = hm.TRAIN_RATIO

# Correct slider ranges from actual building data
SLIDER_CONFIG = {
    'Cooling_Power':              (0.0,   75000.0, 35000.0, "Cooling Power (W)"),
    'Flowrate':                   (0.0,   2.5,     0.9,     "Flowrate (L/s)"),
    'CHWR-CHWS':                  (0.0,   20.0,    8.0,     "CHWR-CHWS (°C)"),
    'Offcoil_Temperature':        (8.0,   40.0,    22.0,    "Offcoil Temperature (°C)"),
    'Return_air_Co2':             (300.0, 1500.0,  450.0,   "Return Air CO₂ (ppm)"),
    'Return_air_static_pressure': (-15.0, 120.0,   20.0,    "Return Air Pressure (Pa)"),
    'Return_air_RH':              (50.0,  95.0,    65.0,    "Return Air Humidity (%)"),
}

# ── HELPERS ───────────────────────────────────────────────────────

# FIX 9 — cache data loading so sliders don't retrigger file read
@st.cache_data
def load_dataframe(file_bytes, file_extension):
    """Read uploaded file into DataFrame. Cached to avoid reload on rerender."""
    if file_extension == ".csv":
        return pd.read_csv(
            io.BytesIO(file_bytes),
            sep=None, on_bad_lines='warn', engine='python'
        )
    elif file_extension in [".xlsx", ".xls"]:
        return pd.read_excel(io.BytesIO(file_bytes))
    return None


def get_ai_insights(df, latest_reading, model_id="llama-3.3-70b-versatile"):
    """Send data summary and last reading to Groq and return AI analysis + optimal values."""
    if not client:
        return "Groq client not initialised. Check your API key.", None
    
    prompt = f"""
    Analyse the following HVAC sensor dataset and provide 3-4 professional actionable insights for building energy optimisation.
    
    Current Building State (Last Reading):
    {latest_reading.to_dict()}
    
    Physics Constraints to follow:
    1. Cooling Power Increase -> PMV Decrease
    2. Flowrate Increase -> PMV Decrease (more cold water)
    3. Return AI CO2 is a proxy for heat load.
    
    Task:
    1. Provide a textual analysis (3-4 bullet points).
    2. Suggest OPTIMAL numerical values for all 7 features to achieve PMV = 0.0.
    
    IMPORTANT: Provide the optimal values at the end of your response in a JSON block like this:
    OPTIMAL_VALUES: {{"Cooling_Power": X, "Flowrate": Y, "CHWR-CHWS": Z, "Offcoil_Temperature": A, "Return_air_Co2": B, "Return_air_static_pressure": C, "Return_air_RH": D}}
    """
    try:
        completion = client.chat.completions.create(
            model    = model_id,
            messages = [
                {"role": "system", "content":
                    "You are a professional building engineer. Always provide data-driven recommendations that respect thermodynamics."},
                {"role": "user", "content": prompt}
            ],
            temperature = 0.3,
        )
        content = completion.choices[0].message.content
        
        # Parse JSON block
        opt_values = None
        if "OPTIMAL_VALUES:" in content:
            try:
                json_str = content.split("OPTIMAL_VALUES:")[1].strip()
                # Find the first { and last }
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                opt_values = json.loads(json_str[start:end])
            except:
                pass
        
        return content, opt_values
    except Exception as e:
        error_msg = str(e)
        if "Connection error" in error_msg or "timeout" in error_msg.lower():
            return (f"⚠️ **AI Connection Error**: The request timed out or was blocked. "
                    f"Please check your internet connection or try again in a moment. "
                    f"(Details: {error_msg})"), None
        return f"Error during AI analysis: {error_msg}", None


def make_windows_local(X, y, win=WINDOW):
    """Slide a window of `win` rows to create training samples."""
    Xw, yw = [], []
    for i in range(len(X) - win):
        Xw.append(X[i : i+win])
        yw.append(y[i + win])
    return np.array(Xw), np.array(yw)


def perform_hvac_eda(df):
    """Generate visualisations for Exploratory Data Analysis with educational explanations."""
    import seaborn as sns
    
    st.markdown("#### 📈 Exploratory Data Analysis Results")
    st.info("EDA helps us understand the relationships, trends, and quality of our sensor data before training models.")
    
    # 1. Correlation Heatmap
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


def train_selected_model(df):
    """
    Train the LSTM model on uploaded data.
    """
    df = df.sort_values('DateTime').reset_index(drop=True) if 'DateTime' in df.columns else df
    
    # Prepare data
    X_train_w, y_train_w, X_test_w, y_test_w, \
    y_train_raw_w, y_test_raw_w, \
    feat_scaler, pmv_scaler, error_msg = hm.prepare_hvac_data(df)
    
    if error_msg:
        return {'error': error_msg}
    
    try:
        model, history = hm.train_lstm(X_train_w, y_train_w, X_test_w, y_test_w)
        
        # Guard: Only predict if we have test data in this file
        if X_test_w is not None and getattr(X_test_w, "size", 0) > 0:
            best_model_name = "LSTM" # only LSTM supported now
            preds_sc = model.predict(X_test_w, verbose=0)
            preds_raw, mae, rmse, r2 = hm.evaluate_model(y_test_raw_w, preds_sc, pmv_scaler)
        else:
            preds_raw, mae, rmse, r2 = np.array([]), 0.0, 0.0, 0.0
            
        return {
            'model': model,
            'type': 'LSTM',
            'feat_scaler': feat_scaler,
            'pmv_scaler': pmv_scaler,
            'mae_test': mae,
            'rmse_test': rmse,
            'r2_test': r2,
            'history': history,
            'preds_raw': preds_raw,
            'y_test_raw': y_test_raw_w,
            'error': None,
            'has_test_results': len(preds_raw) > 0
        }
    
    except Exception as e:
        return {'error': f'Training error: {str(e)}'}


# Prophet removed from supported models per user request.


# ── SIDEBAR ───────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">Menu Options</div>',
                unsafe_allow_html=True)

    nav_choice = st.selectbox(
        label   = "Navigation",
        options = ["Home", "Dashboard", "Reports"],
        index   = 1, # Default to Dashboard for immediate interactability
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
    
    model_name = st.selectbox(
        label   = "Select Model",
        options = ["Llama 3.1 8B", "Llama 3.3 70B"],
        index   = 1,
        key     = f"model_select_{st.session_state.reset_counter}"
    )
    
    model_id = ("llama-3.1-8b-instant"
                if "8B" in model_name else "llama-3.3-70b-versatile")

    st.markdown("<div style='margin-top: -5px;'></div>", unsafe_allow_html=True)
    st.button("Clear/Reset", key="reset_btn", on_click=reset_app_state)

    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    st.markdown("""
        <div style="text-align:center;color:rgba(255,255,255,0.7);
                    font-size:0.8rem;font-weight:600;margin-bottom:10px;">
            BUILD AND DEPLOYED ON
        </div>
        <div style="display:flex;justify-content:space-between;
                    align-items:center;gap:10px;padding:0 10px;">
            <img src="https://i.im.ge/2026/03/17/eKAOPG.oie-png.png"
                 style="width:45%;height:80px;background:white;
                        border-radius:4px;padding:2px;">
            <img src="https://i.im.ge/2026/03/16/eKItKG.image.png"
                 style="width:45%;height:80px;background:white;
                        border-radius:4px;padding:2px;">
        </div>
    """, unsafe_allow_html=True)


# ── MAIN NAVIGATION ROUTING ──────────────────────────────────────
if nav_choice == "Home":
    st.markdown("""
        <div style="text-align:center; padding: 50px 20px;">
            <h1 style="font-size:3rem; color:#1A237E;">Welcome to Thermal AI</h1>
            <p style="font-size:1.2rem; color:#5F6368; max-width:800px; margin: 0 auto;">
                Optimizing building energy efficiency and occupant comfort using advanced 
                LSTM Neural Networks and Physics-Informed AI Insights.
            </p>
            <div style="display:flex; justify-content:center; gap:30px; margin-top:40px;">
                <div style="background:white; padding:20px; border-radius:15px; box-shadow:0 4px 15px rgba(0,0,0,0.05); width:200px;">
                    <h3 style="color:#1A73E8;">📊 EDA</h3>
                    <p style="font-size:0.9rem;">Deep dive into your sensor patterns.</p>
                </div>
                <div style="background:white; padding:20px; border-radius:15px; box-shadow:0 4px 15px rgba(0,0,0,0.05); width:200px;">
                    <h3 style="color:#66BB6A;">🧠 LSTM</h3>
                    <p style="font-size:0.9rem;">Predictive Thermal Comfort forecasting.</p>
                </div>
                <div style="background:white; padding:20px; border-radius:15px; box-shadow:0 4px 15px rgba(0,0,0,0.05); width:200px;">
                    <h3 style="color:#FFA726;">✨ AI</h3>
                    <p style="font-size:0.9rem;">Data-driven energy optimization.</p>
                </div>
            </div>
            <div style="margin-top:50px;">
                <p>👈 <b>Select "Dashboard" from the sidebar to start analysis.</b></p>
            </div>
        </div>
    """, unsafe_allow_html=True)

elif nav_choice == "Reports":
    st.markdown("<h2 style='text-align:center;'>📈 Model Performance Reports</h2>", unsafe_allow_html=True)
    if st.session_state.get("model_trained", False):
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{st.session_state.model_mae:.4f}")
        col2.metric("RMSE", f"{st.session_state.model_rmse:.4f}")
        col3.metric("R² Score", f"{st.session_state.model_r2:.4f}")
        
        st.info("💡 Complete the Test Evaluation in the Dashboard to see detailed row-by-row comparisons here.")
    else:
        st.warning("⚠️ No model has been trained yet. Please go to the **Dashboard**, upload data, and train the LSTM model to generate a report.")

else: # nav_choice == "Dashboard"
    # ── LOGO & HEADER (Existing) ──────────────────────────────────
    logo_url = "https://i.im.ge/2026/03/16/eKItKG.image.png"
    col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
    with col_l2:
        st.markdown(f"""
            <div style="display:flex;flex-direction:column;align-items:center;
                        width:100%;margin-top:-50px;">
                <img src="{logo_url}" style="width:100px;">
                <div style="text-align:center;color:#1A237E;font-size:1.5rem;
                            font-weight:500;margin-top:-10px;
                            text-shadow:1px 1px 2px rgba(0,0,0,0.1);">
                   Thermal Comfort Forecasting AI Model
                </div>
                <div style="width:800px;height:3px;
                            background:linear-gradient(90deg,transparent,#1A73E8,transparent);
                            margin-top:5px;"></div>
            </div>
            <div style="margin-bottom:15px;"></div>
        """, unsafe_allow_html=True)

    # ── SELECT FUNCTION AREA ──────────────────────────────────────────
    main_func = "Select"
    if func_choice == "Forecasting":
        # User style layout: Text on left, selectbox on right
        sf_col1, sf_col2 = st.columns([1, 2])
        with sf_col1:
            st.markdown(
                "<div style='margin-top:6px; font-weight:600; font-size:1.25rem; color:#1A237E;'>"
                "Select Function</div>", 
                unsafe_allow_html=True
            )
        with sf_col2:
            main_func = st.selectbox(
                label = "",
                options = ["Select", "Thermal Comfort Forecasting"],
                label_visibility = "collapsed",
                key = f"main_func_select_{st.session_state.reset_counter}"
            )

    # ── FILE UPLOADER ─────────────────────────────────────────────────
    uploaded_file = None
    if main_func == "Thermal Comfort Forecasting":
        # Refactored to horizontal layout
        up_col1, up_col2 = st.columns([1, 2])
        with up_col1:
            st.markdown(
            "<div style='margin-top:6px; font-weight:600; font-size:1.25rem; color:#1A237E;'>"
            "Upload Training Data</div>", 
            unsafe_allow_html=True
        )
        with up_col2:
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


    # ── MAIN DATA & MODELLING AREA ────────────────────────────────────
    if uploaded_file is not None:
        # Reset progressive flow if file changes
        file_key = f"file_{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.get("last_file_key") != file_key:
            st.session_state.last_file_key = file_key
            st.session_state.data_preprocessed = False
            st.session_state.insights_generated = False
            st.session_state.show_eda = False
            if "hvac_model" in st.session_state:
                del st.session_state.hvac_model

        file_ext   = os.path.splitext(uploaded_file.name)[1].lower()
        file_bytes = uploaded_file.read()   # read once, cache handles reuse

        # FIX 9 — cached data loading
        try:
            df = load_dataframe(file_bytes, file_ext)
            if df is not None:
                st.success(f"**✅ File Loaded**: {uploaded_file.name}  "
                           f"({df.shape[0]} rows × {df.shape[1]} columns)")
            else:
                st.error("Unsupported file format.")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    sep=None, encoding='ISO-8859-1',
                    on_bad_lines='warn', engine='python'
                )
                st.success(f"✅ Loaded: {uploaded_file.name} (ISO-8859-1 encoding)")
            except Exception as e:
                st.error(f"Encoding error: {e}")
                df = None
        except Exception as e:
            st.error(f"Error reading file: {e}")
            df = None

        if df is not None:

            # ── DATA PREVIEW ─────────────────────────────────────────
            with st.expander("📄 **Detailed Data Preview**", expanded=False):
                st.dataframe(df, use_container_width=True)
                st.caption(f"**Dimensions**: {df.shape[0]} rows × "
                           f"{df.shape[1]} columns")

            # ── DATA EDA ──────────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
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

            # ── PREPROCESSING ─────────────────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            pp_col1, pp_col2 = st.columns([2, 1])
            with pp_col1:
                st.markdown("<h3 style='margin-top:0px;'>⚙️ Data Preprocessing</h3>", unsafe_allow_html=True)
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
                    
                    # Flexible matching logic
                    norm = lambda s: s.lower().replace(' ','').replace('_','').replace('-','')
                    col_map = {norm(c): c for c in df.columns}
                    
                    # Check features
                    missing_features = [f for f in HVAC_FEATURES if norm(f) not in col_map]
                    # Check target
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
                        # Map to actual column names in the df
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

            # ── AI INSIGHTS ───────────────────────────────────────────
            if st.session_state.data_preprocessed:
                st.markdown("<br>", unsafe_allow_html=True)
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
                    with st.spinner(f"Groq ({model_name}) is analysing your data..."):
                        # Pass the latest reading too
                        latest_reading = df.sort_values('DateTime')[HVAC_FEATURES].iloc[-1]
                        insights, opt_values = get_ai_insights(df, latest_reading, model_id=model_id)
                        st.session_state.insights_generated = True
                        st.session_state.ai_insights_text = insights
                        st.session_state.ai_recommendations = opt_values

                if st.session_state.get("insights_generated"):
                    with st.expander("🔍 **Detailed AI Analysis**", expanded=True):
                        st.markdown(f"""
                            {st.session_state.ai_insights_text.split('OPTIMAL_VALUES:')[0]}
                        """, unsafe_allow_html=True)

            # ── HVAC MODELLING ────────────────────────────────────────
            if st.session_state.get("insights_generated"):
                st.markdown("---")
                st.markdown("<h4>🔬 HVAC Thermal Comfort Modelling</h4>", unsafe_allow_html=True)
                
                sm_col1, sm_col2 = st.columns([1, 2])
                with sm_col1:
                    st.markdown(
                        "<div style='margin-top:10px; font-weight:600; font-size:1.1rem; color:#1A237E;'>"
                        "Model Selection</div>", 
                        unsafe_allow_html=True
                    )
                with sm_col2:
                    hvac_model_choice = st.selectbox(
                        label = "Model Selection",
                        options = ["Select Model", "LSTM", "PCDL", "PCEL", "Agentic Forecast"],
                        label_visibility = "collapsed",
                        key = f"hvac_model_{st.session_state.reset_counter}"
                    )

                if hvac_model_choice == "LSTM":
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
                        # Ensure DateTime column is sorted
                        if 'DateTime' in df.columns:
                            df['DateTime'] = pd.to_datetime(df['DateTime'])
                            df = df.sort_values('DateTime').reset_index(drop=True)

                        with st.spinner("⏳ Training LSTM model..."):
                            result = train_selected_model(df)
                            
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
                                
                                # Store real last 12 readings
                                X_raw = df[HVAC_FEATURES].values
                                st.session_state.last_12_raw = X_raw[-WINDOW:] if len(X_raw) >= WINDOW else X_raw
                                
                                if result.get('has_test_results'):
                                    st.success("✅ LSTM model trained and evaluated successfully!")
                                else:
                                    st.success("✅ LSTM model trained successfully! (Evaluation held for Test Section)")

            # ── TEST DATA EVALUATION ─────────────────────────────────────
            if st.session_state.get("model_trained", False):
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
                    test_file = st.file_uploader(
                        label = "Upload Test Data",
                        type  = ["csv", "xlsx", "xls"],
                        key   = f"test_file_{st.session_state.reset_counter}",
                        label_visibility = "collapsed",
                    )
                    
                if test_file is not None:
                    test_ext = os.path.splitext(test_file.name)[1].lower()
                    test_bytes = test_file.read()
                    test_df = load_dataframe(test_bytes, test_ext)
                    
                    if test_df is not None:
                        norm = lambda s: s.lower().replace(' ','').replace('_','').replace('-','')
                        col_map = {norm(c): c for c in test_df.columns if norm(c)}
                        missing = [f for f in HVAC_FEATURES if norm(f) not in col_map]
                        
                        if missing:
                            st.error(f"❌ Test Data Missing features: {', '.join(missing)}")
                        else:
                            actual_features = [col_map[norm(f)] for f in HVAC_FEATURES]
                            test_clean = test_df.dropna(subset=actual_features).copy()
                            
                            if not test_clean.empty:
                                with st.spinner("Calculating PMV predictions..."):
                                    actual_pmv_list = []
                                    forecast_pmv_list = []
                                    formula_pmv_list = []
                                    
                                    # Initial history is the last 12 readings from the training/prev data
                                    current_history = st.session_state.last_12_raw.copy()
                                    model = st.session_state.hvac_model
                                    feat_scaler = st.session_state.hvac_feat_scaler
                                    pmv_scaler = st.session_state.hvac_pmv_scaler
                                    
                                    # Check if PMV target exists in test file
                                    target_col_test = next((c for c in test_df.columns if c.lower() == HVAC_TARGET.lower()), None)
                                    
                                    for idx, row in test_clean.iterrows():
                                        row_dict = {f: row[col_map[norm(f)]] for f in HVAC_FEATURES}
                                        
                                        # 1. ACTUAL (Ground Truth)
                                        if target_col_test:
                                            actual_val = float(row[target_col_test])
                                        else:
                                            # Fallback to formula if PMV missing in test data
                                            actual_val = hm.estimate_pmv_from_sensors(row_dict)
                                        actual_pmv_list.append(actual_val)
                                        
                                        # 2. Formula reference (always calculate for comparison)
                                        formula_val = hm.estimate_pmv_from_sensors(row_dict)
                                        formula_pmv_list.append(formula_val)
                                        
                                        # 3. AI FORECAST
                                        X_history_scaled = feat_scaler.transform(current_history).reshape(1, WINDOW, len(HVAC_FEATURES)).astype('float32')
                                        if hasattr(model, '__call__'):
                                            pred_tensor = model(X_history_scaled, training=False)
                                            pred_scaled = pred_tensor.numpy()[0][0]
                                        else:
                                            pred_scaled = model.predict(X_history_scaled, verbose=0)[0][0]
                                        
                                        forecast_pmv = float(pmv_scaler.inverse_transform([[pred_scaled]])[0][0])
                                        forecast_pmv_list.append(forecast_pmv)
                                        
                                        # 4. UPDATE HISTORY
                                        new_input = np.array([row_dict[f] for f in HVAC_FEATURES], dtype='float32')
                                        current_history = np.vstack([current_history[1:], new_input])
                                    
                                    test_clean['Actual PMV (Data)'] = actual_pmv_list
                                    test_clean['AI Forecast PMV (LSTM)'] = forecast_pmv_list
                                    test_clean['Residual (Difference)'] = test_clean['Actual PMV (Data)'] - test_clean['AI Forecast PMV (LSTM)']
                                    
                                    # Add Human-Friendly Comfort Status
                                    test_clean['Comfort Status (Data)'] = test_clean['Actual PMV (Data)'].apply(hm.get_comfort_descriptor)
                                    test_clean['Comfort Status (AI)'] = test_clean['AI Forecast PMV (LSTM)'].apply(hm.get_comfort_descriptor)

                                    st.success("✅ Test Evaluation Complete!")
                                    
                                    if not target_col_test:
                                        st.warning("⚠️ No 'PMV' column found in test file. Using Physical Formula as ground truth.")
                                    
                                    st.markdown("""
                                        **What the colors mean:**
                                        *   🟦 **Blue/Cool**: Negative PMV (Feeling Cold)
                                        *   ⬜ **White/Light**: PMV near 0 (Comfortable/Ideal)
                                        *   🟥 **Red/Warm**: Positive PMV (Feeling Hot)
                                    """)
                                    display_cols = [
                                        'Actual PMV (Data)', 'Comfort Status (Data)',
                                        'AI Forecast PMV (LSTM)', 'Comfort Status (AI)',
                                        'Residual (Difference)'
                                    ]
                                    st.dataframe(test_clean[display_cols].style.format({
                                        'Actual PMV (Data)': "{:.2f}",
                                        'AI Forecast PMV (LSTM)': "{:.2f}",
                                        'Residual (Difference)': "{:.2f}"
                                    }).background_gradient(cmap="coolwarm", subset=['Actual PMV (Data)', 'AI Forecast PMV (LSTM)', 'Residual (Difference)']), 
                                    use_container_width=True)
                                    
                                    fig, ax = plt.subplots(figsize=(10, 4))
                                    ax.plot(range(len(test_clean)), test_clean['Actual PMV (Data)'].values, label='Actual PMV (Ground Truth)', color='#1A73E8', linewidth=2)
                                    ax.plot(range(len(test_clean)), test_clean['AI Forecast PMV (LSTM)'].values, label='AI Forecast PMV', color='#D81B60', linestyle='--', linewidth=2)
                                    ax.set_title("Actual vs Forecasted PMV (Test Dataset)", fontsize=12, pad=15)
                                    ax.grid(True, linestyle=':', alpha=0.6)
                                    ax.set_xlabel("Time Step (Row)")
                                    ax.set_ylabel("PMV Value")
                                    ax.legend()
                                    # Fix graph scale to avoid formula outliers if needed
                                    if not target_col_test:
                                        ax.set_ylim([-4, 4]) # Standard PMV range + buffer
                                    st.pyplot(fig)

                                    # Display Metrics after PMV predictions (table and plot)
                                    st.markdown("---")
                                    st.markdown("#### 📏 Model Performance Metrics (on Test Data)")
                                    
                                    actuals = np.array(actual_pmv_list)
                                    forecasts = np.array(forecast_pmv_list)
                                    
                                    test_mae = mean_absolute_error(actuals, forecasts)
                                    test_rmse = np.sqrt(mean_squared_error(actuals, forecasts))
                                    test_r2 = r2_score(actuals, forecasts)
                                    
                                    # New metrics
                                    sum_residual = np.sum(actuals - forecasts)
                                    
                                    # MAPE handling zeros (Standard: skip or add small epsilon)
                                    # Using epsilon approach for stability
                                    epsilon = 1e-10
                                    mape = np.mean(np.abs((actuals - forecasts) / (np.abs(actuals) + epsilon))) * 100

                                    m1, m2, m3 = st.columns(3)
                                    m1.metric(label="MAE (Lower = Better)", value=f"{test_mae:.4f}")
                                    m2.metric(label="RMSE (Lower = Better)", value=f"{test_rmse:.4f}")
                                    m3.metric(label="R² Score (Higher = Better)", value=f"{test_r2:.4f}")
                                    
                                    m4, m5 = st.columns(2)
                                    m4.metric(label="Sum of Residuals", value=f"{sum_residual:.4f}", 
                                             help="Sum of (Actual - AI). Positive means underestimating, Negative means overestimating.")
                                    m5.metric(label="MAPE (%)", value=f"{mape:.2f}%",
                                             help="Mean Absolute Percentage Error. Accuracy in percentage terms.")
