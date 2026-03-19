import streamlit as st
import pandas as pd
import base64
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
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

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
    page_title  = "Thermal Comfort Forecasting",
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
            except (json.JSONDecodeError, ValueError) as e:
                st.warning(f"⚠️ Failed to parse AI's optimal values: {e}")
        
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
            preds_sc = model.predict(X_test_w, verbose=0)
            preds_raw, mae, rmse, r2, bias, mape = hm.evaluate_model(y_test_raw_w, preds_sc, pmv_scaler)
        else:
            preds_raw, mae, rmse, r2, bias, mape = np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0
            
        return {
            'model': model,
            'type': 'LSTM',
            'feat_scaler': feat_scaler,
            'pmv_scaler': pmv_scaler,
            'mae_test': mae,
            'rmse_test': rmse,
            'r2_test': r2,
            'bias_test': bias,
            'mape_test': mape,
            'history': history,
            'preds_raw': preds_raw,
            'y_test_raw': y_test_raw_w,
            'error': None,
            'has_test_results': len(preds_raw) > 0
        }
    
    except Exception as e:
        return {'error': f'Training error: {str(e)}'}


# Prophet removed from supported models per user request.

def display_chatbot():
    """
    Renders the Chatbot UI with Groq LLM integration.
    """
    st.markdown("""
        <style>
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .stChatMessage {
            animation: fadeIn 0.5s ease-out forwards;
        }
        .bot-greeting {
            background: linear-gradient(90deg, #E3F2FD, #FFFFFF);
            padding: 15px;
            border-radius: 10px;
            border-left: 5px solid #1A73E8;
            margin-bottom: 20px;
            animation: fadeIn 0.8s ease-out;
        }
        </style>
    """, unsafe_allow_html=True)

    # client is already initialized at the top of the script
    model_name = st.session_state.get("model_select", "Llama 3.3 70B")
    model_id = "llama-3.3-70b-versatile" if "70B" in model_name else "llama-3.1-8b-instant"

    st.markdown('<div class="bot-greeting"><b>🤖 Agentic Forecast Assistant</b>: Hello! I can help you analyze HVAC thermal comfort patterns, optimize energy settings, and explain model metrics.</div>', unsafe_allow_html=True)
    
    # ── SEPARATED AGENTIC DATA ZONE ────────────────────────────────────
    with st.container():
        st.markdown("#### 📁 Agentic Sandbox (Private Data)")
        if st.session_state.agentic_df is None:
            st.markdown("""
                <style>
                @keyframes pulse-border {
                    0% { border-color: rgba(26, 115, 232, 0.4); box-shadow: 0 0 0 0 rgba(26, 115, 232, 0.2); }
                    50% { border-color: rgba(26, 115, 232, 1); box-shadow: 0 0 20px 0 rgba(26, 115, 232, 0.4); }
                    100% { border-color: rgba(26, 115, 232, 0.4); box-shadow: 0 0 0 0 rgba(26, 115, 232, 0.2); }
                }
                .chat-upload-box {
                    background: linear-gradient(135deg, #E3F2FD 0%, #F1F8E9 100%);
                    padding: 30px 20px;
                    border: 3px dashed #1A73E8;
                    border-radius: 15px;
                    text-align: center;
                    animation: pulse-border 3s infinite ease-in-out;
                }
                </style>
                <div class="chat-upload-box">
                    <h5 style="color:#1A237E; margin:0; font-weight:700;">DRAG AND DROP FOR AGENTIC SANDBOX</h5>
                    <p style="color:#5F6368; font-size: 0.9rem;">(This file will not replace your main Dashboard data)</p>
                </div>
            """, unsafe_allow_html=True)
            
            bot_file = st.file_uploader(
                label = "Agentic Sandbox Uploader",
                type  = ["csv", "xlsx", "xls"],
                key   = "bot_file_upload_sandbox",
                label_visibility = "collapsed"
            )
            if bot_file:
                # Handle the upload immediately
                file_ext = os.path.splitext(bot_file.name)[1].lower()
                df = load_dataframe(bot_file.read(), file_ext)
                if df is not None:
                    st.session_state.agentic_df = df
                    st.session_state.agentic_df_name = bot_file.name
                    # Notify the conversation
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": f"✅ **Sandbox Initialized!** I've loaded `{bot_file.name}` into my private agentic sandbox. This data is isolated from your main Dashboard workspace."
                    })
                    st.rerun()
        else:
            st.success(f"**✅ Sandbox Active**: {st.session_state.agentic_df_name}")
            if st.button("🗑️ Clear Private Data", key="clear_agentic_data"):
                st.session_state.agentic_df = None
                st.session_state.agentic_df_name = ""
                st.session_state.messages = [] # Clear chat history as requested
                st.rerun()

    # ── CONVERSATION TRANSCRIPT ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### 💬 Conversation Transcript")

    # Context gathering (CRITICAL for LLM logic below)
    # Prioritize Agentic Sandbox data if available
    is_using_sandbox = st.session_state.get("agentic_df") is not None
    active_df = st.session_state.agentic_df if is_using_sandbox else st.session_state.get("main_df")
    has_df = active_df is not None
    
    train_status = st.session_state.get("data_preprocessed", False)
    model_status = st.session_state.get("model_trained", False)
    test_status = st.session_state.get("test_data_loaded", False)

    # Initialize messages if empty (Greeting)
    if not st.session_state.messages:
        initial_msg = "Hello! I'm your **Agentic Forecast Assistant**. Please use the **Data Manager** above to upload your file, and I'll be ready to help!"
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("How can I help you today?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Logic to handle data-specific questions
        context_prompt = ""
        if not has_df:
            context_prompt = "User has not uploaded any HVAC sensor data yet."
        else:
            # Generate a summary of the active data
            row_count = len(active_df)
            cols = ", ".join(active_df.columns.tolist())
            numeric_cols = active_df.select_dtypes(include=['number']).columns.tolist()
            
            # Basic stats for numeric columns
            stats_str = ""
            if numeric_cols:
                stats = active_df[numeric_cols].describe().loc[['mean', 'min', 'max']]
                stats_str = f"Statistics for numeric columns:\n{stats.to_string()}\n"
            
            # Get the last few rows for real-time context
            last_rows_str = ""
            try:
                # Use only HVAC features if possible
                norm = lambda s: s.lower().replace(' ','').replace('_','').replace('-','')
                col_map = {norm(c): c for c in active_df.columns}
                available_features = [col_map[norm(f)] for f in HVAC_FEATURES if norm(f) in col_map]
                if available_features:
                    last_rows = active_df[available_features].tail(5)
                    last_rows_str = f"Last 5 data points (Features only):\n{last_rows.to_string()}\n"
            except:
                pass

            # Predict the next value if model is available
            prediction_str = ""
            
            # AUTO-TRAIN: If no model exists but data does, train it silently for the chatbot
            if not model_status and has_df:
                try:
                    with st.status("🤖 Assistant is initializing predictive engine...", expanded=False):
                        result = train_selected_model(active_df)
                        if 'error' not in result or not result['error']:
                            st.session_state.hvac_model       = result['model']
                            st.session_state.hvac_feat_scaler = result['feat_scaler']
                            st.session_state.hvac_pmv_scaler  = result['pmv_scaler']
                            st.session_state.model_trained    = True
                            # Refresh model_status for the logic below
                            model_status = True
                except:
                    pass

            if model_status and st.session_state.get("hvac_model") is not None:
                try:
                    # Prepare the window from active_df
                    norm = lambda s: s.lower().replace(' ','').replace('_','').replace('-','')
                    col_map = {norm(c): c for c in active_df.columns}
                    actual_features = [col_map[norm(f)] for f in HVAC_FEATURES if norm(f) in col_map]
                    
                    if len(actual_features) == len(HVAC_FEATURES):
                        X_raw = active_df[actual_features].values
                        if len(X_raw) >= WINDOW:
                            feat_scaler = st.session_state.hvac_feat_scaler
                            pmv_scaler = st.session_state.hvac_pmv_scaler
                            model = st.session_state.hvac_model
                            
                            # 1-step prediction (5 mins)
                            window_raw = X_raw[-WINDOW:].copy()
                            window_sc = feat_scaler.transform(window_raw).reshape(1, WINDOW, len(HVAC_FEATURES)).astype('float32')
                            pred_sc = model(window_sc, training=False).numpy()[0][0] if hasattr(model, '__call__') else model.predict(window_sc, verbose=0)[0][0]
                            pred_5m = float(pmv_scaler.inverse_transform([[pred_sc]])[0][0])
                            
                            # 12-step recursive prediction (1 hour)
                            # Note: This assumes features stay constant, which is a simplification
                            curr_window = window_raw.copy()
                            preds_1h = []
                            for _ in range(12):
                                win_sc = feat_scaler.transform(curr_window).reshape(1, WINDOW, len(HVAC_FEATURES)).astype('float32')
                                p_sc = model(win_sc, training=False).numpy()[0][0] if hasattr(model, '__call__') else model.predict(win_sc, verbose=0)[0][0]
                                p_raw = float(pmv_scaler.inverse_transform([[p_sc]])[0][0])
                                preds_1h.append(p_raw)
                                # Slide window (re-using the last row's features but could be improved)
                                next_row = curr_window[-1].copy() 
                                curr_window = np.vstack([curr_window[1:], next_row])
                            
                            avg_1h = np.mean(preds_1h)
                            prediction_str = (
                                f"NEXT PMV PREDICTION (5 mins): {pred_5m:.4f}\n"
                                f"FORECASTED PMV (1 hour average): {avg_1h:.4f}\n"
                                f"FORECASTED PMV TREND (12 steps): {', '.join([f'{p:.2f}' for p in preds_1h])}\n"
                            )
                except Exception as e:
                    prediction_str = f"Could not generate live prediction: {e}\n"

            context_prompt = (
                f"Active Dataset: {st.session_state.get('agentic_df_name', 'Main Dataset')}\n"
                f"Total Rows: {row_count}\n"
                f"Columns: {cols}\n"
                f"{stats_str}\n"
                f"{last_rows_str}\n"
                f"{prediction_str}"
            )

            # Add model performance if trained and using main data
            if not is_using_sandbox and model_status:
                mae = st.session_state.get("model_mae", 0)
                rmse = st.session_state.get("model_rmse", 0)
                r2 = st.session_state.get("model_r2", 0)
                context_prompt += f"\nModel Performance (LSTM): MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}"

        with st.chat_message("assistant"):
            # Use the global 'client' variable
            if not client:
                st.error("❌ Groq client not initialized. Please check your API key.")
            else:
                try:
                    full_system_prompt = (
                        "You are a professional HVAC AI Forecast Assistant. You have access to real-time sensor data and your own predictive LSTM model. "
                        "Your primary goal is to provide direct, data-driven answers and forecasts. "
                        f"Current Data Context:\n{context_prompt}\n\n"
                        "Guidelines:\n"
                        "1. NEVER tell the user to go to the Dashboard to train a model. You have already handled the modeling in the background if needed. "
                        "2. ALWAYS use the 'NEXT PMV PREDICTION' and 'FORECASTED PMV' provided in the context to answer forecasting questions. "
                        "3. When you provide a prediction, EXPLICITLY state that you are using your 'LSTM Neural Network' to generate the value. "
                        "4. Analyze the 'Last 5 data points' to identify recent trends and include this in your reasoning. "
                        "5. If a prediction is missing from the context (e.g., due to an error), explain the data limitation but still attempt a physics-based estimate using the statistics provided."
                    )
                        
                    completion = client.chat.completions.create(
                        model=model_id,
                        messages=[
                            {"role": "system", "content": full_system_prompt},
                            *st.session_state.messages
                        ],
                        temperature=0.5,
                    )
                    response = completion.choices[0].message.content
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Error: {e}")


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
    st.markdown(f"""
        <div style="text-align:center;color:rgba(255,255,255,0.7);
                    font-size:0.8rem;font-weight:600;margin-bottom:10px;">
            BUILD AND DEPLOYED ON
        </div>
        <div style="display:flex;justify-content:space-between;
                    align-items:center;gap:10px;padding:0 10px;">
            <img src="{get_image_as_base64('C:/Users/yokes/Downloads/oie_png.png')}"
                 style="width:45%;height:80px;background:white;
                        border-radius:4px;padding:2px;">
            <img src="{get_image_as_base64('C:/Users/yokes/Downloads/image.png')}"
                 style="width:45%;height:80px;background:white;
                        border-radius:4px;padding:2px;">
        </div>
    """, unsafe_allow_html=True)


# ── MAIN DATA CONTEXT (Global) ────────────────────────────────────
# This ensures that selections made in the Dashboard persist across all tabs
if "last_main_func" not in st.session_state:
    st.session_state.last_main_func = "Select"

main_func = st.session_state.last_main_func

# ── MAIN NAVIGATION ROUTING ──────────────────────────────────────
# ── LOGO & HEADER ────────────────────────────────────────────────
logo_src = get_image_as_base64("C:/Users/yokes/Downloads/image.png")
col_l1, col_l2, col_l3 = st.columns([1, 2, 1])
with col_l2:
    st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:center;
                    width:100%;margin-top:-50px;">
            <img src="{logo_src}" style="width:100px;">
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

# Chatbot moved inside Dashboard -> Model Selection -> Agentic Forecast
if nav_choice == "Reports":
    st.markdown("<h2 style='text-align:center;'>📈 Model Performance Reports</h2>", unsafe_allow_html=True)
    if st.session_state.get("model_trained", False):
        # Prefer test metrics if available, otherwise show training metrics
        tm = st.session_state.get("test_metrics", {})
        
        # Consistent data mapping
        mae  = tm.get('mae', st.session_state.get('model_mae', 0.0))
        rmse = tm.get('rmse', st.session_state.get('model_rmse', 0.0))
        r2   = tm.get('r2', st.session_state.get('model_r2', 0.0))
        bias = tm.get('residual_sum', st.session_state.get('model_bias', 0.0))
        mape = tm.get('mape', st.session_state.get('model_mape', 0.0))
        
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("MAE", f"{mae:.4f}", help="Mean Absolute Error (Lower is better)")
        m2.metric("RMSE", f"{rmse:.4f}", help="Root Mean Squared Error (Lower is better)")
        m3.metric("R² Score", f"{r2:.4f}", help="Coefficient of Determination (Higher is better)")
        m4.metric("Bias (Sum)", f"{bias:.4f}", help="Total Sum of Errors (Actual - Predicted)")
        m5.metric("MAPE (%)", f"{mape:.2f}%", help="Mean Absolute Percentage Error")
        
        st.info("💡 Model reporting data derived from combined training and test sequence evaluation.")
    else:
        st.warning("⚠️ No model has been trained yet. Please go to the **Home**, upload data, and train the LSTM model to generate a report.")

else: # nav_choice == "Home"
    # ── SELECT FUNCTION AREA ──────────────────────────────────────────
    # main_func is already initialized globally at the top
    uploaded_file = None
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
            # Persist main_func in session state manually to survive tab switching
            st.session_state.last_main_func = main_func

        if main_func == "Thermal Comfort Forecasting":
            # ── FILE UPLOADER ─────────────────────────────────────────────────
            uploaded_file = None
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
            st.session_state.main_df = None
            if "hvac_model" in st.session_state:
                del st.session_state.hvac_model

        if st.session_state.main_df is None:
            file_ext   = os.path.splitext(uploaded_file.name)[1].lower()
            file_bytes = uploaded_file.read()   # read once, cache handles reuse

            # FIX 9 — cached data loading
            try:
                df = load_dataframe(file_bytes, file_ext)
                if df is not None:
                    st.session_state.main_df = df
                    st.session_state.main_df_name = uploaded_file.name
                else:
                    st.error("Unsupported file format.")
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.session_state.main_df = None

    # Use persisted df if available
    df = st.session_state.get("main_df")
    if df is not None:
        st.success(f"**✅ Active Dataset**: {st.session_state.main_df_name}  "
                    f"({df.shape[0]} rows × {df.shape[1]} columns)")

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

            # ── HVAC MODELLING AREA ───────────────────────────────────
            if st.session_state.get("insights_generated"):
                st.markdown("---")
                st.markdown("<h4>🔬 HVAC Thermal Comfort Modelling</h4>", unsafe_allow_html=True)

                # Model Selection - Two-column layout: Label on left, Box on right
                ms_col1, ms_col2 = st.columns([1, 2])
                with ms_col1:
                    st.markdown(
                        "<div style='margin-top:10px; font-weight:600; font-size:1.1rem; color:#1A237E;'>"
                        "Model Selection</div>", 
                        unsafe_allow_html=True
                    )
                with ms_col2:
                    hvac_model_choice = st.selectbox(
                        label = "Select Model",
                        options = ["Select Model", "LSTM", "PCEL", "PCDL", "Agentic Forecast"],
                        label_visibility = "collapsed",
                        key = f"hvac_model_{st.session_state.reset_counter}"
                    )

                if hvac_model_choice == "Agentic Forecast":
                    st.markdown("---")
                    st.markdown("#### 🤖 Agentic Forecast (AI Assistant)")
                    display_chatbot()

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
                                st.session_state.model_bias       = result['bias_test']
                                st.session_state.model_mape       = result['mape_test']
                                
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
                    
                    test_df = None
                    if test_file is not None:
                        test_ext = os.path.splitext(test_file.name)[1].lower()
                        test_bytes = test_file.read()
                        test_df = load_dataframe(test_bytes, test_ext)

                    if test_df is not None:
                        st.session_state.test_data_loaded = True
                        # ... mapping and cleaning ...
                        norm = lambda s: s.lower().replace(' ','').replace('_','').replace('-','')
                        col_map = {norm(c): c for c in test_df.columns if norm(c)}
                        actual_features = [col_map[norm(f)] for f in HVAC_FEATURES if norm(f) in col_map]
                        test_clean = test_df.dropna(subset=actual_features).copy()
                        
                        if not test_clean.empty:
                            with st.spinner("Calculating PMV predictions..."):
                                # ... (existing calculation logic) ...
                                actual_pmv_list = []
                                forecast_pmv_list = []
                                current_history = st.session_state.last_12_raw.copy()
                                model = st.session_state.hvac_model
                                feat_scaler = st.session_state.hvac_feat_scaler
                                pmv_scaler = st.session_state.hvac_pmv_scaler
                                target_col_test = next((c for c in test_df.columns if c.lower() == HVAC_TARGET.lower()), None)

                                for idx, row in test_clean.iterrows():
                                    row_dict = {f: row[col_map[norm(f)]] for f in HVAC_FEATURES}
                                    actual_val = float(row[target_col_test]) if target_col_test else hm.estimate_pmv_from_sensors(row_dict)
                                    actual_pmv_list.append(actual_val)

                                    X_history_scaled = feat_scaler.transform(current_history).reshape(1, WINDOW, len(HVAC_FEATURES)).astype('float32')
                                    pred_scaled = model(X_history_scaled, training=False).numpy()[0][0] if hasattr(model, '__call__') else model.predict(X_history_scaled, verbose=0)[0][0]
                                    forecast_pmv = float(pmv_scaler.inverse_transform([[pred_scaled]])[0][0])
                                    forecast_pmv_list.append(forecast_pmv)
                                    
                                    new_input = np.array([row_dict[f] for f in HVAC_FEATURES], dtype='float32')
                                    current_history = np.vstack([current_history[1:], new_input])

                                test_clean['Actual PMV (Data)'] = actual_pmv_list
                                test_clean['AI Forecast PMV (LSTM)'] = forecast_pmv_list
                                test_clean['Residual (Difference)'] = test_clean['Actual PMV (Data)'] - test_clean['AI Forecast PMV (LSTM)']
                                test_clean['Comfort Status (Data)'] = test_clean['Actual PMV (Data)'].apply(hm.get_comfort_descriptor)
                                test_clean['Comfort Status (AI)'] = test_clean['AI Forecast PMV (LSTM)'].apply(hm.get_comfort_descriptor)
                                
                                st.session_state.test_df_results = test_clean
                                
                                # Metrics calculation
                                actuals = np.array(actual_pmv_list)
                                forecasts = np.array(forecast_pmv_list)
                                
                                # Designate hvac_models as the single source of truth for metrics
                                # We pass a dummy scaled array because evaluate_model expects scaled predictions 
                                # to inverse scale them, but since we already have forecast_pmv_list (raw),
                                # we can just use the internal calculation if we modify it to accept raw.
                                # However, to follow the current hm.evaluate_model signature:
                                _, mae, rmse, r2, bias, mape = hm.evaluate_model(
                                    actuals, 
                                    pmv_scaler.transform(forecasts.reshape(-1,1)).ravel(), 
                                    pmv_scaler
                                )
                                
                                st.session_state.test_metrics = {
                                    'mae': mae,
                                    'rmse': rmse,
                                    'r2': r2,
                                    'residual_sum': bias,
                                    'mape': mape,
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
                    display_cols = ['Actual PMV (Data)', 'Comfort Status (Data)', 'AI Forecast PMV (LSTM)', 'Comfort Status (AI)', 'Residual (Difference)']
                    st.dataframe(test_clean[display_cols].style.format({
                        'Actual PMV (Data)': "{:.2f}",
                        'AI Forecast PMV (LSTM)': "{:.2f}",
                        'Residual (Difference)': "{:.2f}"
                    }).background_gradient(cmap="coolwarm", subset=['Actual PMV (Data)', 'AI Forecast PMV (LSTM)', 'Residual (Difference)']), 
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
                        y=test_clean['AI Forecast PMV (LSTM)'].values,
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
