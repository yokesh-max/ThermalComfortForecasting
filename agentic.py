import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import io
import time
from anthropic import Anthropic
from dotenv import load_dotenv
import hvac_models as hm
import lstm

# ── ENV & CLAUDE ──────────────────────────────────────────────────
load_dotenv()
claude_api_key = os.getenv("claude_api_key")
claude_client = Anthropic(api_key=claude_api_key) if claude_api_key else None

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

def stream_text_animation(text, delay=0.01, is_code=False, language="python"):
    """Simulates a creative typewriter/streaming effect for text or code."""
    placeholder = st.empty()
    full_text = ""
    
    # Adding a glowing container for the stream
    prefix = '<div class="streaming-container">'
    suffix = '</div>'
    
    for char in text:
        full_text += char
        if is_code:
            placeholder.code(full_text, language=language)
        else:
            placeholder.markdown(full_text)
        time.sleep(delay)
    return full_text

def get_ai_insights(df, latest_reading, model_id="claude-sonnet-4-5"):
    """Send data summary and last reading to Claude and return AI analysis + optimal values."""
    if not claude_client:
        return "Claude client not initialised. Check your API key.", None
    
    prompt = f"""
    Analyse the following HVAC sensor dataset and provide an elaborate, detailed, and comprehensive professional analysis for building energy optimisation.
    
    Current Building State (Last Reading):
    {latest_reading.to_dict()}
    
    Physics Constraints to follow:
    1. Cooling Power Increase -> PMV Decrease
    2. Flowrate Increase -> PMV Decrease (more cold water)
    3. Return AI CO2 is a proxy for heat load.
    
    Task:
    1. Provide a textual analysis (exactly 5-6 concise but informative bullet points explaining the reasonings).
    2. Suggest OPTIMAL numerical values for all 7 features to achieve PMV = 0.0.
    
    IMPORTANT: Provide the optimal values at the end of your response in a JSON block like this:
    OPTIMAL_VALUES: {{"Cooling_Power": X, "Flowrate": Y, "CHWR-CHWS": Z, "Offcoil_Temperature": A, "Return_air_Co2": B, "Return_air_static_pressure": C, "Return_air_RH": D}}
    """
    try:
        message = claude_client.messages.create(
            model=model_id,
            max_tokens=10000,
            system="""You are a professional building engineer. Always provide data-driven recommendations that respect thermodynamics.
            FORMATTING RULE: Use only level 5 or 6 headers (##### or ######) for section titles. Never use large headers (#, ##, or ###).""",
            messages=[
                {"role": "user", "content": prompt}
            ],
            tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
            temperature=0.3,
        )
        # Extract text from message content blocks robustly
        content = "".join([block.text for block in message.content if hasattr(block, 'text')])
        
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



def display_chatbot():
    """
    Renders the Chatbot UI with Anthropic Claude integration.
    """
    st.markdown("""
        <style>
        @keyframes slideIn {
            from { opacity: 0; transform: translateX(-20px); filter: blur(5px); }
            to { opacity: 1; transform: translateX(0); filter: blur(0); }
        }
        @keyframes glowPulse {
            0% { box-shadow: 0 0 5px rgba(26, 115, 232, 0.2); }
            50% { box-shadow: 0 0 15px rgba(26, 115, 232, 0.5); }
            100% { box-shadow: 0 0 5px rgba(26, 115, 232, 0.2); }
        }
        @keyframes typing {
            from { width: 0 }
            to { width: 100% }
        }
        .stChatMessage {
            animation: slideIn 0.6s cubic-bezier(0.2, 0.8, 0.2, 1) forwards;
            border-radius: 15px !important;
            margin-bottom: 15px !important;
            transition: all 0.3s ease;
        }
        .stChatMessage:hover {
            transform: scale(1.01);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .bot-greeting {
            background: linear-gradient(135deg, #f0f4ff, #ffffff);
            padding: 20px;
            border-radius: 12px;
            border-left: 6px solid #1A73E8;
            margin-bottom: 25px;
            animation: slideIn 0.8s ease-out;
            box-shadow: 0 2px 10px rgba(26, 115, 232, 0.1);
        }
        .streaming-text {
            color: #1A237E;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-size: 1.05rem;
            line-height: 1.6;
            display: inline-block;
            padding-right: 5px;
        }
        .thinking-status {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #1A73E8;
            font-weight: 600;
            margin: 10px 0;
        }
        .dot-flashing {
            position: relative;
            width: 10px;
            height: 10px;
            border-radius: 5px;
            background-color: #1A73E8;
            color: #1A73E8;
            animation: dotFlashing 1s infinite linear alternate;
            animation-delay: .5s;
        }
        .dot-flashing::before, .dot-flashing::after {
            content: '';
            display: inline-block;
            position: absolute;
            top: 0;
        }
        .dot-flashing::before {
            left: -15px;
            width: 10px;
            height: 10px;
            border-radius: 5px;
            background-color: #1A73E8;
            color: #1A73E8;
            animation: dotFlashing 1s infinite linear alternate;
            animation-delay: 0s;
        }
        .dot-flashing::after {
            left: 15px;
            width: 10px;
            height: 10px;
            border-radius: 5px;
            background-color: #1A73E8;
            color: #1A73E8;
            animation: dotFlashing 1s infinite linear alternate;
            animation-delay: 1s;
        }
        @keyframes dotFlashing {
            0% { background-color: #1A73E8; }
            50%, 100% { background-color: #ebe6ff; }
        }
        /* Custom header size limit for AI responses */
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            font-size: 1.2rem !important;
            color: #1A237E !important;
            margin-top: 15px !important;
            margin-bottom: 10px !important;
        }
        .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
            font-size: 1.1rem !important;
            color: #1A237E !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Anthropic client is initialized at the top of the script
    model_name = st.session_state.get("model_select", "Claude Sonnet 4.5")
    if "Opus" in model_name:
        model_id = "claude-opus-4-6"
    else:
        model_id = "claude-sonnet-4-6" if "4.6" in model_name else "claude-sonnet-4-5"

    st.markdown('<div class="bot-greeting"><b>🤖 Agentic Forecast Assistant</b>: Hello! I can help you analyze HVAC thermal comfort patterns, optimize energy settings, and explain model metrics.</div>', unsafe_allow_html=True)
    
    # ── SEPARATED AGENTIC DATA ZONE ────────────────────────────────────
    with st.container():
        st.markdown("##### 📁 Agentic Sandbox (Private Data)")
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
            train_count = int(row_count * 0.7)
            test_count = row_count - train_count
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
                available_features = [col_map[norm(f)] for f in hm.FEATURES if norm(f) in col_map]
                if available_features:
                    last_rows = active_df[available_features].tail(5)
                    last_rows_str = f"Last 5 data points (Features only):\n{last_rows.to_string()}\n"
            except:
                pass

            # Predict the next value if model is available
            prediction_str = ""
            
            if model_status and st.session_state.get("hvac_model") is not None:
                try:
                    # Prepare the window from active_df
                    norm = lambda s: s.lower().replace(' ','').replace('_','').replace('-','')
                    col_map = {norm(c): c for c in active_df.columns}
                    col_map = {norm(f): f for f in active_df.columns} # Corrected to map normalized name to original name
                    actual_features = [col_map[norm(f)] for f in hm.FEATURES if norm(f) in col_map]
                    
                    if len(actual_features) == len(hm.FEATURES):
                        X_raw = active_df[actual_features].values
                        if len(X_raw) >= hm.WINDOW:
                            feat_scaler = st.session_state.hvac_feat_scaler
                            pmv_scaler = st.session_state.hvac_pmv_scaler
                            model = st.session_state.hvac_model
                            
                            # 1-step prediction (5 mins)
                            if st.session_state.get("hvac_type") == "LSTM":
                                window_raw = X_raw[-hm.WINDOW:].copy()
                                pred_5m = lstm.predict_lstm(model, None, feat_scaler, pmv_scaler, window_raw, window_raw[-1])
                            else:
                                window_raw = X_raw[-hm.WINDOW:].copy()
                                window_sc = feat_scaler.transform(window_raw).reshape(1, hm.WINDOW, len(hm.FEATURES)).astype('float32')
                                pred_sc = model(window_sc, training=False).numpy()[0][0] if hasattr(model, '__call__') else model.predict(window_sc, verbose=0)[0][0]
                                pred_5m = float(pmv_scaler.inverse_transform([[pred_sc]])[0][0])
                            
                            # 12-step recursive prediction (1 hour)
                            # Note: This assumes features stay constant, which is a simplification
                            curr_window = window_raw.copy()
                            preds_1h = []
                            for _ in range(12):
                                win_sc = feat_scaler.transform(curr_window).reshape(1, hm.WINDOW, len(hm.FEATURES)).astype('float32')
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
                f"Total Rows: {row_count} (Splitting: {train_count} for Training, {test_count} for Testing)\n"
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
            # Use Claude client for chatbot as requested
            if not claude_client:
                st.error("❌ Claude client not initialized. Please check your API key.")
            else:
                try:
                    full_system_prompt = (
                        "You are a professional HVAC AI Forecast Assistant. You have access to real-time sensor data and potentially a predictive LSTM model. "
                        "Your primary goal is to provide concise but informative data-driven answers and forecasts. "
                        f"Current Data Context:\n{context_prompt}\n\n"
                        "Guidelines:\n"
                        "1. Use the 'NEXT PMV PREDICTION' and 'FORECASTED PMV' provided in the context to answer forecasting questions IF they are present. "
                        "2. If no prediction is available (e.g. 'prediction_str' is empty), it means the LSTM model has not been trained on the dashboard. In this case, use your knowledge of building physics and the provided 'Last 5 data points' to estimate the likely comfort level. "
                        "3. Do NOT tell the user to wait for training. Provide an immediate analysis based on the data you see. "
                        "4. When using the model, mention you are using the 'LSTM Neural Network'. When estimating manually, mention you are using 'Physics-based reasoning'. "
                        "5. Analyze trends in the 'Last 5 data points' (e.g. is temperature rising?) to add depth to your response. Aim for exactly 5-6 informative bullet points in your analysis.\n"
                        "6. FORMATTING: Use only level 5 or 6 headers (##### or ######) for section titles like 'THERMAL COMFORT FORECAST'. Never use large headers (#, ##, or ###)."
                    )

                    # Prepare input for Claude API
                    # Claude expects messages in a specific format
                    claude_messages = []
                    for msg in st.session_state.messages:
                        # Skip system messages in the history as we provide it via the 'system' param
                        if msg["role"] == "system": continue
                        claude_messages.append({"role": msg["role"], "content": msg["content"]})
                        
                    # Specific Claude implementation using Sonnet 4.5/4.6 with Agentic Plan animation
                    with st.status("🤖 Agentic Forecast is planning...", expanded=True) as status:
                        st.markdown('<div class="thinking-status">📝 Constructing expert prompt... <div class="dot-flashing"></div></div>', unsafe_allow_html=True)
                        stream_text_animation("", delay=0.01)
                        
                        plan_code = f"""
# AGENTIC EXECUTION PLAN
1. DATA_SPLIT: Dividing data into 70% Train ({int(row_count*0.7) if has_df else 0}) and 30% Test ({row_count - int(row_count*0.7) if has_df else 0})
2. RETRIEVE_CONTEXT: Fetching last {row_count if has_df else 0} rows
3. ANALYZE_PHYSICS: Checking thermodynamic constraints
4. INFERENCE: Calling {model_name} (Model ID: {model_id})
5. TOOLS: Code Execution enabled (version: 20250825)
6. FORMAT_OUTPUT: Generating actionable building insights
                        """
                        stream_text_animation(plan_code, delay=0.005, is_code=True, language="markdown")
                        
                        if has_df:
                            st.markdown('<div class="thinking-status">🔍 Scanning sensor trends... <div class="dot-flashing"></div></div>', unsafe_allow_html=True)
                            stream_text_animation("", delay=0.01)
                        
                        st.markdown('<div class="thinking-status">🚀 Executing inference... <div class="dot-flashing"></div></div>', unsafe_allow_html=True)
                        stream_text_animation("", delay=0.01)
                        
                        message = claude_client.messages.create(
                            model=model_id,
                            max_tokens=10000,
                            system=full_system_prompt,
                            messages=claude_messages,
                            tools=[{"type": "code_execution_20250825", "name": "code_execution"}],
                            temperature=0.5,
                        )
                        status.update(label="✅ Response generated!", state="complete", expanded=False)
                    
                    # Extract text from message content blocks robustly
                    text_parts = []
                    for block in message.content:
                        if hasattr(block, 'text'):
                            text_parts.append(block.text)
                        elif hasattr(block, 'content') and isinstance(block.content, str):
                            text_parts.append(block.content)
                        elif isinstance(block, dict) and 'text' in block:
                            text_parts.append(block['text'])
                    response_text = "".join(text_parts)

                    # ANIMATE: Word-by-word streaming for ChatGPT effect
                    stream_text_animation(response_text, delay=0.005)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                except Exception as e:
                    st.error(f"Error: {e}")
