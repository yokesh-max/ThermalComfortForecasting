import streamlit as st
import pandas as pd
import numpy as np
import os
import json
import io
import time
import base64
from anthropic import Anthropic
from dotenv import load_dotenv
from MODELS import lstm
import re

def estimate_pmv_from_sensors(row_dict):
    """Placeholder estimator — Fanger formula is not used."""
    return 0.0

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
    """Simulates a creative word-by-word or character-by-character streaming effect."""
    placeholder = st.empty()
    full_text = ""
    
    # Use word-by-word for larger text blocks (non-code) for "awesome" faster feel
    if not is_code and len(text.split()) > 5:
        words = text.split(" ")
        for i, word in enumerate(words):
            full_text += word + (" " if i < len(words)-1 else "")
            placeholder.markdown(full_text + "▌")
            # Adaptive delay based on word length, average around requested delay
            time.sleep(max(0.01, delay * 5)) 
        placeholder.markdown(full_text)
    else:
        # Character-based for code or very short snippets
        for char in text:
            full_text += char
            if is_code:
                placeholder.code(full_text, language=language)
            else:
                placeholder.markdown(full_text + "▌")
            if delay > 0:
                time.sleep(delay)
        placeholder.markdown(full_text)
    return full_text

def get_ai_insights(df, latest_reading, model_id="claude-sonnet-4-5"):
    """Send data summary and last reading to Claude and return AI analysis + optimal values."""
    if not claude_client:
        return "Claude client not initialised. Check your API key.", None
        
    stats_str = ""
    FEATURES = [
        'Cooling_Power', 'Flowrate', 'CHWR-CHWS', 'Offcoil_Temperature',
        'Return_air_Co2', 'Return_air_static_pressure', 'Return_air_RH'
    ]
    TARGET = 'PMV'
    try:
        norm = lambda s: s.lower().replace(' ','').replace('_','').replace('-','')
        col_map = {norm(c): c for c in df.columns}
        available_features = [col_map[norm(f)] for f in FEATURES if norm(f) in col_map]
        target_col = next((c for c in df.columns if c.lower() == TARGET.lower()), None)
        cols_to_describe = available_features + ([target_col] if target_col else [])
        if cols_to_describe:
            stats = df[cols_to_describe].describe().loc[['mean', 'min', 'max']]
            stats_str = stats.to_string()
    except Exception:
        pass
    
    prompt = f"""
    Analyze the following HVAC sensor dataset and provide a highly concise Professional Overview (no over-explanation).
    
    Overall Dataset Statistics (Mean, Min, Max):
    {stats_str}
    
    Current Building State (Last Reading):
    {latest_reading.to_dict()}
    
    Physics Constraints & System Architecture:
    1. Cooling Power Increase -> PMV Decrease
    2. Flowrate Increase -> PMV Decrease (more cold water)
    3. Return AI CO2 is a proxy for heat load.
    
    Inputs Mapping:
    - Flowrate (m³/s) – Air circulation rate
    - CHWR-CHWS (°C) – Chilled water temperature difference (cooling load indicator)
    - Cooling_Power (W) – Primary control input
    - Return_air_static_pressure (Pa) – System pressure
    - Return_air_Co2 (ppm) – Indoor air quality indicator
    - Offcoil_Temperature (°C) – Air handler discharge temperature
    - Return_air_RH (%) – Indoor relative humidity
    
    Output Objective:
    - PMV value 5 minutes ahead
    
    BREVITY RULE: 
    Do not explain the inputs or outputs listed above. Only use them to inform your 2-3 bullet point analysis.
    
    Task:
    1. Provide a very concise textual analysis (exactly 2-3 short bullet points).
    2. Suggest OPTIMAL numerical values for all 7 features to achieve PMV = 0.0.
    
    FORMATTING:
    - Use Title: "##### 🏗️ Professional HVAC Analysis for PMV-Neutral Comfort"
    - Each section MUST be extremely brief (maximum 1-2 lines or 2-3 bullet points).
    - Provide the 2-3 bullet points under the heading "##### 🔍 Key Insights".
    - Provide a Vertical Markdown Table (7 rows x 2 columns) titled "##### ✅ Optimal HVAC Settings" with columns "Parameters" and "Target Values".
    - Finally, provide the JSON block at the very end.
    
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



# ── FILE ICON MAPPING ─────────────────────────────────────────────
FILE_ICONS = {
    '.png': '🖼️', '.jpg': '🖼️', '.jpeg': '🖼️', '.gif': '🖼️', '.svg': '🖼️',
    '.csv': '📊', '.xlsx': '📊', '.xls': '📊',
    '.md': '📝', '.txt': '📄', '.json': '📋', '.html': '🌐',
    '.py': '🐍', '.pdf': '📕',
}
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.webp'}
TEXT_EXTENSIONS  = {'.csv', '.md', '.txt', '.json', '.html', '.py', '.log', '.yaml', '.yml'}

def _get_file_icon(filename):
    ext = os.path.splitext(filename)[1].lower()
    return FILE_ICONS.get(ext, '📎')

def _get_mime_type(filename):
    ext = os.path.splitext(filename)[1].lower()
    mime_map = {
        '.png': 'image/png', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
        '.gif': 'image/gif', '.svg': 'image/svg+xml', '.bmp': 'image/bmp',
        '.webp': 'image/webp', '.csv': 'text/csv', '.md': 'text/markdown',
        '.txt': 'text/plain', '.json': 'application/json', '.html': 'text/html',
        '.py': 'text/x-python', '.pdf': 'application/pdf',
        '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    }
    return mime_map.get(ext, 'application/octet-stream')


def _extract_filenames_from_code(code_str):
    """
    Parse Python/bash code for file-creation patterns and return
    a dict mapping extension-category to list of filenames found.
    e.g. {'image': ['PCDL_PMV_Forecast_Advanced.png'], 'csv': ['PCDL_PMV_60min_Forecast.csv']}
    """
    import re
    found = {'image': [], 'csv': [], 'text': [], 'other': []}
    
    # savefig('filename.png') / savefig("filename.png")
    for m in re.finditer(r"savefig\s*\(\s*['\"]([^'\"]+)['\"]", code_str):
        fname = os.path.basename(m.group(1))
        found['image'].append(fname)
    
    # to_csv('filename.csv') / to_csv("filename.csv")
    for m in re.finditer(r"to_csv\s*\(\s*['\"]([^'\"]+)['\"]", code_str):
        found['csv'].append(os.path.basename(m.group(1)))
    
    # to_excel('filename.xlsx')
    for m in re.finditer(r"to_excel\s*\(\s*['\"]([^'\"]+)['\"]", code_str):
        found['other'].append(os.path.basename(m.group(1)))
    
    # to_json('filename.json')
    for m in re.finditer(r"to_json\s*\(\s*['\"]([^'\"]+)['\"]", code_str):
        found['text'].append(os.path.basename(m.group(1)))
    
    # to_markdown('filename.md')
    for m in re.finditer(r"to_markdown\s*\(\s*['\"]([^'\"]+)['\"]", code_str):
        found['text'].append(os.path.basename(m.group(1)))
    
    # open('filename', 'w') patterns
    for m in re.finditer(r"open\s*\(\s*['\"]([^'\"]+)['\"].*?['\"]w", code_str):
        fname = os.path.basename(m.group(1))
        ext = os.path.splitext(fname)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            found['image'].append(fname)
        elif ext == '.csv':
            found['csv'].append(fname)
        elif ext in TEXT_EXTENSIONS:
            found['text'].append(fname)
        else:
            found['other'].append(fname)
    
    return found


def extract_generated_files(message):
    """
    Extract generated files (images, CSVs, text, etc.) from Claude
    code-execution response content blocks.
    Uses a two-pass approach:
      Pass 1 — collect filenames from bash commands + text_editor create paths
      Pass 2 — extract file data and match to real filenames
    Returns a list of dicts: [{name, data_b64, media_type, ext}, ...]
    """
    files = []
    if not hasattr(message, 'content'):
        return files

    blocks = message.content

    # ── PASS 1: Collect filenames from commands ──────────────────────
    image_names = []   # ordered list of image filenames from savefig etc.
    text_editor_names = []  # filenames from text_editor create commands
    
    for block in blocks:
        btype = getattr(block, 'type', '') if not isinstance(block, dict) else block.get('type', '')
        
        # Bash commands → scan code for savefig / to_csv / etc.
        if btype == 'server_tool_use':
            tool_name = getattr(block, 'name', '') if not isinstance(block, dict) else block.get('name', '')
            inp = getattr(block, 'input', {}) if not isinstance(block, dict) else block.get('input', {})
            
            if tool_name == 'bash_code_execution' and inp:
                cmd = inp.get('command', '') if isinstance(inp, dict) else getattr(inp, 'command', '')
                if cmd:
                    found = _extract_filenames_from_code(cmd)
                    image_names.extend(found['image'])
                    # CSV / text / other files created via bash script
                    for fn in found['csv'] + found['text'] + found['other']:
                        text_editor_names.append(fn)
            
            elif tool_name == 'text_editor_code_execution' and inp:
                cmd = inp.get('command', '') if isinstance(inp, dict) else getattr(inp, 'command', '')
                path = inp.get('path', '') if isinstance(inp, dict) else getattr(inp, 'path', '')
                if cmd == 'create' and path:
                    text_editor_names.append(os.path.basename(path))

    # ── PASS 2: Extract file data ────────────────────────────────────
    image_counter = 0
    te_create_counter = 0  # tracks text_editor create order
    
    for block in blocks:
        btype = getattr(block, 'type', '') if not isinstance(block, dict) else block.get('type', '')

        # ── Images from code_execution_result ──
        if btype == 'code_execution_result':
            sub_content = getattr(block, 'content', []) if not isinstance(block, dict) else block.get('content', [])
            if not isinstance(sub_content, list):
                sub_content = [sub_content]
            for item in sub_content:
                item_type = getattr(item, 'type', '') if not isinstance(item, dict) else item.get('type', '')
                if item_type == 'image':
                    src = getattr(item, 'source', None) if not isinstance(item, dict) else item.get('source')
                    if src:
                        data_b64 = getattr(src, 'data', '') if not isinstance(src, dict) else src.get('data', '')
                        media = getattr(src, 'media_type', 'image/png') if not isinstance(src, dict) else src.get('media_type', 'image/png')
                        ext = '.' + media.split('/')[-1].split('+')[0] if '/' in media else '.png'
                        # Use real filename if available
                        if image_counter < len(image_names):
                            fname = image_names[image_counter]
                        else:
                            fname = f"generated_chart_{image_counter + 1}{ext}"
                        image_counter += 1
                        files.append({'name': fname, 'data_b64': data_b64, 'media_type': media, 'ext': os.path.splitext(fname)[1].lower()})

        # ── server_tool_use: capture file text from text_editor create ──
        if btype == 'server_tool_use':
            tool_name = getattr(block, 'name', '') if not isinstance(block, dict) else block.get('name', '')
            inp = getattr(block, 'input', {}) if not isinstance(block, dict) else block.get('input', {})
            
            if tool_name == 'text_editor_code_execution' and inp:
                cmd = inp.get('command', '') if isinstance(inp, dict) else getattr(inp, 'command', '')
                path = inp.get('path', '') if isinstance(inp, dict) else getattr(inp, 'path', '')
                file_text = inp.get('file_text', '') if isinstance(inp, dict) else getattr(inp, 'file_text', '')
                if cmd == 'create' and path and file_text:
                    fname = os.path.basename(path)
                    ext = os.path.splitext(fname)[1].lower()
                    data_b64 = base64.b64encode(file_text.encode('utf-8')).decode('utf-8')
                    media = _get_mime_type(fname)
                    files.append({'name': fname, 'data_b64': data_b64, 'media_type': media, 'ext': ext})

        # ── bash_code_execution_tool_result: capture stdout files ──
        if btype == 'bash_code_execution_tool_result':
            content_obj = getattr(block, 'content', None) if not isinstance(block, dict) else block.get('content')
            if content_obj:
                stdout = getattr(content_obj, 'stdout', '') if not isinstance(content_obj, dict) else content_obj.get('stdout', '')
                if stdout:
                    # Convention: FILE_OUTPUT:<filename>:<base64data>
                    if 'FILE_OUTPUT:' in stdout:
                        for line in stdout.split('\n'):
                            if line.startswith('FILE_OUTPUT:'):
                                parts = line.split(':', 2)
                                if len(parts) == 3:
                                    fname = parts[1]
                                    data_b64 = parts[2]
                                    ext = os.path.splitext(fname)[1].lower()
                                    media = _get_mime_type(fname)
                                    files.append({'name': fname, 'data_b64': data_b64, 'media_type': media, 'ext': ext})
                    
                    # Detect base64-encoded file output from bash scripts
                    # e.g. "Saved forecast_data.csv" → check if we have pending text_editor filenames
                    # For CSV/text files generated by Python scripts via bash,
                    # they won't appear as content blocks — we capture them from
                    # the savefig/to_csv patterns if Claude also outputs them via text_editor

    # Deduplicate by filename (keep first occurrence)
    seen = set()
    deduped = []
    for f in files:
        if f['name'] not in seen:
            seen.add(f['name'])
            deduped.append(f)
    return deduped


@st.dialog("📂 File Preview", width="large")
def preview_file_dialog(file_info):
    """Streamlit dialog that shows a file preview with download and close."""
    fname = file_info['name']
    data_b64 = file_info['data_b64']
    ext = file_info.get('ext', os.path.splitext(fname)[1]).lower()
    media_type = file_info.get('media_type', 'application/octet-stream')
    raw_bytes = base64.b64decode(data_b64)

    st.markdown(f"##### {_get_file_icon(fname)} {fname}")
    st.markdown("---")

    # ── Preview based on file type ──
    if ext in IMAGE_EXTENSIONS:
        st.image(raw_bytes, caption=fname, use_container_width=True)
    elif ext == '.csv':
        try:
            df = pd.read_csv(io.BytesIO(raw_bytes))
            st.dataframe(df, use_container_width=True)
        except Exception:
            st.code(raw_bytes.decode('utf-8', errors='replace'), language='text')
    elif ext == '.md':
        st.markdown(raw_bytes.decode('utf-8', errors='replace'))
    elif ext == '.json':
        try:
            parsed = json.loads(raw_bytes.decode('utf-8'))
            st.json(parsed)
        except Exception:
            st.code(raw_bytes.decode('utf-8', errors='replace'), language='json')
    elif ext in TEXT_EXTENSIONS:
        lang_map = {'.py': 'python', '.html': 'html', '.yaml': 'yaml', '.yml': 'yaml', '.txt': 'text'}
        st.code(raw_bytes.decode('utf-8', errors='replace'), language=lang_map.get(ext, 'text'))
    else:
        st.info(f"Preview not available for `{ext}` files. Use the download button below.")

    st.markdown("---")
    col_dl, col_close = st.columns(2)
    with col_dl:
        st.download_button(
            label="⬇️ Download File",
            data=raw_bytes,
            file_name=fname,
            mime=media_type,
            use_container_width=True,
            type="primary"
        )
    with col_close:
        if st.button("✖️ Close", use_container_width=True):
            st.rerun()


def display_file_cards(files, msg_idx=0):
    """Render clickable file cards for a list of generated files."""
    if not files:
        return
    st.markdown('<div class="file-cards-row">', unsafe_allow_html=True)
    cols = st.columns(min(len(files), 4))
    for i, f in enumerate(files):
        with cols[i % len(cols)]:
            icon = _get_file_icon(f['name'])
            ext_label = f.get('ext', '').replace('.', '').upper() or 'FILE'
            if st.button(
                f"{icon}  {f['name']}",
                key=f"file_card_{msg_idx}_{i}_{f['name']}",
                use_container_width=True,
                help=f"Click to preview ({ext_label})"
            ):
                preview_file_dialog(f)
    st.markdown('</div>', unsafe_allow_html=True)


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
        /* ── Generated File Cards ── */
        .file-card {
            display: inline-flex;
            align-items: center;
            gap: 10px;
            background: linear-gradient(135deg, #EDE7F6, #E3F2FD);
            border: 1px solid #B0BEC5;
            border-radius: 12px;
            padding: 12px 18px;
            margin: 6px 8px 6px 0;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 2px 6px rgba(0,0,0,0.08);
            max-width: 280px;
        }
        .file-card:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 18px rgba(26, 115, 232, 0.25);
            border-color: #1A73E8;
        }
        .file-card .file-icon {
            font-size: 1.8rem;
            flex-shrink: 0;
        }
        .file-card .file-info {
            display: flex;
            flex-direction: column;
            min-width: 0;
        }
        .file-card .file-name {
            font-weight: 700;
            font-size: 0.88rem;
            color: #1A237E;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .file-card .file-type {
            font-size: 0.72rem;
            color: #5F6368;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .file-cards-row {
            display: flex;
            flex-wrap: wrap;
            margin-top: 10px;
            gap: 4px;
        }
        </style>
    """, unsafe_allow_html=True)

    # Anthropic client is initialized at the top of the script
    reset_cnt = st.session_state.get("reset_counter", 0)
    model_name = st.session_state.get(f"model_select_{reset_cnt}", "Claude Sonnet 4.5")
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
        if st.session_state.get("agentic_df") is not None:
            initial_msg = f"Hello! I'm your **Agentic Forecast Assistant**. I've successfully synced with your **Agentic Sandbox** data (`{st.session_state.agentic_df_name}`). I'm ready to analyze it or answer any questions!"
        else:
            initial_msg = "Hello! I'm your **Agentic Forecast Assistant**. I'm ready to help you analyze HVAC data. You can upload a file to my **Agentic Sandbox** above for isolated analysis or ask me about the dashboard data."
        st.session_state.messages.append({"role": "assistant", "content": initial_msg})

    # Display chat messages
    for msg_idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            is_json_rendered = False
            if message["role"] == "assistant" and "pmv_prediction" in message.get("content", ""):
                try:
                    content = message["content"]
                    json_str = content.strip()
                    if "```json" in content:
                        json_str = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        # Sometimes placed in a generic code block
                        blocks = content.split("```")
                        if len(blocks) >= 3:
                            json_str = blocks[1].strip()
                            if json_str.startswith("json\n"):
                                json_str = json_str[5:]
                    
                    # Fallback to pure bracket extraction if text wraps it
                    if not json_str.startswith("{"):
                        start = json_str.find("{")
                        end = json_str.rfind("}")
                        if start != -1 and end != -1:
                            json_str = json_str[start:end+1]
                    
                    data = json.loads(json_str)
                    if data.get("type") == "pmv_prediction":
                        # Render the Markdown Report
                        st.markdown(data.get("report", ""))
                        
                        # Render Plot and Table
                        pmv_data = data.get("data", {})
                        timestamps = pmv_data.get("timestamps", [])
                        pmv_values = pmv_data.get("pmv_values", [])
                        if timestamps and pmv_values:
                            import plotly.graph_objects as go
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=timestamps, y=pmv_values, 
                                mode='lines+markers', line=dict(color='#1A73E8', width=3),
                                marker=dict(color='#FF5722', size=8)
                            ))
                            fig.update_layout(title="🔮 Predicted Thermal Comfort (PMV)", height=350, template="plotly_white")
                            st.plotly_chart(fig, use_container_width=True)
                            
                            with st.expander("📊 View JSON Data Table", expanded=False):
                                if len(timestamps) == len(pmv_values):
                                    df_json = pd.DataFrame({"Time": timestamps, "Predicted PMV": pmv_values})
                                    st.dataframe(df_json, use_container_width=True)
                                    
                                    cols = st.columns(2)
                                    csv_data = df_json.to_csv(index=False)
                                    cols[0].download_button("📥 Download Data (CSV)", data=csv_data, file_name=f"pmv_forecast_{msg_idx}.csv", mime="text/csv", key=f"dl_csv_{msg_idx}")
                                    
                                    md_report = data.get("report", "")
                                    # Clean markdown for download (remove # headers and ** bold)
                                    md_clean = re.sub(r'^#+\s*', '', md_report, flags=re.MULTILINE)
                                    md_clean = re.sub(r'\*\*(.*?)\*\*', r'\1', md_clean)
                                    
                                    # Append Time Taken only to the downloadable MD content
                                    duration = message.get("duration", 0.0)
                                    md_download_content = f"{md_clean}\n\n---\nTime Taken\nAI Generation Time: {duration:.2f} seconds"
                                    
                                    cols[1].download_button("📝 Download Report (MD)", data=md_download_content, file_name=f"pmv_analysis_{msg_idx}.md", mime="text/markdown", key=f"dl_md_{msg_idx}")
                                else:
                                    st.warning("⚠️ Data length mismatch in JSON response.")
                        is_json_rendered = True
                except Exception:
                    pass
            
            if not is_json_rendered:
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

            # Provide a CSV snippet for the AI to read using its code tool
            csv_snippet = ""
            try:
                # Provide the last 100 rows as a CSV string for more robust AI training
                csv_snippet = active_df.tail(100).to_csv(index=False)
            except:
                pass

            active_source_str = (
                f"Active Data Source: In-Memory Sandbox Data ('{st.session_state.agentic_df_name}')" if is_using_sandbox 
                else f"Active Data Source: Main Dashboard Data ('{st.session_state.main_df_name}')"
            )
            context_prompt = (
                f"{active_source_str}\n"
                f"Total Rows: {row_count} (Splitting: {train_count} for Training, {test_count} for Testing)\n"
                f"Columns: {cols}\n"
                f"{stats_str}\n"
                f"{last_rows_str}\n"
                f"--- FULL DATA SNIPPET (Up to 1000 Rows for AI Internal Training) ---\n"
                f"{csv_snippet}\n"
                f"--- END FULL DATA ---"
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
                    sandbox_instruction = ""
                    if is_using_sandbox:
                        sandbox_instruction = (
                            "**SANDBOX MODE ACTIVE**: The user has uploaded an isolated dataset to your private sandbox. "
                            "For detailed analysis, model training, or complex queries, you MUST use the provided `DATA/active_data_context.csv` file. "
                            "This file contains the complete sandbox data corresponding to the user's latest upload.\n\n"
                        )

                    # Sync active dataframe to disk for Claude's code_execution tool
                    if has_df:
                        try:
                            # Ensure DATA directory exists
                            os.makedirs("DATA", exist_ok=True)
                            # Use the full dataframe for context, but limit size if extremely large
                            active_df.to_csv("DATA/active_data_context.csv", index=False)
                        except Exception as e:
                            st.warning(f"⚠️ Could not sync sandbox to disk: {e}")

                    full_system_prompt = (
                        f"{sandbox_instruction}"
                        "You are a Senior HVAC Systems Engineer and AI Specialist.\n"
                        "Your goal is to provide highly detailed, professional, and human-readable thermal comfort forecasts that directly address the user's specific questions.\n\n"
                        "ABSOLUTE RULES — PRODUCTION ENVIRONMENT:\n"
                        "- You are in a PRODUCTION dashboard seen by real building engineers.\n"
                        "- NEVER output Python code, import statements, code blocks, or programming syntax of any kind.\n"
                        "- NEVER show code fences (```), variable assignments, print() calls, or library imports.\n"
                        "- NEVER mention 'code_execution', 'pd.read_csv', 'sklearn', 'tensorflow', or any programming tool.\n"
                        "- ALL your responses must be clean, professional English text or valid JSON only.\n"
                        "- If you need to perform calculations, do them silently and present ONLY the final results.\n\n"
                        "DATA ANALYSIS APPROACH:\n"
                        "- The full dataset statistics and recent readings are provided below in 'Current Data Context'.\n"
                        "- Use these statistics (mean, min, max, trends) along with your HVAC engineering expertise to generate accurate forecasts.\n"
                        "- Apply thermodynamic principles: Cooling Power ↑ → PMV ↓, Flowrate ↑ → PMV ↓, CO2 ↑ → occupancy/heat load ↑.\n"
                        "- Base your PMV predictions on the observed data patterns and physics constraints.\n\n"
                        "CORE OPERATING PRINCIPLES:\n"
                        "1. CONTEXTUAL RELEVANCY: ALWAYS prioritize answering the user's specific question.\n"
                        "2. CLARITY & DETAIL: Responses must be clear, detailed, and accurate. Use professional engineering terminology but remain accessible.\n"
                        "3. INTERNAL EXPERTISE: Use your own internal HVAC knowledge to explain building dynamics. You are the domain expert.\n"
                        "4. DATA INTEGRITY: Use the provided sensor data statistics accurately. Do not hallucinate values.\n"
                        "5. DOWNLOADABLE OUTPUTS: Your JSON output must contain the full report (MD) and calculated PMV values. The UI generates download buttons automatically.\n"
                        "6. STRUCTURED OUTPUT: ALWAYS return structured output in JSON format when data or forecasts are involved.\n\n"
                        "SYSTEM ARCHITECTURE (DATA MAPPING):\n"
                        "Inputs (provided in Current Data Context):\n"
                        "- Flowrate (m³/s) – Air circulation rate\n"
                        "- CHWR-CHWS (°C) – Chilled water temperature difference (cooling load indicator)\n"
                        "- Cooling_Power (W) – Primary control input\n"
                        "- Return_air_static_pressure (Pa) – System pressure\n"
                        "- Return_air_Co2 (ppm) – Indoor air quality indicator\n"
                        "- Offcoil_Temperature (°C) – Air handler discharge temperature\n"
                        "- Return_air_RH (%) – Indoor relative humidity\n\n"
                        "Output (Objective):\n"
                        "- PMV value 5 minutes ahead\n\n"
                        f"Current Data Context:\n{context_prompt}\n\n"
                        "WHEN the user asks for PMV prediction, a specific model output (LSTM, PCDL, PCEL), or a general forecast:\n"
                        "- Return ONLY valid JSON in this format:\n\n"
                        "{\n"
                        '  "type": "pmv_prediction",\n'
                        '  "data": {\n'
                        '    "timestamps": ["HH:MM", "..."],\n'
                        '    "pmv_values": [float, float, ...]\n'
                        '  },\n'
                        '  "report": "Markdown formatted executive report here"\n'
                        "}\n\n"
                        "REPORT GUIDELINES:\n"
                        "- The 'report' field MUST be a comprehensive Markdown document with the following sections:\n"
                        "  ## 🔮 Advanced HVAC Forecast — [Timeframe]\n"
                        "  **Direct Response**: Start with a 1-2 paragraph section that explicitly answers the user's specific question using the data and context provided.\n\n"
                        "  **Executive Summary**: A 2-3 sentence overview of the current comfort state and the upcoming forecast trend in plain, readable English.\n\n"
                        "  ### 📊 Forecast Metrics Summary\n"
                        "  | Metric | Value | Interpretation |\n"
                        "  |--------|-------|----------------|\n"
                        "  | **Next 5-min PMV** | `val` | [Brief description of feeling] |\n"
                        "  | **1-Hour Avg PMV** | `val` | [Stability assessment] |\n"
                        "  | **Min/Max Range** | `min` to `max` | [Volatility assessment] |\n"
                        "  | **Trend Status** | [Emoji] [Trend] | [Reason for trend] |\n"
                        "  | **Comfort Zone** | [Status] | [Action needed?] |\n\n"
                        "  ### 🔍 In-Depth Physics & Sensor Analysis\n"
                        "  - **Thermal Load Impact**: Analyze how 'Return Air CO2' (occupancy proxy) and 'Return Air RH' are influencing the PMV.\n"
                        "  - **HVAC Efficiency**: Evaluate the effectiveness of 'Cooling Power' and 'Flowrate' based on the 'CHWR-CHWS' differential.\n"
                        "  - **Thermodynamic Constraints**: Explain how the 'Offcoil Temperature' is responding to current building demands.\n\n"
                        "  ### 💡 Engineering Recommendations\n"
                        "  - Provide 2-3 specific, actionable settings adjustments to optimize comfort (PMV -> 0.0) while minimizing energy consumption.\n\n"
                        "  ### 🏗️ AI Model Context\n"
                        "  - **Methodology**: Explain in plain English how the forecast was derived from the sensor data patterns.\n"
                        "  - **Features Used**: List the sensors analyzed.\n"
                        "  - **Confidence**: State your confidence level based on data quality and consistency.\n\n"
                        "- DO NOT include any conversational filler outside the JSON block.\n\n"
                        "WHEN the user asks to compare models (e.g., \"compare LSTM, PCEL, and PCDL\"):\n"
                        "- Respond in plain text (no JSON).\n"
                        "- Provide a concise but valuable comparison in a Markdown table based on your internal expertise.\n"
                        "- The table MUST compare the models on these criteria: Core Architecture, Key Strength, Best Use-Case, and Potential Weakness.\n\n"
                        "IF the request is purely conversational (greetings, general questions):\n"
                        "- Respond in a professional, helpful engineering tone in plain text (no JSON) that directly answers the user's question.\n"
                    )

                    # Prepare input for Claude API
                    claude_messages = []
                    for msg in st.session_state.messages:
                        if msg["role"] == "system": continue
                        claude_messages.append({"role": msg["role"], "content": msg["content"]})
                        
                    # Determine intent: skip plan animation for greetings or short non-data questions
                    user_msg = prompt.strip().lower()
                    is_greeting = user_msg in ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "hi there", "hello there", "sup"]
                    is_short = len(user_msg.split()) <= 3
                    ds_keywords = ["data", "file", "csv", "predict", "forecast", "pmv", "temp", "co2", "humid", "rh", "flowrate", "power", "model", "lstm", "pcel", "pcdl", "analy", "metric", "accuracy", "mae", "rmse", "plot", "chart"]
                    is_data_question = any(k in user_msg for k in ds_keywords) or (not is_greeting and not is_short)
                    
                    # Agentic Plan animation
                    with st.status("🤖 Agentic Forecast is planning...", expanded=True) as status:
                        anim_container = st.empty()
                        if is_data_question:
                            with anim_container.container():
                                st.markdown('<div class="thinking-status">📝 Constructing expert prompt... <div class="dot-flashing"></div></div>', unsafe_allow_html=True)
                                
                                plan_code = f"""
##### 📋 AGENTIC EXECUTION PLAN
1. DATA_SPLIT: Dividing data into 70% Train ({int(row_count*0.7) if has_df else 0}) and 30% Test ({row_count - int(row_count*0.7) if has_df else 0})
2. RETRIEVE_CONTEXT: Fetching last {row_count if has_df else 0} rows
3. ANALYZE_PHYSICS: Checking thermodynamic constraints
4. INFERENCE: Calling {model_name} (Model ID: {model_id})
5. FORMAT_OUTPUT: Generating actionable building insights
                                """
                                stream_text_animation(plan_code, delay=0.002, is_code=True, language="markdown")
                                
                                if has_df:
                                    st.markdown('<div class="thinking-status">🔍 Scanning sensor trends... <div class="dot-flashing"></div></div>', unsafe_allow_html=True)
                                
                                st.markdown('<div class="thinking-status">🚀 Executing inference... <div class="dot-flashing"></div></div>', unsafe_allow_html=True)
                        else:
                            with anim_container.container():
                                st.markdown('<div class="thinking-status">🤔 Processing conversational query... <div class="dot-flashing"></div></div>', unsafe_allow_html=True)
                            
                        import time
                        start_time = time.perf_counter()
                        message = claude_client.messages.create(
                            model=model_id,
                            max_tokens=10000,
                            system=full_system_prompt,
                            messages=claude_messages,
                            temperature=0.5,
                        )
                        end_time = time.perf_counter()
                        ai_duration = end_time - start_time
                        
                        anim_container.empty()
                        status.update(label="✅ Response generated!", state="complete", expanded=False)
                    
                    # Extract ONLY text blocks — skip any code/tool blocks entirely
                    text_parts = []
                    for block in message.content:
                        block_type = getattr(block, 'type', None)
                        # Only accept pure text blocks; skip code_execution, tool_use, etc.
                        if block_type == 'text' and hasattr(block, 'text'):
                            text_parts.append(block.text)
                        elif isinstance(block, dict) and block.get('type') == 'text' and 'text' in block:
                            text_parts.append(block['text'])
                    response_text = "".join(text_parts)

                    # Replace standard animated streaming with JSON-aware output rendering
                    try:
                        json_str = response_text.strip()
                        if "```json" in response_text:
                            json_str = response_text.split("```json")[1].split("```")[0].strip()
                        elif "```" in response_text:
                            blocks = response_text.split("```")
                            if len(blocks) >= 3:
                                json_str = blocks[1].strip()
                                if json_str.startswith("json\n"):
                                    json_str = json_str[5:]
                        
                        # Fallback pure bracket extraction
                        if not json_str.startswith("{"):
                            start = json_str.find("{")
                            end = json_str.rfind("}")
                            if start != -1 and end != -1:
                                json_str = json_str[start:end+1]
                            
                        data = json.loads(json_str)
                        if data.get("type") == "pmv_prediction":
                            # It's a valid JSON response! Render nicely in chat immediately
                            stream_text_animation(data.get("report", ""), delay=0.01)
                            
                            pmv_data = data.get("data", {})
                            timestamps = pmv_data.get("timestamps", [])
                            pmv_values = pmv_data.get("pmv_values", [])
                            if timestamps and pmv_values:
                                import plotly.graph_objects as go
                                fig = go.Figure()
                                fig.add_trace(go.Scatter(
                                    x=timestamps, y=pmv_values, 
                                    mode='lines+markers', line=dict(color='#1A73E8', width=3),
                                    marker=dict(color='#FF5722', size=8)
                                ))
                                fig.update_layout(title="🔮 Predicted Thermal Comfort (PMV)", height=350, template="plotly_white")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                with st.expander("📊 View JSON Data Table", expanded=False):
                                    if len(timestamps) == len(pmv_values):
                                        df_json = pd.DataFrame({"Time": timestamps, "Predicted PMV": pmv_values})
                                        st.dataframe(df_json, use_container_width=True)
                                        
                                        idx = len(st.session_state.messages)
                                        cols = st.columns(2)
                                        csv_data = df_json.to_csv(index=False)
                                        cols[0].download_button("📥 Download Data (CSV)", data=csv_data, file_name=f"pmv_forecast_{idx}.csv", mime="text/csv", key=f"dl_csv_{idx}")
                                        
                                        md_report = data.get("report", "")
                                        # Clean markdown for download (remove # headers and ** bold)
                                        md_clean = re.sub(r'^#+\s*', '', md_report, flags=re.MULTILINE)
                                        md_clean = re.sub(r'\*\*(.*?)\*\*', r'\1', md_clean)
                                        
                                        # Append Time Taken only to the downloadable MD content
                                        md_download_content = f"{md_clean}\n\n---\nTime Taken\nAI Generation Time: {ai_duration:.2f} seconds"
                                        
                                        cols[1].download_button("📝 Download Report (MD)", data=md_download_content, file_name=f"pmv_analysis_{idx}.md", mime="text/markdown", key=f"dl_md_{idx}")
                                    else:
                                        st.warning("⚠️ Data length mismatch in JSON response.")
                        else:
                            stream_text_animation(response_text, delay=0.005)
                    except Exception:
                        stream_text_animation(response_text, delay=0.005)

                    # Only store the cleaned-up output to keep history professional
                    final_content_to_store = response_text
                    try:
                        if "data" in locals() and data.get("type") == "pmv_prediction":
                            # Store the JSON or a clean summary to prevent raw text from re-appearing
                            # We keep the JSON format so it can be re-rendered on refresh
                            final_content_to_store = response_text 
                        else:
                            # If it's not JSON, still store it but keep it brief if possible
                            pass
                    except Exception:
                        pass

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": final_content_to_store,
                        "duration": ai_duration if 'ai_duration' in locals() else 0.0
                    })
                except Exception as e:
                    st.error(f"Error: {e}")
