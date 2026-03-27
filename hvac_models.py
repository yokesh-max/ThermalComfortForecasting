import numpy as np
import warnings
import os
import random

# Suppress TensorFlow internal C++ logging (fixes _audio_microfrontend_op.so not found warnings)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Fix random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)

# ── CONFIGURATION ─────────────────────────────────────────────────
FEATURES = [
    'Cooling_Power',              # primary AC control — 75% importance
    'Flowrate',                   # cold water volume
    'CHWR-CHWS',                  # cooling effort
    'Offcoil_Temperature',        # supply air coldness
    'Return_air_Co2',             # occupancy proxy
    'Return_air_static_pressure', # airflow state
    'Return_air_RH',              # room humidity
]
TARGET      = 'PMV'
WINDOW      = 12      # 12 steps × 5 min = 1 hour lookback
TRAIN_RATIO = 0.7     # 70% train / 30% test
EPOCHS      = 100     # max epochs (early stopping will cut this short)
BATCH_SIZE  = 16
PATIENCE    = 15      # early stopping patience
SAVE_DIR    = "saved_models"

# THERMAL COMFORT LOGIC (ISO 7730 / Fanger PMV)


def calculate_fanger_pmv(ta, tr, vel, rh, met=1.2, clo=0.6):
    """
    Standard Fanger PMV formula implementation.
    """
    pa = (rh / 100.0) * 10 * np.exp(16.6536 - 4030.183 / (ta + 235)) 
    m = met * 58.15 
    icl = 0.155 * clo 
    fcl = 1.0 + 1.29 * icl if icl < 0.078 else 1.05 + 0.645 * icl
    hcf = 12.1 * np.sqrt(vel)
    taa = ta + 273.15
    tra = tr + 273.15
    tcl = taa + (35.5 - ta) / (3.5 * icl * fcl + 1)
    hl1 = 3.05 * 0.001 * (5733 - 6.99 * (m - 58.15) - pa)
    hl2 = 0.42 * (m - 58.15 - 58.15) if (m - 58.15) > 58.15 else 0
    hl3 = 1.7 * 0.00001 * m * (5867 - pa)
    hl4 = 0.0014 * m * (34 - ta)
    hl5 = 3.96 * 1e-8 * fcl * (np.power(tcl, 4) - np.power(tra, 4))
    hl6 = fcl * hcf * (tcl - taa)
    ts = 0.303 * np.exp(-0.036 * m) + 0.028
    pmv = ts * (m - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)
    return float(pmv)



def get_comfort_descriptor(pmv):
    """
    Map numerical PMV to human-readable comfort status.
    Ranges:
    - Above +3.0: Very Hot
    - +1.0 to +2.99: Warm
    - -0.99 to +0.99: Comfortable
    - -1.0 to -2.99: Cool
    - Below -3.0: Very Cold
    """
    if pmv >= 3.0: return "🔥 Very Hot"
    if pmv >= 1.0: return "☀️ Warm"
    if pmv > -1.0: return "✅ Comfortable"
    if pmv > -3.0: return "🥶 Cool"
    return "🧊 Very Cold"

# Physics constraints are now primarily handled via validation logic in app.py


# HELPER FUNCTIONS

def make_windows(X, y, win=WINDOW):
    """Slide a window of `win` rows along the data."""
    Xw, yw = [], []
    for i in range(len(X) - win):
        Xw.append(X[i : i+win])
        yw.append(y[i + win])
    return np.array(Xw), np.array(yw)


def prepare_hvac_data(df, train_ratio=TRAIN_RATIO):
    """
    Full pipeline: map columns → split → scale → window.
    Returns: X_train_w, y_train_w, X_test_w, y_test_w,
             y_train_raw_w, y_test_raw_w, feat_scaler, pmv_scaler, error_msg
    """
    current_cols = df.columns.tolist()
    mapped_features = []
    
    norm = lambda s: s.lower().replace(' ','').replace('_','').replace('-','')
    for feat in FEATURES:
        if feat in current_cols:
            mapped_features.append(feat)
        else:
            match = next(
                (c for c in current_cols if norm(c) == norm(feat)), None
            )
            if match:
                mapped_features.append(match)
            else:
                return None, None, None, None, None, None, None, None, \
                       f"Missing required column: {feat}"
    
    target_col = next(
        (c for c in current_cols if c.lower() == TARGET.lower()), None
    )
    if not target_col:
        return None, None, None, None, None, None, None, None, \
               f"Missing target column: {TARGET}"
    
    data = df[mapped_features + [target_col]].copy()
    data = data.interpolate(method='linear', limit=6, limit_direction='both').dropna().reset_index(drop=True)
    X_raw = data[mapped_features].values
    y_raw = data[target_col].values
    
    # Split BEFORE scaling (no data leakage)
    split = int(len(X_raw) * train_ratio)
    X_train_raw = X_raw[:split]
    X_test_raw  = X_raw[split:]
    y_train_raw = y_raw[:split]
    y_test_raw  = y_raw[split:]
    
    # Fit scalers on TRAINING data only
    feat_scaler = MinMaxScaler()
    pmv_scaler  = MinMaxScaler()
    
    X_train_sc = feat_scaler.fit_transform(X_train_raw)
    X_test_sc  = feat_scaler.transform(X_test_raw) if len(X_test_raw) > 0 else np.empty((0, X_train_raw.shape[1]))
    
    y_train_sc = pmv_scaler.fit_transform(y_train_raw.reshape(-1,1)).ravel()
    y_test_sc  = pmv_scaler.transform(y_test_raw.reshape(-1,1).astype('float32')).ravel() if len(y_test_raw) > 0 else np.array([])
    
    # Build windows
    X_train_w, y_train_w   = make_windows(X_train_sc, y_train_sc)
    X_test_w,  y_test_w    = make_windows(X_test_sc,  y_test_sc)
    _,         y_train_raw_w = make_windows(X_train_raw, y_train_raw)
    _,         y_test_raw_w  = make_windows(X_test_raw,  y_test_raw)
    
    return (X_train_w, y_train_w, X_test_w, y_test_w,
            y_train_raw_w, y_test_raw_w,
            feat_scaler, pmv_scaler, None)


def evaluate_model(y_true_raw, preds_scaled, pmv_scaler):
    """Inverse-scale predictions and compute MAE, RMSE, R², Bias, MAPE."""
    if len(y_true_raw) == 0:
        return np.array([]), 0.0, 0.0, 0.0, 0.0, 0.0
        
    preds_raw = pmv_scaler.inverse_transform(
        preds_scaled.reshape(-1,1)).ravel()
    mae  = mean_absolute_error(y_true_raw, preds_raw)
    rmse = np.sqrt(mean_squared_error(y_true_raw, preds_raw))
    r2   = r2_score(y_true_raw, preds_raw)
    
    # New metrics
    bias = np.sum(y_true_raw - preds_raw)
    mape = np.mean(np.abs((y_true_raw - preds_raw) / (np.abs(y_true_raw) + 1e-10))) * 100
    
    return preds_raw, mae, rmse, r2, bias, mape


def check_physics_violations(X_test_raw, preds_raw, cooling_idx=0):
    """
    Check if predictions violate basic physics:
    - High cooling power should lead to low PMV
    Returns: list of violation indices
    """
    violations = []
    cp_norm = (X_test_raw[:, cooling_idx] - X_test_raw[:, cooling_idx].min()) / \
              (X_test_raw[:, cooling_idx].max() - X_test_raw[:, cooling_idx].min())
    
    for i in range(len(preds_raw) - 1):
        cp_change = cp_norm[i+1] - cp_norm[i]
        pmv_change = preds_raw[i+1] - preds_raw[i]
        
        # If cooling increases but PMV increases (should decrease), that's wrong
        if cp_change > 0.1 and pmv_change > 0.2:
            violations.append(i)
    
    return len(violations)



# MODEL BUILDERS


def build_lstm_model(input_shape):
    """Standard LSTM with functional API."""
    inputs = Input(shape=input_shape, name='main_input')
    x = LSTM(128, return_sequences=True, dropout=0.2, name='lstm_1')(inputs)
    x = BatchNormalization()(x)
    x = LSTM(64, return_sequences=True, dropout=0.2, name='lstm_2')(x)
    x = BatchNormalization()(x)
    x = LSTM(32, return_sequences=False, name='lstm_3')(x)
    x = Dropout(0.2)(x)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)
    outputs = Dense(1, activation='linear')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_Model')
    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='mse', metrics=['mae'])
    return model




# TRAINING FUNCTIONS


def train_lstm(X_train, y_train, X_test, y_test):
    """Standard Keras 2 training loop."""
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    # 1. Manual Validation Split (Standard practice for manual 70/30)
    if (X_test is not None and len(X_test) > 0):
        val_data = (X_test, y_test)
        X_train_final, y_train_final = X_train, y_train
    else:
        # 10% split if no test file provided
        split_idx = int(len(X_train) * 0.9)
        if split_idx == 0: split_idx = 1
        X_train_final, X_val = X_train[:split_idx], X_train[split_idx:]
        y_train_final, y_val = y_train[:split_idx], y_train[split_idx:]
        val_data = (X_val, y_val)

    # 2. Setup Callbacks
    monitor_metric = 'val_loss' if val_data is not None else 'loss'
    es = EarlyStopping(monitor=monitor_metric, patience=PATIENCE, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=5, min_lr=1e-5)
    
    # 3. Fit (Standard Keras 2 approach)
    history = model.fit(
        X_train_final, y_train_final, 
        validation_data=val_data,
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        callbacks=[es, rlr], 
        shuffle=False, # Standard for time-series windows in Keras 2
        verbose=1
    )
    
    return model, history




# PREDICTION FUNCTIONS


def predict_lstm(model, X_window, feat_scaler, pmv_scaler, last_12_raw, new_input_raw):
    """
    Predict PMV using LSTM with robust shape handling.
    """
    # Defensive check: ensure last_12_raw has WINDOW rows
    if len(last_12_raw) < WINDOW:
        # Pad with the first row if insufficient data
        padding = np.tile(last_12_raw[0:1], (WINDOW - len(last_12_raw), 1))
        last_12_raw = np.vstack([padding, last_12_raw])
    elif len(last_12_raw) > WINDOW:
        last_12_raw = last_12_raw[-WINDOW:]

    # Scale and update window
    last_12_scaled = feat_scaler.transform(last_12_raw)
    window = last_12_scaled.copy()
    
    new_input_scaled = feat_scaler.transform(np.array([new_input_raw], dtype='float32'))[0]
    window[-1] = new_input_scaled
    
    # Predict using direct call (more stable for shapes in Keras 3)
    window_input = window.reshape(1, WINDOW, len(FEATURES)).astype('float32')
    
    if hasattr(model, '__call__'):
        # Direct call is often better than .predict() for small batches in late TF
        pred_tensor = model(window_input, training=False)
        pred_scaled = pred_tensor.numpy()[0][0]
    else:
        # Fallback for non-keras models (if any)
        pred_scaled = model.predict(window_input)[0][0]
        
    pred_raw = pmv_scaler.inverse_transform([[pred_scaled]])[0][0]

