import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
# (hm import removed)

# ── LOCAL CONFIGURATION (Self-contained) ──────────────────────────
HVAC_FEATURES = [
    'Cooling_Power', 'Flowrate', 'CHWR-CHWS', 'Offcoil_Temperature',
    'Return_air_Co2', 'Return_air_static_pressure', 'Return_air_RH'
]
WINDOW      = 12      # 12 steps × 5 min = 1 hour lookback
EPOCHS      = 100     # max epochs
BATCH_SIZE  = 16
PATIENCE    = 15      # early stopping patience

FEATURES = HVAC_FEATURES # Alias for internal code compatibility

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

def train_lstm(X_train, y_train, X_test, y_test):
    """Standard Keras 2 training loop."""
    model = build_lstm_model((X_train.shape[1], X_train.shape[2]))
    
    if (X_test is not None and len(X_test) > 0):
        val_data = (X_test, y_test)
        X_train_final, y_train_final = X_train, y_train
    else:
        split_idx = int(len(X_train) * 0.9)
        if split_idx == 0: split_idx = 1
        X_train_final, X_val = X_train[:split_idx], X_train[split_idx:]
        y_train_final, y_val = y_train[:split_idx], y_train[split_idx:]
        val_data = (X_val, y_val)

    monitor_metric = 'val_loss' if val_data is not None else 'loss'
    es = EarlyStopping(monitor=monitor_metric, patience=PATIENCE, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor=monitor_metric, factor=0.5, patience=5, min_lr=1e-5)
    
    history = model.fit(
        X_train_final, y_train_final, 
        validation_data=val_data,
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE,
        callbacks=[es, rlr], 
        shuffle=False,
        verbose=1
    )
    
    return model, history

def predict_lstm(model, X_window, feat_scaler, pmv_scaler, last_12_raw, new_input_raw):
    """Predict PMV using LSTM with robust shape handling."""
    if len(last_12_raw) < WINDOW:
        padding = np.tile(last_12_raw[0:1], (WINDOW - len(last_12_raw), 1))
        last_12_raw = np.vstack([padding, last_12_raw])
    elif len(last_12_raw) > WINDOW:
        last_12_raw = last_12_raw[-WINDOW:]

    last_12_scaled = feat_scaler.transform(last_12_raw)
    window = last_12_scaled.copy()
    
    # Wrap in array to match scaler expected format
    new_input_raw_arr = np.array([new_input_raw], dtype='float32')
    new_input_scaled = feat_scaler.transform(new_input_raw_arr)[0]
    window[-1] = new_input_scaled
    
    window_input = window.reshape(1, WINDOW, len(FEATURES)).astype('float32')
    
    if hasattr(model, '__call__'):
        pred_tensor = model(window_input, training=False)
        pred_scaled = pred_tensor.numpy()[0][0]
    else:
        pred_scaled = model.predict(window_input)[0][0]
        
    pred_raw = pmv_scaler.inverse_transform([[pred_scaled]])[0][0]
    return float(pred_raw)
