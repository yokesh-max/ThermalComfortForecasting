"""
pcdl_model.py
─────────────
Physics-Constrained Deep Learning (PCDL) model for HVAC Thermal Comfort (PMV)
forecasting.

Architecture:
  Input → LSTM(128) → Dropout → LSTM(64) → Dropout → Dense(32)
        → Dense(1) → PhysicsLossLayer → Output

The PhysicsLossLayer adds a soft physics penalty during training so that
predictions remain consistent with thermodynamic principles
(e.g. higher cooling power → lower PMV).

Usage:
    from pcdl_model import build_pcdl_model, train_pcdl
    from pcdl_model import save_pcdl_bundle, load_pcdl_bundle
"""

import os
import joblib
import numpy as np
import tensorflow as tf
import random

from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam

from hvac_models import WINDOW, EPOCHS, BATCH_SIZE, PATIENCE, SAVE_DIR

# Fix random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)


# ── PHYSICS LOSS LAYER ─────────────────────────────────────────────────────────

class PhysicsLossLayer(tf.keras.layers.Layer):
    """
    Custom Keras layer that adds a physics-based regularisation penalty.

    The penalty discourages predictions that violate the expected relationship
    between HVAC sensor inputs and PMV (e.g. more cooling → lower PMV).

    Args:
        weight  (float): Strength of the physics penalty (default 0.08).
        pmv_min (float): Minimum PMV for scaling (default -3.0).
        pmv_max (float): Maximum PMV for scaling (default  3.0).
    """

    def __init__(self, weight=0.08, pmv_min=-3.0, pmv_max=3.0, **kwargs):
        super().__init__(**kwargs)
        self.weight  = weight
        self.pmv_min = pmv_min
        self.pmv_max = pmv_max

    def call(self, inputs_tuple):
        features, predictions = inputs_tuple

        # Extract last-timestep features and clamp to [0, 1]
        cooling = tf.clip_by_value(features[:, -1, 0], 0.0, 1.0)
        flow    = tf.clip_by_value(features[:, -1, 1], 0.0, 1.0)
        temp    = tf.clip_by_value(features[:, -1, 3], 0.0, 1.0)
        co2     = tf.clip_by_value(features[:, -1, 4], 0.0, 1.0)
        rh      = tf.clip_by_value(features[:, -1, 6], 0.0, 1.0)

        # Physics-based expected PMV approximation
        expected_pmv = (
            1.1
            - 1.4 * cooling
            - 0.4 * flow
            + 0.6 * temp
            + 0.25 * co2
            + 0.2 * rh
        )
        expected_pmv = tf.clip_by_value(expected_pmv, -3.0, 3.0)

        # Scale expected PMV to [0, 1] to match model output
        expected_pmv_sc = (expected_pmv - self.pmv_min) / (self.pmv_max - self.pmv_min + 1e-10)
        expected_pmv_sc = tf.expand_dims(expected_pmv_sc, -1)

        # Physics penalty
        physics_penalty = tf.reduce_mean(tf.square(predictions - expected_pmv_sc))

        # Guard against NaN before adding loss
        safe_penalty = tf.where(
            tf.math.is_finite(physics_penalty),
            physics_penalty,
            tf.zeros_like(physics_penalty)
        )
        self.add_loss(self.weight * safe_penalty)

        return predictions

    def get_config(self):
        config = super().get_config()
        config.update({
            "weight":  self.weight,
            "pmv_min": self.pmv_min,
            "pmv_max": self.pmv_max,
        })
        return config

    def compute_output_shape(self, input_shape):
        return input_shape[1]


# ── CUSTOM LOSS ────────────────────────────────────────────────────────────────

def _nan_safe_mse(y_true, y_pred):
    """MSE that replaces NaN/Inf with 0 before computing, preventing domain errors."""
    y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))
    return tf.reduce_mean(tf.square(y_true - y_pred))


# ── MODEL BUILDER ──────────────────────────────────────────────────────────────

def build_pcdl_model(input_shape, pmv_min=-3.0, pmv_max=3.0):
    """
    Build the Physics-Constrained Deep Learning (PCDL) model.

    Architecture uses a layered LSTM backbone with a PhysicsLossLayer
    that enforces thermodynamic plausibility during training.

    Args:
        input_shape (tuple): (timesteps, features) e.g. (12, 8).
        pmv_min     (float): Lower bound of PMV range for scaling.
        pmv_max     (float): Upper bound of PMV range for scaling.

    Returns:
        tf.keras.Model: Compiled PCDL model.
    """
    inputs = Input(shape=input_shape)

    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.1)(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dropout(0.05)(x)
    x = Dense(32, activation='relu')(x)

    outputs_raw = Dense(1, activation='linear')(x)

    # Physics Integration
    outputs = PhysicsLossLayer(
        weight=0.08, pmv_min=pmv_min, pmv_max=pmv_max
    )([inputs, outputs_raw])

    model = Model(inputs, outputs, name='PCDL_Model')

    # Compile with gradient clipping to prevent exploding gradients
    model.compile(
        optimizer=Adam(learning_rate=0.0005, clipnorm=1.0),
        loss=_nan_safe_mse,
        metrics=['mae'],
    )
    return model


# ── TRAINING ───────────────────────────────────────────────────────────────────

def train_pcdl(X_train, y_train, X_test, y_test, pmv_min=-3.0, pmv_max=3.0):
    """
    Train the PCDL model with NaN sanitisation and early stopping.

    Args:
        X_train  (np.ndarray): Windowed training features shape (N, WINDOW, features).
        y_train  (np.ndarray): Scaled training targets    shape (N,).
        X_test   (np.ndarray): Windowed test features     shape (M, WINDOW, features).
        y_test   (np.ndarray): Scaled test targets        shape (M,).
        pmv_min  (float)     : Minimum PMV value for physics scaling.
        pmv_max  (float)     : Maximum PMV value for physics scaling.

    Returns:
        tuple: (trained model, Keras History object)
    """
    # Sanitize inputs — replace NaN/Inf
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0,  neginf=0.0)
    y_train = np.nan_to_num(y_train, nan=0.0, posinf=3.0,  neginf=-3.0)
    if X_test is not None and len(X_test) > 0:
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1.0,  neginf=0.0)
        y_test = np.nan_to_num(y_test, nan=0.0, posinf=3.0,  neginf=-3.0)

    model = build_pcdl_model(
        (X_train.shape[1], X_train.shape[2]),
        pmv_min=pmv_min, pmv_max=pmv_max
    )

    # Validation data setup
    if X_test is not None and len(X_test) > 0:
        val_data      = (X_test, y_test)
        X_train_final = X_train
        y_train_final = y_train
    else:
        # Fall back to 10 % internal split
        split_idx = max(1, int(len(X_train) * 0.9))
        X_train_final, X_val = X_train[:split_idx], X_train[split_idx:]
        y_train_final, y_val = y_train[:split_idx], y_train[split_idx:]
        val_data = (X_val, y_val)

    monitor_metric = 'val_loss' if val_data else 'loss'
    callbacks = [
        EarlyStopping(monitor=monitor_metric, patience=PATIENCE, restore_best_weights=True),
    ]

    history = model.fit(
        X_train_final, y_train_final,
        validation_data=val_data,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=False,   # preserve time-series order
        verbose=1,
    )

    return model, history


# ── SAVE / LOAD ────────────────────────────────────────────────────────────────

def save_pcdl_bundle(model, feat_scaler, pmv_scaler, save_dir=SAVE_DIR):
    """Save PCDL model and its scalers to disk."""
    os.makedirs(save_dir, exist_ok=True)
    joblib.dump({"feat": feat_scaler, "pmv": pmv_scaler},
                os.path.join(save_dir, "PCDL_scalers.pkl"))
    model.save(os.path.join(save_dir, "PCDL_model.keras"))
    return True


def load_pcdl_bundle(save_dir=SAVE_DIR):
    """Load PCDL model and scalers from disk."""
    scaler_path = os.path.join(save_dir, "PCDL_scalers.pkl")
    if not os.path.exists(scaler_path):
        return None, None

    scalers    = joblib.load(scaler_path)
    model_path = os.path.join(save_dir, "PCDL_model.keras")

    if os.path.exists(model_path):
        try:
            custom_objects = {
                'PhysicsLossLayer': PhysicsLossLayer,
                '_nan_safe_mse':    _nan_safe_mse,
            }
            model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            return model, scalers
        except Exception as e:
            print(f"PCDL load error: {e}")
            return None, scalers

    return None, scalers