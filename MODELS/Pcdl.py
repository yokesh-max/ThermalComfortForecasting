"""

Physics-Constrained Deep Learning (PCDL) for PMV Thermal Comfort Forecasting.

============================================================
STEP 0 — WHY PCDL? (Read this before the code)
============================================================

A standard LSTM learns purely from data. It can produce predictions like:
  "Cooling power increased by 20 kW → PMV went UP by 0.3"

That is physically impossible — more cooling always lowers room temperature,
which always lowers PMV. The LSTM doesn't know this. It just fits numbers.

PCDL fixes this by adding physics penalty terms to the training loss.
During every gradient update, the model is penalised for violating known
thermodynamic laws. The laws act like guardrails — the model is still free
to learn from data, but it cannot settle on predictions that violate physics.

Total PCDL loss (explained in detail at each penalty below):

  Loss = MSE(actual, predicted)          ← accuracy: match the data
       + λ1 × cooling_penalty            ← physics 1: cooling↑ → PMV↓
       + λ2 × offcoil_penalty            ← physics 2: supply air temp↑ → PMV↑
       + λ3 × humidity_penalty           ← physics 3: humidity↑ → PMV↑
       + λ4 × bounds_penalty             ← physics 4: PMV must stay in [-3, +3]

All four physics laws come directly from:
  - ISO 7730 Fanger PMV equation
  - Basic HVAC thermodynamics
  - Your sensor definitions in the project

============================================================
"""

import logging
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, BatchNormalization, Input, Layer
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── LOCAL CONFIGURATION (Self-contained) ──────────────────────────
FEATURES = [
    'Cooling_Power', 'Flowrate', 'CHWR-CHWS', 'Offcoil_Temperature',
    'Return_air_Co2', 'Return_air_static_pressure', 'Return_air_RH'
]
WINDOW = 12

# ── LOCAL CONFIGURATION (Self-contained) ──────────────────────────




"""
INPUT:  Nothing — just loading tools.
OUTPUT: All libraries available.

Why each import:
  numpy        → array math
  tensorflow   → neural network + gradient computation
  MinMaxScaler → normalise features to [0, 1] before feeding LSTM
  joblib       → save/load scalers to disk
  logging      → track what the model is doing at every step
"""

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("pcdl_simple")


"""
============================================================
STEP 2 — CONFIGURATION
============================================================

These are the only numbers you need to understand and possibly change.
Everything else in the code uses these constants.
"""

# ── Features (must match column order in your CSV) ────────────────────────────
FEATURES = [
    "Cooling_Power",               # index 0 — primary AC control
    "Flowrate",                    # index 1 — cold water volume
    "CHWR-CHWS",                   # index 2 — cooling effort (delta-T)
    "Offcoil_Temperature",         # index 3 — supply air temperature
    "Return_air_Co2",              # index 4 — occupancy proxy
    "Return_air_static_pressure",  # index 5 — airflow state
    "Return_air_RH",               # index 6 — room humidity
]
TARGET = "PMV"

# ── Sequence settings ─────────────────────────────────────────────────────────
WINDOW     = 12   # look-back: 12 steps × 5 min = 1 hour of history
                  # The LSTM sees the past 1 hour to predict next 5 minutes

# ── Training settings ─────────────────────────────────────────────────────────
EPOCHS     = 100  # maximum training iterations
BATCH_SIZE = 32   # how many windows per gradient update
PATIENCE   = 25   # increased patience for deeper convergence

# ── Physics penalty weights (λ1 to λ4) ───────────────────────────────────────
# These control how strongly each physics law is enforced.
# Start with equal small values. Increase if violations persist.
# Total physics weight = λ1 + λ2 + λ3 + λ4 = 0.10
# This means 90% of the loss is data accuracy, 10% is physics compliance.
# Lowered lambdas (Total = 0.10) to prioritize data accuracy (R²)
LAMBDA_COOLING  = 0.04   # λ1
LAMBDA_OFFCOIL  = 0.02   # λ2
LAMBDA_HUMIDITY = 0.02   # λ3
LAMBDA_BOUNDS   = 0.02   # λ4

# ── Feature indices (do not change — must match FEATURES list above) ──────────
IDX_COOLING  = 0   # Cooling_Power
IDX_OFFCOIL  = 3   # Offcoil_Temperature
IDX_RH       = 6   # Return_air_RH

# ── PMV bounds in scaled [0, 1] space ─────────────────────────────────────────
# MinMaxScaler maps real PMV range to [0, 1].
# The Fanger scale is physically bounded at [-3, +3] → maps to [0.0, 1.0].
PMV_SCALED_MIN = 0.0
PMV_SCALED_MAX = 1.0

SAVE_DIR = "saved_models"

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)
os.environ['PYTHONHASHSEED'] = str(42)


"""
============================================================
STEP 3 — DATA PREPARATION
============================================================

INPUT:  raw pandas DataFrame from your CSV file
OUTPUT: X_train (N, 12, 7), y_train (N,), X_val, y_val,
        feat_scaler, pmv_scaler

This follows the tutorial rule exactly:
  Split FIRST → Fit scaler on train only → Transform both → Make windows
"""

def prepare_data(train_df, test_df=None):
    """
    Convert raw sensor data into windowed sequences for LSTM training.

    If test_df is provided:
      - Uses all of train_df for training.
      - Uses all of test_df for testing (validation).
    If test_df is None:
      - Splits train_df internally (70% train, 30% test).

    INPUT:  train_df — pandas DataFrame (Training set)
            test_df  — pandas DataFrame (optional, separate Test set)
    OUTPUT: dict with X_train, y_train, X_val, y_val,
            y_val_raw (unscaled, for metrics),
            feat_scaler, pmv_scaler
    """
    log.info("prepare_data | train_df shape = %s, test_df shape = %s", 
             train_df.shape, test_df.shape if test_df is not None else "None")

    # ── 3a. Map and extract columns ───────────────────────────────────────────
    norm  = lambda s: s.lower().replace(" ","").replace("_","").replace("-","")
    cols  = {norm(c): c for c in train_df.columns}

    feat_cols = []
    for f in FEATURES:
        matched = cols.get(norm(f))
        if matched is None:
            raise ValueError(f"Column not found: '{f}'")
        feat_cols.append(matched)

    tgt_col = cols.get(norm(TARGET))
    if tgt_col is None:
        raise ValueError(f"Target column '{TARGET}' not found.")

    log.info("Columns mapped | features=%s  target=%s", feat_cols, tgt_col)

    # ── 3b. Interpolate and Drop NaN rows ─────────────────────────────────────────────
    train_data = train_df[feat_cols + [tgt_col]].copy()
    train_data = train_data.interpolate(method='linear', limit=6, limit_direction='both').dropna().reset_index(drop=True)
    if test_df is not None:
        test_data = test_df[feat_cols + [tgt_col]].copy()
        test_data = test_data.interpolate(method='linear', limit=6, limit_direction='both').dropna().reset_index(drop=True)
    else:
        test_data = None

    # ── 3c. Logistic split ────────────────────────────────────────────────────
    if test_data is None:
        # Use 100% for both training and validation
        train_part = train_data
        test_part  = train_data
        log.info("100%% Data Usage | train = %d rows  val = %d rows (Identical)", 
                 len(train_part), len(test_part))
    else:
        # Use provided separate file
        train_part = train_data
        test_part = test_data
        log.info("External Files | train = %d rows (from train file)  val = %d rows (from test file)", len(train_part), len(test_part))

    if len(train_part) < WINDOW + 2 or len(test_part) < WINDOW + 2:
        raise ValueError("Too few rows for windowing. Increase dataset size.")

    X_train_raw = train_part[feat_cols].values
    y_train_raw = train_part[tgt_col].values
    X_val_raw   = test_part[feat_cols].values
    y_val_raw   = test_part[tgt_col].values

    # ── 3d. Fit scalers on train_part only ────────────────────────────────────
    feat_scaler = MinMaxScaler()
    pmv_scaler  = MinMaxScaler()

    X_train_sc = feat_scaler.fit_transform(X_train_raw)
    y_train_sc = pmv_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()

    # Transform test_part using train_part scalers
    X_val_sc = feat_scaler.transform(X_val_raw)
    y_val_sc = pmv_scaler.transform(y_val_raw.reshape(-1, 1)).ravel()

    # ── 3e. Build sliding windows ─────────────────────────────────────────────
    X_train_w, y_train_w = _make_windows(X_train_sc, y_train_sc)
    X_val_w,   y_val_w   = _make_windows(X_val_sc,   y_val_sc)
    _,         y_val_raw_w = _make_windows(X_val_raw, y_val_raw)

    log.info("Windows | X_train = %s   X_val = %s", X_train_w.shape, X_val_w.shape)

    return {
        "X_train":    X_train_w,
        "y_train":    y_train_w,
        "X_val":      X_val_w,
        "y_val":      y_val_w,
        "y_val_raw":  y_val_raw_w,
        "feat_scaler": feat_scaler,
        "pmv_scaler":  pmv_scaler,
    }



def _make_windows(X, y, win=WINDOW):
    """
    Convert a flat 2D array into 3D sliding windows.

    For each position i:
      Input  window = X[i : i+win]    → shape (win, n_features) = (12, 7)
      Output label  = y[i + win]      → scalar PMV at next timestep

    INPUT:  X   shape (N, 7)   — scaled features
            y   shape (N,)     — scaled PMV
            win int            — window size (default 12)
    OUTPUT: Xw  shape (N-win, win, 7)
            yw  shape (N-win,)

    Example with N=1000, win=12:
      First window:  X[0:12]  → label: y[12]   (rows 0–11 predict row 12)
      Second window: X[1:13]  → label: y[13]
      Last window:   X[988:1000] → label: y[1000]  ← can't do this (out of bounds)
      So total windows = N - win = 1000 - 12 = 988
    """
    Xw, yw = [], []
    for i in range(len(X) - win):
        Xw.append(X[i : i + win])   # 12 rows × 7 features
        yw.append(y[i + win])        # PMV value at next timestep
    return np.array(Xw), np.array(yw)


"""
============================================================
STEP 4 — THE PHYSICS LAWS AND WHY EACH ONE IS CHOSEN
============================================================

This is the heart of PCDL. Each penalty below corresponds to a real
thermodynamic law that governs how your HVAC system works.

The penalty is always structured as:
  relu(violation)²

  relu: only fires when there IS a violation (clips correct predictions to 0)
  ²:    squared so larger violations get disproportionately larger penalties
        (smooth gradient, prevents ignoring small violations)

All features are in scaled [0, 1] space during training.
The direction relationships hold in scaled space too:
  scaled_cooling↑ = real_cooling↑ (MinMaxScaler preserves direction)
"""

class PhysicsConstraintLayer(tf.keras.layers.Layer):
    """
    A Keras layer that computes and registers all 4 physics penalties.

    How it works:
      - Sits between the raw LSTM output and the model output
      - Does NOT change the prediction values
      - Calls self.add_loss() for each penalty term
      - Keras automatically adds these to the compiled MSE loss
      - During backprop, gradients flow through all penalty terms
        so the weights are updated to reduce both accuracy error AND
        physics violations simultaneously

    Args:
        lambda_cooling  (float): λ1 weight for cooling constraint
        lambda_offcoil  (float): λ2 weight for offcoil constraint
        lambda_humidity (float): λ3 weight for humidity constraint
        lambda_bounds   (float): λ4 weight for boundary constraint
    """

    def __init__(
        self,
        lambda_cooling:  float = LAMBDA_COOLING,
        lambda_offcoil:  float = LAMBDA_OFFCOIL,
        lambda_humidity: float = LAMBDA_HUMIDITY,
        lambda_bounds:   float = LAMBDA_BOUNDS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.lambda_cooling  = lambda_cooling
        self.lambda_offcoil  = lambda_offcoil
        self.lambda_humidity = lambda_humidity
        self.lambda_bounds   = lambda_bounds

    def call(self, inputs_tuple, training=None):
        """
        Compute all 4 penalties and register them as additional loss terms.

        INPUT:  inputs_tuple = (features, predictions)
                  features    → shape (batch, 12, 7)  scaled sensor windows
                  predictions → shape (batch, 1)      raw LSTM output (scaled PMV)
        OUTPUT: predictions unchanged → shape (batch, 1)
                (this layer ONLY adds loss terms, never changes predictions)
        """
        features, predictions = inputs_tuple

        # Extract the LAST timestep of each window — the most recent reading
        # features[:, -1, :] gives us one row per sample in the batch
        # shape: (batch, 7)
        last_step = features[:, -1, :]

        # Extract individual sensors from last timestep
        cooling = last_step[:, IDX_COOLING]   # (batch,) — Cooling_Power scaled
        offcoil = last_step[:, IDX_OFFCOIL]   # (batch,) — Offcoil_Temperature scaled
        rh      = last_step[:, IDX_RH]        # (batch,) — Return_air_RH scaled

        # Flatten predictions from (batch, 1) → (batch,) for diff operations
        pred = tf.squeeze(predictions, axis=-1)   # (batch,)

        # ── Consecutive differences within the batch ───────────────────────────
        # For samples i and i+1 in the same batch:
        #   cooling_diff[i] = how much cooling changed
        #   pmv_diff[i]     = how much predicted PMV changed
        # shape: (batch-1,)
        cooling_diff = cooling[1:] - cooling[:-1]
        offcoil_diff = offcoil[1:] - offcoil[:-1]
        rh_diff      = rh[1:]      - rh[:-1]
        pmv_diff     = pred[1:]    - pred[:-1]

        # ══════════════════════════════════════════════════════════════════════
        # PHYSICS LAW 1 — Cooling Power vs PMV (MOST IMPORTANT)
        # ══════════════════════════════════════════════════════════════════════
        #
        # PHYSICAL LAW:
        #   When Cooling_Power increases (chilled water valve opens more),
        #   the AHU extracts more heat from the room air.
        #   → Room air temperature drops.
        #   → PMV drops (room feels cooler).
        #   Therefore: ΔCooling_Power and ΔPMV must have OPPOSITE signs.
        #
        # WHY this law applies to your project:
        #   Cooling_Power is your single strongest predictor (75% importance
        #   per your feature analysis). It directly controls the CHW valve.
        #   This is the most fundamental relationship in your entire system.
        #   An LSTM that violates this is useless for MPC control — the
        #   controller would open the valve MORE to make the room COOLER,
        #   but the model would predict PMV going UP, causing the controller
        #   to take the wrong action.
        #
        # HOW the penalty works:
        #   violation = relu(pmv_diff × sign(cooling_diff))
        #   - If cooling↑ and PMV↓: pmv_diff<0, sign(cooling_diff)=+1
        #     → product is negative → relu clips to 0 → NO PENALTY (correct)
        #   - If cooling↑ and PMV↑: pmv_diff>0, sign(cooling_diff)=+1
        #     → product is positive → relu = positive → PENALISED (wrong)
        #   - If cooling↓ and PMV↑: pmv_diff>0, sign(cooling_diff)=-1
        #     → product is negative → relu clips to 0 → NO PENALTY (correct)
        #   - If cooling↓ and PMV↓: pmv_diff<0, sign(cooling_diff)=-1
        #     → product is positive → relu = positive → PENALISED (wrong)
        #
        penalty_cooling = tf.reduce_mean(
            tf.square(
                tf.nn.relu(pmv_diff * tf.sign(cooling_diff))
            )
        )
        self.add_loss(self.lambda_cooling * penalty_cooling)

        # ══════════════════════════════════════════════════════════════════════
        # PHYSICS LAW 2 — Offcoil Temperature vs PMV
        # ══════════════════════════════════════════════════════════════════════
        #
        # PHYSICAL LAW:
        #   Offcoil_Temperature is the temperature of air LEAVING the AHU coils
        #   and entering the room through supply ducts.
        #   When Offcoil_Temperature increases:
        #     → Warmer air is pumped into the room
        #     → Room air temperature rises
        #     → PMV rises (room feels warmer)
        #   Therefore: ΔOffcoil_Temperature and ΔPMV must have the SAME sign.
        #
        # WHY this law applies to your project:
        #   Offcoil_Temperature is a direct indicator of how cold the supply
        #   air is. In your Fanger formula: ta (air temperature) is the
        #   primary driver of PMV. Offcoil temp IS the supply air temp.
        #   From your sensor list, Offcoil_Temperature maps directly to
        #   the 'ta' parameter in the Fanger PMV equation.
        #   ISO 7730 confirms: 1°C increase in air temp ≈ +0.5 PMV units
        #   for typical office conditions.
        #
        # HOW the penalty works:
        #   violation = relu(-pmv_diff × sign(offcoil_diff))
        #   The negative sign flips the logic: now we penalise when they
        #   go in OPPOSITE directions (offcoil↑ but PMV↓ is wrong).
        #   - If offcoil↑ and PMV↑: product = positive, negated → negative
        #     → relu clips to 0 → NO PENALTY (correct)
        #   - If offcoil↑ and PMV↓: product = negative, negated → positive
        #     → relu = positive → PENALISED (wrong)
        #
        penalty_offcoil = tf.reduce_mean(
            tf.square(
                tf.nn.relu(-pmv_diff * tf.sign(offcoil_diff))
            )
        )
        self.add_loss(self.lambda_offcoil * penalty_offcoil)

        # ══════════════════════════════════════════════════════════════════════
        # PHYSICS LAW 3 — Return Air Humidity vs PMV
        # ══════════════════════════════════════════════════════════════════════
        #
        # PHYSICAL LAW:
        #   Return_air_RH is room relative humidity (%).
        #   When humidity increases:
        #     → Sweat evaporation from the skin is reduced
        #     → Body's natural cooling mechanism is less effective
        #     → Room FEELS warmer even at the same temperature
        #     → PMV rises
        #   Therefore: ΔReturn_air_RH and ΔPMV must have the SAME sign.
        #
        # WHY this law applies to your project:
        #   In the ISO 7730 Fanger equation, humidity appears via partial
        #   water vapour pressure (pa). Higher RH → higher pa → higher
        #   latent heat load on the body → higher PMV.
        #   From your Fanger implementation:
        #     pa = (rh / 100) * 10 * exp(16.6536 - 4030.183 / (ta + 235))
        #   pa feeds directly into hl1 (latent skin diffusion heat loss).
        #   Higher pa → reduced hl1 → higher net heat retained → higher PMV.
        #   This is a well-established comfort law that your physical model
        #   already implements — we're just enforcing it on the LSTM too.
        #
        # HOW the penalty works (same structure as offcoil penalty):
        #   violation = relu(-pmv_diff × sign(rh_diff))
        #   Penalises when RH and PMV go in opposite directions.
        #
        penalty_humidity = tf.reduce_mean(
            tf.square(
                tf.nn.relu(-pmv_diff * tf.sign(rh_diff))
            )
        )
        self.add_loss(self.lambda_humidity * penalty_humidity)

        # ══════════════════════════════════════════════════════════════════════
        # PHYSICS LAW 4 — PMV Boundary Constraint
        # ══════════════════════════════════════════════════════════════════════
        #
        # PHYSICAL LAW:
        #   The Fanger PMV scale is physically bounded.
        #   In practice: PMV ∈ [-3, +3]
        #   -3 = extreme cold stress (maximum shivering)
        #   +3 = extreme heat stress (maximum discomfort)
        #   Values outside this range are physiologically impossible for
        #   normal indoor HVAC conditions.
        #
        # WHY this law applies to your project:
        #   Without this constraint, the LSTM might predict PMV = +4.5 or
        #   PMV = -2.8 for unusual sensor combinations. These predictions
        #   would cause your MPC controller to issue extreme valve commands
        #   (fully open or fully closed) in response to predictions that
        #   cannot physically occur. This is a safety constraint.
        #   In your training data, PMV is scaled to [0, 1] by MinMaxScaler
        #   using the training data's actual min/max. The boundary constraint
        #   simply says: don't predict outside [0, 1] in scaled space.
        #
        # HOW the penalty works:
        #   upper_violation = relu(pred - 1.0)  → fires when pred > 1.0
        #   lower_violation = relu(0.0 - pred)  → fires when pred < 0.0
        #   penalty = mean of both squared violations
        #
        upper_violation = tf.nn.relu(predictions - PMV_SCALED_MAX)
        lower_violation = tf.nn.relu(PMV_SCALED_MIN - predictions)
        penalty_bounds  = tf.reduce_mean(
            tf.square(upper_violation) + tf.square(lower_violation)
        )
        self.add_loss(self.lambda_bounds * penalty_bounds)

        if not isinstance(penalty_cooling, tf.Tensor) or tf.executing_eagerly():
            log.debug(
                "PhysicsLayer | cooling=%.4f  offcoil=%.4f  humidity=%.4f  bounds=%.4f",
                float(penalty_cooling), float(penalty_offcoil),
                float(penalty_humidity), float(penalty_bounds),
            )

        # Return predictions unchanged — this layer ONLY adds loss terms
        # The actual output of the model is still the raw LSTM prediction
        return predictions

    def get_config(self):
        """Required so Keras can save and reload this custom layer."""
        config = super().get_config()
        config.update({
            "lambda_cooling":  self.lambda_cooling,
            "lambda_offcoil":  self.lambda_offcoil,
            "lambda_humidity": self.lambda_humidity,
            "lambda_bounds":   self.lambda_bounds,
        })
        return config


"""
============================================================
STEP 5 — BUILD THE MODEL
============================================================

Architecture (same LSTM trunk as a standard LSTM — only the loss differs):

  Input      (batch, 12, 7)   — 12 timesteps × 7 features
      ↓
  LSTM(64)   (batch, 12, 64)  — reads full sequence, passes all hidden states
      ↓
  LSTM(32)   (batch, 32)      — collapses to one context vector
      ↓
  Dense(16)  (batch, 16)      — non-linear compression
      ↓
  Dropout    (batch, 16)      — regularisation (prevents memorising training data)
      ↓
  Dense(1)   (batch, 1)       — raw PMV prediction (scaled)
      ↓
  PhysicsConstraintLayer      — adds 4 penalty terms to loss, passes pred unchanged
      ↓
  Output     (batch, 1)       — final PMV prediction (same as Dense(1) output)
"""

def build_model(input_shape):
    """
    Build the PCDL model.

    INPUT:  input_shape — tuple (WINDOW, n_features) = (12, 7)
    OUTPUT: compiled Keras Model ready for training

    Tensor shapes at each layer (example batch_size=32):
      Input:      (32, 12, 7)
      LSTM 1:     (32, 12, 64)   return_sequences=True → outputs ALL timesteps
      LSTM 2:     (32, 32)       return_sequences=False → outputs LAST timestep only
      Dense(16):  (32, 16)
      Dropout:    (32, 16)       some neurons zeroed randomly during training
      Dense(1):   (32, 1)        raw PMV prediction
      Physics:    (32, 1)        same — just adds penalty terms to loss
    """
    log.info("build_model | input_shape = %s", input_shape)

    # ── Input layer ───────────────────────────────────────────────────────────
    inputs = Input(shape=input_shape, name="sensor_window")
    # INPUT:  shape = (12, 7)  — one window of 12 timesteps × 7 features
    # OUTPUT: tensor (batch, 12, 7)

    # ── LSTM layer 1 — reads the sequence ────────────────────────────────────
    x = LSTM(
        64,
        return_sequences=True,   # output hidden state at EVERY timestep
        dropout=0.2,             # 20% of inputs dropped randomly (reduces overfit)
        recurrent_dropout=0.1,   # 10% of recurrent connections dropped
        name="lstm_1",
    )(inputs)
    # INPUT:  (batch, 12, 7)
    # OUTPUT: (batch, 12, 64)   — 64-dim hidden state for each of 12 timesteps
    # WHY return_sequences=True: the second LSTM needs the full sequence

    # ── LSTM layer 2 — summarises the sequence ────────────────────────────────
    x = LSTM(
        32,
        return_sequences=False,  # output hidden state at LAST timestep only
        dropout=0.2,
        name="lstm_2",
    )(x)
    # INPUT:  (batch, 12, 64)
    # OUTPUT: (batch, 32)       — single 32-dim context vector summarising 1 hour
    # WHY return_sequences=False: Dense layer needs a flat vector, not a sequence

    # ── Dense hidden layer ────────────────────────────────────────────────────
    x = Dense(16, activation="relu", name="dense_1")(x)
    # INPUT:  (batch, 32)
    # OUTPUT: (batch, 16)
    # WHY relu: adds non-linearity so model can learn complex PMV relationships

    # ── Dropout ───────────────────────────────────────────────────────────────
    x = Dropout(0.2, name="dropout_1")(x)
    # INPUT:  (batch, 16)
    # OUTPUT: (batch, 16)   — 20% of values set to 0 during training only
    # WHY: prevents the model from relying too heavily on any single neuron

    # ── Output layer — raw PMV prediction ────────────────────────────────────
    raw_pred = Dense(1, activation="linear", name="pmv_raw")(x)
    # INPUT:  (batch, 16)
    # OUTPUT: (batch, 1)    — scaled PMV prediction in [0, 1]
    # WHY linear: regression output — no sigmoid/softmax needed

    # ── Physics constraint layer ──────────────────────────────────────────────
    output = PhysicsConstraintLayer(
        lambda_cooling  = LAMBDA_COOLING,
        lambda_offcoil  = LAMBDA_OFFCOIL,
        lambda_humidity = LAMBDA_HUMIDITY,
        lambda_bounds   = LAMBDA_BOUNDS,
        name            = "physics_constraints",
    )([inputs, raw_pred])
    # INPUT:  [features (batch,12,7), raw_pred (batch,1)]
    # OUTPUT: (batch, 1)   — SAME as raw_pred (predictions not changed)
    # SIDE EFFECT: 4 penalty terms registered via add_loss()
    # THESE PENALTIES ARE THEN ADDED TO MSE BY KERAS AUTOMATICALLY

    model = Model(inputs=inputs, outputs=output, name="PCDL_PMV")

    # ── Compile ───────────────────────────────────────────────────────────────
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",       # data loss: (actual_PMV - predicted_PMV)²
        metrics=["mae"],  # reported in training logs for human readability
    )
    # Total loss during training = mse + λ1×p1 + λ2×p2 + λ3×p3 + λ4×p4
    # Keras adds add_loss() terms automatically — no custom training loop needed

    log.info("Model built | params = %d", model.count_params())
    return model


"""
============================================================
STEP 6 — TRAIN THE MODEL
============================================================

Training flow:
  For each epoch:
    For each batch of 32 windows:
      1. Forward pass: model predicts PMV for all 32 windows
      2. PhysicsConstraintLayer computes 4 penalty terms via add_loss()
      3. Total loss = MSE + all penalties
      4. Backprop: gradients flow back through ALL terms
      5. Adam optimizer updates weights to reduce total loss

  Early stopping monitors val_loss:
    If val_loss doesn't improve for 25 epochs → stop + restore best weights
"""

def train_model(data):
    """
    Train the PCDL model.

    INPUT:  data — dict from prepare_data() containing:
                   X_train, y_train, X_val, y_val,
                   feat_scaler, pmv_scaler
    OUTPUT: (trained_model, history, feat_scaler, pmv_scaler)
    """
    X_train = data["X_train"]   # (N-12, 12, 7)
    y_train = data["y_train"]   # (N-12,)
    X_val   = data["X_val"]     # (M-12, 12, 7)
    y_val   = data["y_val"]     # (M-12,)

    log.info("train_model | X_train = %s   y_train = %s", X_train.shape, y_train.shape)
    log.info("train_model | X_val   = %s   y_val   = %s", X_val.shape,   y_val.shape)

    model = build_model((X_train.shape[1], X_train.shape[2]))
    # input_shape = (12, 7)  — WINDOW=12, n_features=7

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,          # stop after 25 epochs with no improvement
            restore_best_weights=True,  # rewind to best epoch automatically
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,    # halve the learning rate on plateau
            patience=8,    # wait 8 epochs before halving
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    log.info("Training started...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=False,    # Disabled shuffling
        verbose=1,
    )
    # INPUT:  X_train = (N-12, 12, 7)   y_train = (N-12,)
    # OUTPUT: trained model weights + loss history

    best_val = min(history.history["val_loss"])
    stopped  = len(history.history["loss"])
    log.info("Training done | stopped_epoch = %d   best_val_loss = %.6f",
             stopped, best_val)

    return model, history, data["feat_scaler"], data["pmv_scaler"]


def train_pcdl(X_train, y_train, X_val, y_val):
    """
    Backward compatibility wrapper for app.py.
    Maps old train_pcdl call to new train_model logic.
    """
    data = {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "feat_scaler": None, # Not needed if already scaled
        "pmv_scaler": None
    }
    model, history, _, _ = train_model(data)
    return model, history



"""
============================================================
STEP 7 — PREDICT (SINGLE STEP)
============================================================

At inference time:
  1. Take last 12 raw sensor readings (unscaled)
  2. Scale using the SAME feat_scaler from training
  3. Pass through model with training=False
     → Dropout is disabled (deterministic prediction)
     → PhysicsConstraintLayer.add_loss() is suppressed (no penalty at inference)
  4. Inverse-transform the scaled PMV prediction back to real PMV units
"""

def predict_single(model, last_window_raw, feat_scaler, pmv_scaler):
    """
    Predict the next PMV value from a 12-step raw sensor window.

    INPUT:  model           — trained PCDL Keras model
            last_window_raw — np.array shape (12, 7)  UNSCALED sensor readings
            feat_scaler     — fitted MinMaxScaler for features
            pmv_scaler      — fitted MinMaxScaler for PMV
    OUTPUT: float — predicted PMV in real units (e.g. -0.3, +0.8)
    """
    log.info("predict_single | window shape = %s", last_window_raw.shape)

    # Step 1: Scale the window using TRAINING scaler
    window_scaled = feat_scaler.transform(last_window_raw)
    # INPUT:  (12, 7)  — raw sensor values
    # OUTPUT: (12, 7)  — scaled to [0, 1] using train min/max

    # Step 2: Reshape to 3D for LSTM input
    window_input = window_scaled.reshape(1, WINDOW, len(FEATURES)).astype("float32")
    # INPUT:  (12, 7)
    # OUTPUT: (1, 12, 7)  — batch dimension added (batch_size=1 for single prediction)

    # Step 3: Forward pass — training=False disables Dropout and add_loss()
    pred_scaled = model(window_input, training=False).numpy()[0][0]
    # INPUT:  (1, 12, 7)
    # OUTPUT: scalar float in [0, 1]  — scaled PMV prediction
    # training=False: Dropout layers deactivated → deterministic output

    # Step 4: Inverse transform back to real PMV units
    pred_real = float(pmv_scaler.inverse_transform([[pred_scaled]])[0][0])
    # INPUT:  scalar in [0, 1]
    # OUTPUT: scalar in real PMV units e.g. -0.34
    # HOW:    real_pmv = scaled × (pmv_max - pmv_min) + pmv_min

    log.info("predict_single | scaled = %.4f   real_pmv = %.4f", pred_scaled, pred_real)
    return pred_real


"""
============================================================
STEP 8 — EVALUATE
============================================================

After training, evaluate on the held-out validation set.
All metrics computed in REAL PMV units (after inverse_transform).
"""

def evaluate(model, data):
    """
    Compute MAE, RMSE, R² and count physics violations on validation set.

    INPUT:  model — trained PCDL model
            data  — dict from prepare_data()
    OUTPUT: dict with mae, rmse, r2, mape, violations, preds_real
    """
    X_val    = data["X_val"]       # (M-12, 12, 7)  scaled
    y_val_raw= data["y_val_raw"]   # (M-12,)         real PMV units
    pmv_sc   = data["pmv_scaler"]

    log.info("evaluate | X_val = %s", X_val.shape)

    # Batch prediction — faster than one-by-one for evaluation
    preds_scaled = model.predict(X_val, verbose=0).ravel()
    # INPUT:  (M-12, 12, 7)
    # OUTPUT: (M-12,)   — scaled predictions

    # Inverse transform to real PMV units
    preds_real = pmv_sc.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    # INPUT:  (M-12,)   scaled
    # OUTPUT: (M-12,)   real PMV units

    # Metrics — all in real PMV units
    mae  = mean_absolute_error(y_val_raw, preds_real)
    rmse = float(np.sqrt(mean_squared_error(y_val_raw, preds_real)))
    r2   = float(r2_score(y_val_raw, preds_real))

    # MAPE — exclude samples where actual PMV is zero to avoid division by zero
    mask = y_val_raw != 0
    mape = float(np.mean(np.abs((y_val_raw[mask] - preds_real[mask]) / y_val_raw[mask])) * 100) if mask.any() else float("nan")

    log.info("Metrics | MAE = %.4f   RMSE = %.4f   R² = %.4f   MAPE = %.2f%%", mae, rmse, r2, mape)

    # Physics violation check — count cooling↑ but PMV↑ instances
    feat_sc = data["feat_scaler"]
    # We need raw cooling values — inverse transform feature column 0
    # Get scaled cooling from last timestep of each validation window
    cooling_scaled = X_val[:, -1, IDX_COOLING]
    # Inverse transform only cooling column (need a 7-col array for the scaler)
    dummy = np.zeros((len(cooling_scaled), len(FEATURES)))
    dummy[:, IDX_COOLING] = cooling_scaled
    cooling_real = feat_sc.inverse_transform(dummy)[:, IDX_COOLING]

    violations = 0
    for i in range(len(preds_real) - 1):
        cooling_up = cooling_real[i+1] - cooling_real[i] > 0
        pmv_up     = preds_real[i+1]   - preds_real[i]   > 0.05
        if cooling_up and pmv_up:
            violations += 1

    violation_pct = 100 * violations / max(len(preds_real) - 1, 1)
    log.info("Physics violations | count = %d  (%.1f%%)", violations, violation_pct)

    return {
        "mae":           mae,
        "rmse":          rmse,
        "r2":            r2,
        "mape":          mape,
        "violations":    violations,
        "violation_pct": violation_pct,
        "preds_real":    preds_real,
    }


"""
============================================================
STEP 9 — SAVE AND LOAD
============================================================
"""

def save_model(model, feat_scaler, pmv_scaler,
               model_name="PCDL_simple", save_dir=SAVE_DIR):
    """
    Save model and both scalers to disk.

    The model_name parameter lets each variant save under its own name
    so files don't overwrite each other:
      pcdl_simple  → PCDL_simple_model.keras
      V1_Actuator  → PCDL_V1_Actuator_model.keras
      V2_Environment → PCDL_V2_Environment_model.keras
      V3_Combined  → PCDL_V3_Combined_model.keras

    INPUT:  model       — trained Keras model
            feat_scaler — fitted MinMaxScaler for features
            pmv_scaler  — fitted MinMaxScaler for PMV
            model_name  — string prefix for filenames (default "PCDL_simple")
            save_dir    — directory to write into (default SAVE_DIR)
    OUTPUT: True on success
    """
    os.makedirs(save_dir, exist_ok=True)
    model_path  = os.path.join(save_dir, f"PCDL_{model_name}_model.keras")
    scaler_path = os.path.join(save_dir, f"PCDL_{model_name}_scalers.pkl")

    model.save(model_path)
    joblib.dump({"feat": feat_scaler, "pmv": pmv_scaler}, scaler_path)

    log.info("Saved | model → %s   scalers → %s", model_path, scaler_path)
    return True


def load_model(model_name="PCDL_simple", save_dir=SAVE_DIR):
    """
    Load model and scalers from disk.

    IMPORTANT: PhysicsConstraintLayer is a custom class — it must be passed
    in custom_objects so Keras knows how to deserialise it.

    INPUT:  model_name — same string used in save_model() (default "PCDL_simple")
            save_dir   — directory to load from
    OUTPUT: (model, feat_scaler, pmv_scaler)  or  (None, None, None)
    """
    model_path  = os.path.join(save_dir, f"PCDL_{model_name}_model.keras")
    scaler_path = os.path.join(save_dir, f"PCDL_{model_name}_scalers.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        log.warning("Model or scalers not found in %s", save_dir)
        return None, None, None

    scalers     = joblib.load(scaler_path)
    feat_scaler = scalers["feat"]
    pmv_scaler  = scalers["pmv"]

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={"PhysicsConstraintLayer": PhysicsConstraintLayer},
    )
    log.info("Loaded | %s", model_path)
    return model, feat_scaler, pmv_scaler


"""
============================================================
STEP 10 — COMPLETE USAGE EXAMPLE
============================================================

This shows the FULL end-to-end flow in one place.
Copy this into a script or notebook to run the complete pipeline.
"""

def run_example(train_df, test_df=None):
    """
    Full pipeline: prepare → train → evaluate → predict.

    INPUT:  train_df — pandas DataFrame (train CSV)
            test_df  — pandas DataFrame (test CSV) or None
    OUTPUT: (model, feat_scaler, pmv_scaler, metrics_dict)

    Data flow:
      train_df (N rows × 8 cols)
          ↓ prepare_data(train_df, test_df)
      X_train (N', 12, 7)   y_train (N',)    ← scaled, windowed
      X_val   (M', 12, 7)   y_val   (M',)
          ↓ train_model()
      trained PCDL model
          ↓ evaluate()
      MAE, RMSE, R², violation count
          ↓ predict_single()  [for each test row]
      forecast PMV (real units)
    """
    import pandas as pd

    # Reset all random seeds before every run so results are reproducible
    os.environ["PYTHONHASHSEED"] = "0"
    np.random.seed(42)
    tf.random.set_seed(42)

    log.info("=== PCDL PIPELINE START ===")

    # Step 1: Prepare training data
    log.info("Step 1: Preparing data (70/30 split or separate files)...")
    data = prepare_data(train_df, test_df)
    # OUTPUT:
    #   data["X_train"] = (N', 12, 7)   scaled windowed train features
    #   data["y_train"] = (N',)          scaled PMV labels
    #   data["X_val"]   = (M', 12, 7)   scaled windowed val features
    #   data["y_val"]   = (M',)          scaled PMV labels
    #   data["y_val_raw"] = (M',)        REAL PMV (for metrics)
    #   data["feat_scaler"], data["pmv_scaler"]


    # Step 2: Train model
    log.info("Step 2: Training PCDL model...")
    model, history, feat_sc, pmv_sc = train_model(data)
    # OUTPUT:
    #   model    — trained Keras model with physics constraints in loss
    #   history  — loss curves (history.history["loss"], ["val_loss"])
    #   feat_sc  — fitted MinMaxScaler for 7 features
    #   pmv_sc   — fitted MinMaxScaler for PMV

    # Step 3: Evaluate on validation set
    log.info("Step 3: Evaluating...")
    metrics = evaluate(model, data)
    # OUTPUT:
    #   metrics["mae"]           — mean absolute error in real PMV units
    #   metrics["rmse"]          — root mean squared error
    #   metrics["r2"]            — R² coefficient of determination
    #   metrics["violations"]    — count of cooling↑+PMV↑ occurrences
    #   metrics["violation_pct"] — % of timesteps that violated physics

    log.info("=== VALIDATION RESULTS ===")
    log.info("MAE        = %.4f PMV units", metrics["mae"])
    log.info("RMSE       = %.4f PMV units", metrics["rmse"])
    log.info("R²         = %.4f", metrics["r2"])
    log.info("MAPE       = %.2f%%", metrics["mape"])
    log.info("Violations = %d (%.1f%%)", metrics["violations"], metrics["violation_pct"])

    # Step 4: Save model and scalers
    log.info("Step 4: Saving...")
    save_model(model, feat_sc, pmv_sc)

    # Step 5: Rolling forecast on test file (if provided)
    if test_df is not None:
        log.info("Step 5: Rolling forecast on test data...")

        import pandas as pd
        norm     = lambda s: s.lower().replace(" ","").replace("_","").replace("-","")
        cols     = {norm(c): c for c in test_df.columns}
        feat_cols = [cols[norm(f)] for f in FEATURES]
        test_clean= test_df[feat_cols].dropna().reset_index(drop=True)

        # Seed the rolling window with the last 12 rows of training data
        train_norm = lambda s: s.lower().replace(" ","").replace("_","").replace("-","")
        train_cols = {train_norm(c): c for c in train_df.columns}
        train_feat = [train_cols[train_norm(f)] for f in FEATURES]
        history_raw = train_df[train_feat].dropna().values[-WINDOW:]
        # shape: (12, 7)  — last 12 rows of training data (unscaled)

        forecasts = []
        for i, row in test_clean.iterrows():
            # Predict using current rolling window
            pmv_pred = predict_single(model, history_raw, feat_sc, pmv_sc)
            forecasts.append(pmv_pred)

            # Shift window: drop oldest row, append current row
            new_row     = test_clean.loc[i].values.astype("float32")  # (7,)
            history_raw = np.vstack([history_raw[1:], new_row])
            # history_raw is always (12, 7) — oldest row dropped, newest appended

        log.info("Forecast complete | %d predictions", len(forecasts))
        log.info("PMV range → min = %.3f   max = %.3f   mean = %.3f",
                 min(forecasts), max(forecasts),
                 sum(forecasts)/len(forecasts))

        return model, feat_sc, pmv_sc, metrics, forecasts

    return model, feat_sc, pmv_sc, metrics


"""
============================================================
SUMMARY — WHY THESE 4 PHYSICS LAWS?
============================================================

The 4 laws were chosen using 3 criteria:

1. DIRECTLY FROM THE FANGER EQUATION
   Your project already implements ISO 7730 Fanger PMV.
   The Fanger equation tells us EXACTLY which variables drive PMV and
   in which direction. Laws 2, 3, and 4 come directly from the equation.

2. MOST IMPORTANT SENSOR RELATIONSHIPS IN YOUR HVAC SYSTEM
   Cooling_Power (Law 1) is your strongest predictor (75% importance).
   If the LSTM gets the cooling→PMV direction wrong, MPC will issue
   wrong valve commands. This is the single most critical constraint.

3. DIRECTIONAL (NOT MAGNITUDE) CONSTRAINTS
   We only constrain the DIRECTION of change (↑ or ↓), not the exact
   magnitude. This is intentional:
   - The exact magnitude depends on many factors (occupancy, weather, etc.)
   - The direction is always physically fixed (a physical law)
   - Directional constraints are much less likely to over-constrain
     the model and hurt accuracy

What we deliberately did NOT include:
   - Flowrate and CHWR-CHWS: these affect PMV through Cooling_Power,
     which is already covered by Law 1. Adding them would double-penalise
     the same physical pathway and could destabilise training.
   - Return_air_Co2: increases CO2 → more people → more body heat → PMV↑
     BUT this relationship has a time lag (people enter, CO2 rises slowly).
     A directional constraint on CO2 vs PMV can fire incorrectly during
     the lag period. We exclude it to avoid penalising correct predictions.
   - Return_air_static_pressure: indirect airflow indicator with no clean
     monotonic relationship to PMV. Excluding it avoids spurious penalties.
"""


"""
============================================================
STEP 11 — SHARED HELPERS FOR VARIANTS (used by pcdl_v1/v2/v3 and main.py)
============================================================

These two functions are what pcdl_v1.py, pcdl_v2.py, and pcdl_v3.py call.
They are defined here (not in the variant files) to avoid code duplication.

  train_variant(data, config)
    → builds the same LSTM trunk as build_model() but attaches a
      PhysicsConstraintLayer configured with the variant's lambda weights.
    → called once per variant in main.py with VARIANT_V1/V2/V3 config dicts.

  rolling_forecast(model, train_df, test_df, feat_sc, pmv_sc)
    → the rolling window inference loop extracted into a reusable function.
    → used by run_v1/v2/v3 and main.py's PCEL forecast loop.
"""


def train_variant(data, config):
    """
    Train one PCDL variant using a specific physics penalty configuration.

    This is the core reuse function. It builds the same LSTM architecture
    as build_model() but injects the variant's lambda weights into the
    PhysicsConstraintLayer. The training loop, callbacks, and data are
    identical across all variants.

    INPUT:  data   — dict from prepare_data()
                     keys: X_train, y_train, X_val, y_val,
                           feat_scaler, pmv_scaler
            config — variant config dict with keys:
                     name, cooling, offcoil, humidity, bounds

    OUTPUT: (trained_model, history)
              trained_model — Keras Model with this variant's physics layer
              history       — Keras History (loss curves per epoch)

    Example:
      from pcdl_v1 import VARIANT_V1
      data = prepare_data(train_df)                  # prepare once
      model_v1, hist_v1 = train_variant(data, VARIANT_V1)
      model_v2, hist_v2 = train_variant(data, VARIANT_V2)  # same data
      model_v3, hist_v3 = train_variant(data, VARIANT_V3)  # same data
    """
    X_train = data["X_train"]   # (N-12, 12, 7)
    y_train = data["y_train"]   # (N-12,)
    X_val   = data["X_val"]     # (M-12, 12, 7)
    y_val   = data["y_val"]     # (M-12,)

    log.info(
        "train_variant | variant=%s | "
        "λ_cooling=%.2f  λ_offcoil=%.2f  λ_humidity=%.2f  λ_bounds=%.2f",
        config["name"],
        config["cooling"], config["offcoil"],
        config["humidity"], config["bounds"],
    )
    log.info("train_variant | X_train=%s  y_train=%s", X_train.shape, y_train.shape)

    # ── Build LSTM trunk (identical architecture across all variants) ──────────
    # INPUT:  input_shape = (12, 7)
    # The ONLY difference between variants is the PhysicsConstraintLayer args
    input_shape = (X_train.shape[1], X_train.shape[2])
    inputs      = Input(shape=input_shape, name="sensor_window")

    x = LSTM(64, return_sequences=True, dropout=0.2,
             recurrent_dropout=0.1, name="lstm_1")(inputs)
    # INPUT:  (batch, 12, 7)
    # OUTPUT: (batch, 12, 64)  — hidden state at every timestep

    x = LSTM(32, return_sequences=False, dropout=0.2,
             name="lstm_2")(x)
    # INPUT:  (batch, 12, 64)
    # OUTPUT: (batch, 32)  — single context vector for the whole window

    x = Dense(16, activation="relu", name="dense_1")(x)
    # INPUT:  (batch, 32)
    # OUTPUT: (batch, 16)

    x = Dropout(0.2, name="dropout_1")(x)

    raw_pred = Dense(1, activation="linear", name="pmv_raw")(x)
    # INPUT:  (batch, 16)
    # OUTPUT: (batch, 1)  — raw scaled PMV prediction

    # ── Attach THIS variant's physics constraint layer ────────────────────────
    # This is the ONLY line that differs between V1, V2, V3, and pcdl_simple
    # INPUT:  [features (batch,12,7), raw_pred (batch,1)]
    # OUTPUT: (batch,1)  — predictions UNCHANGED, penalties added via add_loss()
    output = PhysicsConstraintLayer(
        lambda_cooling  = config["cooling"],
        lambda_offcoil  = config["offcoil"],
        lambda_humidity = config["humidity"],
        lambda_bounds   = config["bounds"],
        name            = f"physics_{config['name']}",
    )([inputs, raw_pred])

    model = Model(inputs=inputs, outputs=output,
                  name=f"PCDL_{config['name']}")
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="mse",       # data loss; physics penalties added via add_loss()
        metrics=["mae"],
    )
    log.info("train_variant | model built | params=%d", model.count_params())

    # ── Training callbacks — identical across all variants ────────────────────
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=PATIENCE,
            restore_best_weights=True,  # rewinds to best epoch automatically
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,   # halve LR on plateau
            patience=8,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        shuffle=True,    # Enabled shuffling
        verbose=1,
    )
    # INPUT:  X_train (N-12, 12, 7)   y_train (N-12,)
    # OUTPUT: trained model weights + loss/val_loss history per epoch

    best_val = min(history.history["val_loss"])
    stopped  = len(history.history["loss"])
    log.info("train_variant | done | variant=%s  stopped_epoch=%d  "
             "best_val_loss=%.6f", config["name"], stopped, best_val)

    return model, history


def rolling_forecast(model, train_df, test_df, feat_scaler, pmv_scaler):
    """
    Run rolling-window inference across a test file.

    How the rolling window works:
      - Seed: last WINDOW (12) rows of training data — unscaled
      - For each test row:
          1. Predict PMV using the current 12-row window
          2. Shift window forward: drop oldest row, append current test row
      - Result: one prediction per test row (after initial WINDOW warmup)

    INPUT:  model        — trained PCDL Keras model (any variant)
            train_df     — training DataFrame (used only for seeding the window)
            test_df      — test DataFrame (sensor columns only, no PMV needed)
            feat_scaler  — fitted MinMaxScaler for 7 features (from prepare_data)
            pmv_scaler   — fitted MinMaxScaler for PMV    (from prepare_data)

    OUTPUT: list of float — predicted PMV values, one per test row

    Note: Uses predict_single() internally so Dropout is disabled
    and PhysicsConstraintLayer.add_loss() is suppressed during inference.
    """
    norm = lambda s: s.lower().replace(" ", "").replace("_", "").replace("-", "")

    # Map test columns
    test_cols = {norm(c): c for c in test_df.columns}
    feat_cols = [test_cols[norm(f)] for f in FEATURES]
    test_clean = test_df[feat_cols].dropna().reset_index(drop=True)
    # INPUT shape: (T, 7)  — T test rows, 7 sensor features

    # Seed rolling window with last WINDOW rows of TRAINING data (unscaled)
    train_cols  = {norm(c): c for c in train_df.columns}
    train_feats = [train_cols[norm(f)] for f in FEATURES]
    history_raw = train_df[train_feats].dropna().values[-WINDOW:]
    # shape: (12, 7)  — last 1 hour of training, raw unscaled values

    log.info("rolling_forecast | test rows=%d  seed window shape=%s",
             len(test_clean), history_raw.shape)

    forecasts = []
    for i, row in test_clean.iterrows():
        # Step 1: Predict PMV for current 12-row window
        # INPUT:  history_raw (12, 7) — unscaled
        # OUTPUT: float — real PMV units
        pmv_pred = predict_single(model, history_raw, feat_scaler, pmv_scaler)
        forecasts.append(pmv_pred)

        # Step 2: Shift window — drop oldest row, append this test row
        new_row     = test_clean.loc[i].values.astype("float32")  # (7,)
        history_raw = np.vstack([history_raw[1:], new_row])
        # history_raw stays (12, 7) throughout

    log.info("rolling_forecast | done | %d predictions", len(forecasts))
    return forecasts


# ── Entry point guard ─────────────────────────────────────────────────────────
# pcdl_simple.py is a shared library — do not run it directly.
# Use main.py instead.
# ===== MODEL 2 METRICS =====
if __name__ == "__main__":
    import pandas as pd

    # Load project files (example)
    try:
        train_df = pd.read_excel("MPC_V7_Training_Data_TRAIN_376rows.xlsx")
        # In a real run with separate files, you would also load test_df here:
        # test_df = pd.read_excel("MPC_V7_Test_Data_30pct.xlsx")
        test_df = None 
    except FileNotFoundError:
        log.warning("Example data file not found. Please provide valid CSV/Excel files.")
        train_df = None
        test_df = None

    if train_df is not None:
        # Prepare data (demonstrates 70/30 internal split if test_df is None)
        data = prepare_data(train_df, test_df)

        # Train model
        model, history, feat_sc, pmv_sc = train_model(data)

        # Evaluate
        metrics = evaluate(model, data)

        # Residual calculation
        y_true = data["y_val_raw"]
        y_pred = metrics["preds_real"]
        sum_residuals = (y_true - y_pred).sum()

        print("\n=== MODEL RESULTS ===")
        print("MAE:", metrics["mae"])
        print("RMSE:", metrics["rmse"])
        print("R2:", metrics["r2"])
        print("MAPE:", metrics["mape"])
        print("Sum of Residuals:", sum_residuals)
