"""
main.py
=======
Entry point for the PCDL / PCEL PMV forecasting pipeline.

This file orchestrates three levels of execution:

  Level 1 — pcdl_simple    : Single PCDL model (all 4 penalties, default lambdas)
  Level 2 — pcdl_v1/v2/v3 : Three PCDL variants (each with a different penalty focus)
  Level 3 — PCEL           : Ensemble of V1 + V2 + V3 predictions (equal 1/3 weights)

All ML logic lives in pcdl_simple.py.
Variant configurations (lambda weights) live in pcdl_v1/v2/v3.py.
This file only:
  - Loads data
  - Calls the right pipelines
  - Logs the comparison table
  - Runs the PCEL ensemble blend

============================================================
FILE STRUCTURE
============================================================

  pcdl_simple.py  ← shared library
    │  prepare_data()       — column mapping, scaling, sliding windows
    │  PhysicsConstraintLayer — Keras layer with all 4 physics penalties
    │  build_model()        — LSTM trunk (used by pcdl_simple only)
    │  train_variant()      — LSTM trunk + variant-specific physics layer
    │  train_model()        — wraps train_variant with default lambdas
    │  predict_single()     — single-step inference (training=False)
    │  rolling_forecast()   — rolling window over test file
    │  evaluate()           — MAE, RMSE, R², MAPE, physics violations
    │  save_model()         — saves model + scalers with named prefix
    └  load_model()         — loads model + scalers by name

  pcdl_v1.py  ← imports pcdl_simple, defines VARIANT_V1, exposes run_v1()
  pcdl_v2.py  ← imports pcdl_simple, defines VARIANT_V2, exposes run_v2()
  pcdl_v3.py  ← imports pcdl_simple, defines VARIANT_V3, exposes run_v3()

  main.py     ← YOU ARE HERE
    Imports pcdl_simple + v1 + v2 + v3
    Runs all pipelines and prints comparison table

============================================================
HOW TO RUN
============================================================

  python main.py

  Edit the DATA CONFIGURATION section below to point to your files.
  To skip a pipeline (e.g. run only PCEL), set the RUN_* flags to False.

============================================================
"""

import logging
import os
import sys
import random
import numpy as np
import pandas as pd
import tensorflow as tf

# ── Local imports ─────────────────────────────────────────────────────────────
import Pcdl as pcdl   # shared library: data prep, training, evaluate
from pcdl_v1 import VARIANT_V1, run_v1
from pcdl_v2 import VARIANT_V2, run_v2
from pcdl_v3 import VARIANT_V3, run_v3
from pcdl_v4 import VARIANT_V4, run_v4
from pcdl_v5 import VARIANT_V5, run_v5
# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("main")

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# ══════════════════════════════════════════════════════════════════════════════
# DATA CONFIGURATION — edit these paths and flags before running
# ══════════════════════════════════════════════════════════════════════════════

# Data configuration (handled dynamically in app.py or when calling train_pcel)
TRAIN_FILE = None
TEST_FILE  = None


# Pipeline execution flags — set to False to skip any level
RUN_SIMPLE = False   # Level 1: single PCDL (all 4 penalties, default lambdas)
RUN_V1     = True    # Level 2a: PCDL V1 — Actuator
RUN_V2     = True    # Level 2b: PCDL V2 — Environment
RUN_V3     = True    # Level 2c: PCDL V3 — Combined
RUN_V4     = True    # Level 2d: PCDL V4 — Cooling/Offcoil
RUN_V5     = True    # Level 2e: PCDL V5 — Humidity
RUN_PCEL   = True    # Level 3:  PCEL ensemble (requires V1-V5 trained)


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — shared metrics log banner
# ══════════════════════════════════════════════════════════════════════════════

def _log_metrics(label, metrics):
    """
    Log one model's evaluation metrics in a consistent format.
    Called after each model is evaluated so results are easy to scan in console.

    INPUT:  label   — string name for the model (e.g. "PCDL_Simple")
            metrics — dict from pcdl.evaluate() with keys:
                      mae, rmse, r2, mape, violations, violation_pct
    """
    log.info("┌─ %s ─────────────────────────────────", label)
    log.info("│  MAE        = %.4f PMV units",    metrics["mae"])
    log.info("│  RMSE       = %.4f PMV units",    metrics["rmse"])
    log.info("│  R²         = %.4f",              metrics["r2"])
    log.info("│  MAPE       = %.2f%%",             metrics["mape"])
    log.info("│  Violations = %d  (%.1f%%)",
             metrics["violations"], metrics["violation_pct"])
    log.info("└──────────────────────────────────────────")


# ══════════════════════════════════════════════════════════════════════════════
# PCEL ENSEMBLE WRAPPER
# ══════════════════════════════════════════════════════════════════════════════

class PCELWrapper:
    """
    Unified interface for the PCEL ensemble.
    Acts like a single Keras model by averaging predictions from 5 variants.
    """
    def __init__(self, models):
        self.models = models # {"v1": m1, "v2": m2, "v3": m3, "v4": m4, "v5": m5}
        
    def predict(self, X, verbose=0, **kwargs):
        """Average predictions from all 5 models in scaled space."""
        preds = []
        for m in self.models.values():
            p = m.predict(X, verbose=verbose, **kwargs)
            preds.append(p)
        return np.mean(preds, axis=0)
        
    def __call__(self, X, training=False):
        """Handle functional API calls by averaging tensor outputs."""
        preds = []
        for m in self.models.values():
            p = m(X, training=training)
            preds.append(p)
        return tf.reduce_mean(preds, axis=0)


# ══════════════════════════════════════════════════════════════════════════════
# PCEL FUNCTIONS — defined here so all variant results are in one place
# ══════════════════════════════════════════════════════════════════════════════

def run_pcel(models, data, train_df, test_df=None):
    """
    Ensemble blend of PCDL V1 + V2 + V3 predictions.

    PCEL = Physics-Constrained Ensemble Learning.
    Each variant enforces a different physics law subset.
    Their predictions are averaged with equal weights (1/3 each).

    Blend formula:
      PCEL_pred = (1/5) × (pred_v1 + pred_v2 + pred_v3 + pred_v4 + pred_v5)

    Why equal weights?
      Each variant targets a different failure mode — there is no prior
      reason to trust one more than the others. Equal weights are the
      principled default; they can be tuned later if validation shows
      one variant is consistently more accurate.

    INPUT:  models   — dict {"v1": model, "v2": model, "v3": model}
                       trained Keras models from run_v1/v2/v3
            data     — dict from prepare_data() — shared across all variants
                       (contains feat_scaler and pmv_scaler)
            train_df — training DataFrame — used to seed rolling window
            test_df  — test DataFrame or None

    OUTPUT: (pcel_metrics, pcel_forecasts)
              pcel_metrics   — dict: mae, rmse, r2, mape, violations, violation_pct
                               computed on the validation set using blended predictions
              pcel_forecasts — list of dicts per test row (if test_df given):
                               [{pcel_pmv, pred_v1, pred_v2, pred_v3}, ...]
                               or None if test_df is None
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    feat_sc   = data["feat_scaler"]
    pmv_sc    = data["pmv_scaler"]
    X_val     = data["X_val"]         # (M-12, 12, 7)  scaled
    y_val_raw = data["y_val_raw"]     # (M-12,)          real PMV

    log.info("=== PCEL ENSEMBLE START ===")
    log.info("Blending: V1+V2+V3+V4+V5 with equal weights (0.20 each)")

    # ── Batch predictions from each variant on validation set ─────────────────
    # Each variant predicts on the SAME scaled windows
    # INPUT:  X_val (M-12, 12, 7)  — same for all three
    # OUTPUT: (M-12,) array per variant, in scaled space
    def _batch_predict(model):
        """Run batch inference and return real-unit predictions."""
        preds_sc = model.predict(X_val, verbose=0).ravel()
        # INPUT:  (M-12, 12, 7)
        # OUTPUT: (M-12,)  scaled PMV predictions
        return pmv_sc.inverse_transform(preds_sc.reshape(-1, 1)).ravel()
        # OUTPUT: (M-12,)  real PMV units

    preds_v1 = _batch_predict(models["v1"])   # (M-12,) real PMV
    preds_v2 = _batch_predict(models["v2"])   # (M-12,) real PMV
    preds_v3 = _batch_predict(models["v3"])   # (M-12,) real PMV
    preds_v4 = _batch_predict(models["v4"])   # (M-12,) real PMV
    preds_v5 = _batch_predict(models["v5"])   # (M-12,) real PMV

    log.info("Variant predictions on val set | "
             "V1 mean=%.4f  V2 mean=%.4f  V3 mean=%.4f  V4 mean=%.4f  V5 mean=%.4f",
             preds_v1.mean(), preds_v2.mean(), preds_v3.mean(), preds_v4.mean(), preds_v5.mean())

    # ── Equal-weight blend ────────────────────────────────────────────────────
    # INPUT:  five (M-12,) arrays
    # OUTPUT: (M-12,)  ensemble predictions
    pcel_preds = (preds_v1 + preds_v2 + preds_v3 + preds_v4 + preds_v5) / 5.0

    # ── Metrics on PCEL ensemble predictions ──────────────────────────────────
    mae  = float(mean_absolute_error(y_val_raw, pcel_preds))
    rmse = float(np.sqrt(mean_squared_error(y_val_raw, pcel_preds)))
    r2   = float(r2_score(y_val_raw, pcel_preds))

    # MAPE — exclude samples where actual PMV is zero
    mask = y_val_raw != 0
    mape = float(np.mean(
        np.abs((y_val_raw[mask] - pcel_preds[mask]) / y_val_raw[mask])
    ) * 100) if mask.any() else float("nan")

    # Physics violation count on ensemble predictions
    dummy = np.zeros((len(X_val), len(pcdl.FEATURES)))
    dummy[:, pcdl.IDX_COOLING] = X_val[:, -1, pcdl.IDX_COOLING]
    cooling_real = feat_sc.inverse_transform(dummy)[:, pcdl.IDX_COOLING]

    violations = 0
    for i in range(len(pcel_preds) - 1):
        if (cooling_real[i+1] - cooling_real[i] > 0 and
                pcel_preds[i+1] - pcel_preds[i] > 0.05):
            violations += 1
    violation_pct = 100 * violations / max(len(pcel_preds) - 1, 1)

    pcel_metrics = {
        "mae":           mae,
        "rmse":          rmse,
        "r2":            r2,
        "mape":          mape,
        "violations":    violations,
        "violation_pct": violation_pct,
        "preds_real":    pcel_preds,
    }

    # ── Rolling forecast on test file (optional) ──────────────────────────────
    pcel_forecasts = None
    if test_df is not None:
        log.info("Running PCEL rolling forecast on test data...")

        norm      = lambda s: s.lower().replace(" ","").replace("_","").replace("-","")
        test_cols = {norm(c): c for c in test_df.columns}
        feat_cols = [test_cols[norm(f)] for f in pcdl.FEATURES]
        test_clean = test_df[feat_cols].dropna().reset_index(drop=True)

        # Seed rolling window: last WINDOW rows of training data (unscaled)
        train_cols  = {norm(c): c for c in train_df.columns}
        train_feats = [train_cols[norm(f)] for f in pcdl.FEATURES]
        history_raw = train_df[train_feats].dropna().values[-pcdl.WINDOW:]

        pcel_forecasts = []
        for i, row in test_clean.iterrows():
            # Get individual predictions from each variant for this window
            p_v1 = pcdl.predict_single(models["v1"], history_raw, feat_sc, pmv_sc)
            p_v2 = pcdl.predict_single(models["v2"], history_raw, feat_sc, pmv_sc)
            p_v3 = pcdl.predict_single(models["v3"], history_raw, feat_sc, pmv_sc)
            p_v4 = pcdl.predict_single(models["v4"], history_raw, feat_sc, pmv_sc)
            p_v5 = pcdl.predict_single(models["v5"], history_raw, feat_sc, pmv_sc)

            pcel_pred = (p_v1 + p_v2 + p_v3 + p_v4 + p_v5) / 5.0

            pcel_forecasts.append({
                "pcel_pmv": pcel_pred,
                "pred_v1":  p_v1,
                "pred_v2":  p_v2,
                "pred_v3":  p_v3,
                "pred_v4":  p_v4,
                "pred_v5":  p_v5,
            })

            # Shift rolling window forward by 1 timestep
            new_row     = test_clean.loc[i].values.astype("float32")
            history_raw = np.vstack([history_raw[1:], new_row])

        log.info("PCEL forecast complete | %d predictions", len(pcel_forecasts))

    log.info("=== PCEL ENSEMBLE COMPLETE ===")
    return pcel_metrics, pcel_forecasts


# (Removed redundant train_pcel definition)



def train_pcel(df):
    """
    Main entry point for app.py to train the 5-variant PCEL ensemble.
    
    INPUT:  df — training data (will be split 70/30)
    OUTPUT: (PCELWrapper, history_dict, feat_scaler, pmv_scaler, metrics)
    """
    log.info("Training PCEL Ensemble (V1 - V5)...")
    
    # 1. Prepare shared data
    shared_data = pcdl.prepare_data(df)
    
    # 2. Train all 5 variants using their dedicated run_vX functions
    variant_models = {}
    histories = {}
    variant_metrics = {}
    
    # V1
    m1, _, _, met1, _, h1 = run_v1(data=shared_data)
    variant_models["v1"] = m1
    histories["v1"] = h1.history if hasattr(h1, 'history') else h1
    variant_metrics[VARIANT_V1['name']] = met1

    # V2
    m2, _, _, met2, _, h2 = run_v2(data=shared_data)
    variant_models["v2"] = m2
    histories["v2"] = h2.history if hasattr(h2, 'history') else h2
    variant_metrics[VARIANT_V2['name']] = met2

    # V3
    m3, _, _, met3, _, h3 = run_v3(data=shared_data)
    variant_models["v3"] = m3
    histories["v3"] = h3.history if hasattr(h3, 'history') else h3
    variant_metrics[VARIANT_V3['name']] = met3

    # V4
    m4, _, _, met4, _, h4 = run_v4(data=shared_data)
    variant_models["v4"] = m4
    histories["v4"] = h4.history if hasattr(h4, 'history') else h4
    variant_metrics[VARIANT_V4['name']] = met4

    # V5
    m5, _, _, met5, _, h5 = run_v5(data=shared_data)
    variant_models["v5"] = m5
    histories["v5"] = h5.history if hasattr(h5, 'history') else h5
    variant_metrics[VARIANT_V5['name']] = met5

    # 3. Create Wrapper
    pcel_model = PCELWrapper(variant_models)
    
    # 4. Evaluate Ensemble on Validation Set
    pcel_metrics, _ = run_pcel(variant_models, shared_data, df)
    variant_metrics["PCEL Ensemble"] = pcel_metrics
    
    return pcel_model, histories, shared_data["feat_scaler"], shared_data["pmv_scaler"], pcel_metrics, variant_metrics


def _print_comparison_table(all_metrics):
    """
    Print a formatted comparison table of all models at the end of main().

    INPUT:  all_metrics — dict of {model_name: metrics_dict}
                          e.g. {"PCDL_Simple": {...}, "V1": {...}, "PCEL": {...}}

    The table logs clearly in the console so you can copy it into your report.
    """
    log.info("")
    log.info("=" * 72)
    log.info("FINAL COMPARISON TABLE — ALL MODELS")
    log.info("=" * 72)
    log.info("%-18s | %-8s | %-8s | %-8s | %-8s | %s",
             "Model", "MAE", "RMSE", "R²", "MAPE%", "Violations")
    log.info("-" * 72)
    for name, m in all_metrics.items():
        log.info("%-18s | %-8.4f | %-8.4f | %-8.4f | %-8.2f | %d (%.1f%%)",
                 name,
                 m["mae"], m["rmse"], m["r2"], m["mape"],
                 m["violations"], m["violation_pct"])
    log.info("=" * 72)
    log.info("")
    log.info("Interpretation guide:")
    log.info("  MAE / RMSE   — lower is better (in real PMV units)")
    log.info("  R²           — higher is better (1.0 = perfect fit)")
    log.info("  MAPE%%        — lower is better (unreliable near PMV=0)")
    log.info("  Violations   — lower is better (physics direction errors)")
    log.info("  PCEL should show lower RMSE than any individual variant")
    log.info("  V1 should show fewer cooling violations than PCDL_Simple")
    log.info("  V3 should show fewest total violations (strictest model)")
    log.info("")

## ══════════════════════════════════════════════════════════════════════════════
# VARIANT CONFIGURATIONS — define the physics-loss experiments here
# Must match train_variant(): name, cooling, offcoil, humidity, bounds
# ══════════════════════════════════════════════════════════════════════════════

# VARIANT_V1 = {
#     "name": "V1",
#     "cooling": 0.30,
#     "offcoil": 0.00,
#     "humidity": 0.00,
#     "bounds": 0.10,
# }

# VARIANT_V2 = {
#     "name": "V2",
#     "cooling": 0.00,
#     "offcoil": 0.20,
#     "humidity": 0.10,
#     "bounds": 0.10,
# }

# VARIANT_V3 = {
#     "name": "V3",
#     "cooling": 0.15,
#     "offcoil": 0.10,
#     "humidity": 0.10,
#     "bounds": 0.05,
# }
# VARIANT_V1 = {
#     "name": "V4",
#     "cooling": 0.10,
#     "offcoil": 0.10,
#     "humidity": 0.10,
#     "bounds": 0.05,
# }
# VARIANT_V2 = {
#     "name": "V5",
#     "cooling": 0.05,
#     "offcoil": 0.25,
#     "humidity": 0.20,
#     "bounds": 0.10,
# }
# VARIANT_V3 = {
#     "name": "V6",
#     "cooling": 0.40,
#     "offcoil": 0.00,
#     "humidity": 0.00,
#     "bounds": 0.10,
# }
# ══════════════════════════════════════════════════════════════════════════════
# MAIN — runs when you execute: python main.py
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    log.info("=" * 72)
    log.info("PCDL / PCEL PIPELINE — MAIN ENTRY POINT")
    log.info("=" * 72)

    if TRAIN_FILE is None or not os.path.exists(TRAIN_FILE):
        log.warning("No TRAIN_FILE defined or found. Skipping standalone execution.")
        log.info("PCEL module is ready for use via train_pcel(df).")
        # Exit the main block early
        sys.exit(0)

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info("Loading training data from: %s", TRAIN_FILE)
    ext = os.path.splitext(str(TRAIN_FILE))[1].lower() if TRAIN_FILE else ""
    train_df = pd.read_excel(TRAIN_FILE) if ext in (".xlsx", ".xls") \
               else pd.read_csv(TRAIN_FILE)
    log.info("Training data loaded | shape = %s", train_df.shape)

    test_df = None
    if TEST_FILE is not None and os.path.exists(TEST_FILE):
        log.info("Loading test data from: %s", TEST_FILE)
        ext = os.path.splitext(TEST_FILE)[1].lower()
        test_df = pd.read_excel(TEST_FILE) if ext in (".xlsx", ".xls") \
                  else pd.read_csv(TEST_FILE)
        log.info("Test data loaded | shape = %s", test_df.shape)

    # Collect all metrics for the final comparison table
    all_metrics = {}



    # ──────────────────────────────────────────────────────────────────────────
    # LEVEL 1 — PCDL Simple (all 4 penalties, default lambdas)
    # Uses pcdl_simple.run_example() directly
    # ──────────────────────────────────────────────────────────────────────────
    # if RUN_SIMPLE:
    #     log.info("")
    #     log.info("━" * 72)
    #     log.info("LEVEL 1 — PCDL Simple (default lambdas: all 4 penalties)")
    #     log.info("━" * 72)

    #     result = pcdl.run_example(train_df, test_df)
    #     # run_example returns (model, feat_sc, pmv_sc, metrics)
    #     # or (model, feat_sc, pmv_sc, metrics, forecasts) if test_df given
    #     simple_metrics = result[3]

    #     _log_metrics("PCDL_Simple", simple_metrics)
    #     all_metrics["PCDL_Simple"] = simple_metrics

    # ──────────────────────────────────────────────────────────────────────────
    # LEVEL 2 — Three PCDL Variants
    # prepare_data() is called once and shared across all three variants.
    # This ensures all three use identical scalers — critical for PCEL blending.
    # ──────────────────────────────────────────────────────────────────────────
    if RUN_V1 or RUN_V2 or RUN_V3 or RUN_V4 or RUN_V5 or RUN_PCEL:
        log.info("")
        log.info("━" * 72)
        log.info("LEVEL 2 — PCDL Variants (preparing shared data once)")
        log.info("━" * 72)

        # Prepare data ONCE — all three variants share the same scalers.
        # This is the critical rule: if you fit scalers separately per variant,
        # their PMV predictions would be in different scales and averaging them
        # would be meaningless.
        # INPUT:  train_df (N rows × ≥8 cols)
        # OUTPUT: data dict with X_train, y_train, X_val, y_val,
        #         y_val_raw, feat_scaler, pmv_scaler
        log.info("Preparing shared data for all variants...")
        shared_data = pcdl.prepare_data(train_df)
        log.info("Shared data ready | X_train=%s  X_val=%s",
                 shared_data["X_train"].shape, shared_data["X_val"].shape)

        # Store trained models for PCEL ensemble
        variant_models = {}

    # ── V1: Actuator constraint ───────────────────────────────────────────────
    if RUN_V1:
        log.info("")
        log.info("─" * 60)
        log.info("PCDL V1 — Actuator Constraint")
        log.info("─" * 60)

        # Call run_v1 directly with shared data
        model_v1, _, _, metrics_v1, _, _ = run_v1(data=shared_data)
        _log_metrics("PCDL_V1", metrics_v1)
        all_metrics["PCDL_V1"] = metrics_v1
        variant_models["v1"]   = model_v1

    # ── V2: Environmental constraint ─────────────────────────────────────────
    if RUN_V2:
        log.info("")
        log.info("─" * 60)
        log.info("PCDL V2 — Environmental Constraint")
        log.info("─" * 60)

        model_v2, _, _, metrics_v2, _, _ = run_v2(data=shared_data)
        _log_metrics("PCDL_V2", metrics_v2)
        all_metrics["PCDL_V2"] = metrics_v2
        variant_models["v2"]   = model_v2

    # ── V3: All constraints combined ─────────────────────────────────────────
    if RUN_V3:
        log.info("")
        log.info("─" * 60)
        log.info("PCDL V3 — All Constraints")
        log.info("─" * 60)

        model_v3, _, _, metrics_v3, _, _ = run_v3(data=shared_data)
        _log_metrics("PCDL_V3", metrics_v3)
        all_metrics["PCDL_V3"] = metrics_v3
        variant_models["v3"]   = model_v3

    # ── V4: Cooling + Offcoil emphasis ──────────────────────────────────────────
    if RUN_V4:
        log.info("")
        log.info("─" * 60)
        log.info("PCDL V4 — Cooling + Offcoil Constraint")
        log.info("─" * 60)

        model_v4, _, _, metrics_v4, _, _ = run_v4(data=shared_data)
        _log_metrics("PCDL_V4", metrics_v4)
        all_metrics["PCDL_V4"] = metrics_v4
        variant_models["v4"]   = model_v4

    # ── V5: Humidity-sensitive comfort ──────────────────────────────────────────
    if RUN_V5:
        log.info("")
        log.info("─" * 60)
        log.info("PCDL V5 — Humidity Balanced Constraint")
        log.info("─" * 60)

        model_v5, _, _, metrics_v5, _, _ = run_v5(data=shared_data)
        _log_metrics("PCDL_V5", metrics_v5)
        all_metrics["PCDL_V5"] = metrics_v5
        variant_models["v5"]   = model_v5
    # ──────────────────────────────────────────────────────────────────────────
    # LEVEL 3 — PCEL Ensemble (requires all three variants trained)
    # ──────────────────────────────────────────────────────────────────────────
    if RUN_PCEL:
        if not all(k in variant_models for k in ("v1", "v2", "v3", "v4", "v5")):
            log.warning("PCEL skipped — requires V1 through V5 all trained.")
        else:
            log.info("")
            log.info("━" * 72)
            log.info("LEVEL 3 — PCEL Ensemble (V1 + V2 + V3 + V4 + V5)")
            log.info("━" * 72)

            pcel_metrics, pcel_forecasts = run_pcel(
                models   = variant_models,
                data     = shared_data,
                train_df = train_df,
                test_df  = test_df,
            )

            _log_metrics("PCEL", pcel_metrics)
            all_metrics["PCEL"] = pcel_metrics

            # Optionally: print per-row PCEL forecasts for test file
            if pcel_forecasts is not None:
                log.info("PCEL test forecasts (first 5 rows):")
                for i, r in enumerate(pcel_forecasts[:5]):
                    log.info("  row %d | pcel=%.4f  v1=%.4f  v2=%.4f  v3=%.4f",
                             i, r["pcel_pmv"], r["pred_v1"],
                             r["pred_v2"], r["pred_v3"])

    # ── Final comparison table ─────────────────────────────────────────────────
    if all_metrics:
        _print_comparison_table(all_metrics)

    log.info("All pipelines complete.")