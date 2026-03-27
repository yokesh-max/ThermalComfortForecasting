"""
pcdl_v1.py
==========
PCDL Variant 1 — Actuator Control Constraint.

This file defines the physics penalty configuration for V1 and exposes
a single run_v1() function. All heavy lifting (data prep, model building,
training, evaluation, save/load) is done by pcdl_simple.py — this file
only sets the lambda weights and delegates.

============================================================
V1 PHYSICS FOCUS — WHY THIS CONFIGURATION?
============================================================

Variant V1 enforces ONE directional law and ONE safety bound:

  Active penalty 1 — Cooling_Power vs PMV  (λ = 0.30)
  ─────────────────────────────────────────────────────
  Physical law:  When Cooling_Power increases (CHW valve opens more),
                 the AHU extracts more heat → room air cools → PMV drops.
                 Therefore: ΔCooling_Power and ΔPMV must have OPPOSITE signs.

  Why the highest weight (0.30)?
  Cooling_Power has 75% feature importance in this HVAC system.
  It is the primary MPC actuator — the controller directly manipulates it.
  If the LSTM predicts PMV rising when cooling increases, the MPC controller
  will open the valve MORE in a futile loop, wasting energy and causing
  the room to oscillate. This is the single most safety-critical constraint.

  Active penalty 2 — PMV boundary  (λ = 0.10)
  ─────────────────────────────────────────────
  Physical law:  PMV is physically bounded to [-3, +3] by the Fanger equation.
                 Predictions outside [0.0, 1.0] in scaled space are impossible.
  Why included:  Safety guardrail — keeps MPC from issuing extreme valve
                 commands in response to out-of-range predictions.

  Inactive penalties — offcoil (0.00) and humidity (0.00)
  ──────────────────────────────────────────────────────
  V1 is intentionally unconstrained on temperature and humidity channels.
  V2 covers those channels — so V1 and V2 are complementary in the ensemble:
    V1 = actuator accuracy  |  V2 = environmental accuracy

  Total physics weight = 0.30 + 0.10 = 0.40
  (same as V2 and V3 for a fair PCEL comparison)

============================================================
HOW TO USE
============================================================

  Standalone (train V1 only):
    import pcdl_v1
    model, feat_sc, pmv_sc, metrics, _ = pcdl_v1.run_v1(train_df)

  As part of PCEL (called from main.py):
    from pcdl_v1 import VARIANT_V1, run_v1
    model_v1, feat_sc, pmv_sc, metrics_v1, _ = run_v1(train_df)
    # main.py blends V1 + V2 + V3 predictions

============================================================
"""

import logging
import Pcdl as pcdl

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("pcdl_v1")

# ── V1 variant configuration ──────────────────────────────────────────────────
# This dict is the ONLY thing that distinguishes V1 from V2, V3, and
# pcdl_simple. All other code — architecture, training loop, evaluation —
# is shared from pcdl_simple.py via pcdl.train_variant(data, config).
VARIANT_V1 = {
    "name":     "V1_Actuator",   # used in model name + saved file names

    # Physics penalty weights (lambdas)
    "cooling":  0.30,   # λ1 — ACTIVE: strongest weight, cooling is primary actuator
    "offcoil":  0.00,   # λ2 — inactive in V1 (covered by V2)
    "humidity": 0.00,   # λ3 — inactive in V1 (covered by V2)
    "bounds":   0.10,   # λ4 — ACTIVE: safety constraint, always included

    # Total physics weight = 0.40
}


def run_v1(train_df=None, test_df=None, data=None):
    """
    Full pipeline for PCDL Variant 1 (Actuator Constraint).

    Delegates all steps to pcdl_simple.py — no logic is duplicated here.
    The only difference from pcdl_simple.run_example() is that the model
    is built with VARIANT_V1 lambdas: cooling=0.30, bounds=0.10.

    INPUT:  train_df — pandas DataFrame (training file)
                       Required columns: 7 sensor features + PMV
            test_df  — pandas DataFrame (test file) or None

    OUTPUT: (model, feat_sc, pmv_sc, metrics, forecasts)
              model     — trained Keras PCDL V1 model
              feat_sc   — fitted MinMaxScaler for 7 features (shared for PCEL)
              pmv_sc    — fitted MinMaxScaler for PMV     (shared for PCEL)
              metrics   — dict: mae, rmse, r2, mape, violations, violation_pct
              forecasts — list of predicted PMV floats, or None
    """
    log.info("=== PCDL V1 PIPELINE START ===")
    log.info("Variant config | %s | cooling=%.2f  offcoil=%.2f  "
             "humidity=%.2f  bounds=%.2f",
             VARIANT_V1["name"],
             VARIANT_V1["cooling"], VARIANT_V1["offcoil"],
             VARIANT_V1["humidity"], VARIANT_V1["bounds"])

    if data is None:
        log.info("Step 1: Preparing data...")
        data = pcdl.prepare_data(train_df, test_df)
    else:
        log.info("Step 1: Using provided shared data...")

    # ── Step 2: Train with V1 lambdas ─────────────────────────────────────────
    # INPUT:  data dict + VARIANT_V1 config
    # OUTPUT: trained Keras model, Keras History
    # HOW:    builds shared LSTM trunk + PhysicsConstraintLayer(
    #           lambda_cooling=0.30, lambda_offcoil=0.00,
    #           lambda_humidity=0.00, lambda_bounds=0.10 )
    #         total loss = MSE + 0.30×cooling_penalty + 0.10×bounds_penalty
    log.info("Step 2: Training PCDL V1 (cooling + bounds only)...")
    model, history = pcdl.train_variant(data, VARIANT_V1)

    feat_sc = data["feat_scaler"]
    pmv_sc  = data["pmv_scaler"]

    # ── Step 3: Evaluate on validation set ───────────────────────────────────
    # INPUT:  model, data dict
    # OUTPUT: metrics dict — all values in real PMV units (after inverse_transform)
    log.info("Step 3: Evaluating PCDL V1...")
    metrics = pcdl.evaluate(model, data)

    # Log all metrics clearly so V1 result is visible in the console
    log.info("─" * 50)
    log.info("PCDL V1 (%s) — VALIDATION RESULTS", VARIANT_V1["name"])
    log.info("  MAE        = %.4f PMV units", metrics["mae"])
    log.info("  RMSE       = %.4f PMV units", metrics["rmse"])
    log.info("  R²         = %.4f",           metrics["r2"])
    log.info("  MAPE       = %.2f%%",          metrics["mape"])
    log.info("  Violations = %d (%.1f%%)",
             metrics["violations"], metrics["violation_pct"])
    log.info("─" * 50)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    # FILES: saved_models/PCDL_V1_Actuator_model.keras
    #        saved_models/PCDL_V1_Actuator_scalers.pkl
    log.info("Step 4: Saving V1 model...")
    pcdl.save_model(model, feat_sc, pmv_sc, model_name=VARIANT_V1["name"])

    # ── Step 5: Rolling forecast on test file (optional) ─────────────────────
    # INPUT:  test_df — same columns as train_df
    # OUTPUT: list of predicted PMV floats (one per test row after warmup)
    forecasts = None
    if test_df is not None:
        log.info("Step 5: Rolling forecast on test data...")
        forecasts = pcdl.rolling_forecast(
            model, train_df, test_df, feat_sc, pmv_sc
        )
        log.info("V1 Forecast complete | %d predictions | "
                 "min=%.3f  max=%.3f  mean=%.3f",
                 len(forecasts),
                 min(forecasts), max(forecasts),
                 sum(forecasts) / len(forecasts))

    log.info("=== PCDL V1 PIPELINE COMPLETE ===")
    return model, feat_sc, pmv_sc, metrics, forecasts, history