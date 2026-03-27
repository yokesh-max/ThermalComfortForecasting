"""
pcdl_v5.py
==========
PCDL Variant 5 — Humidity-Sensitive Comfort Balance.

This file defines the physics penalty configuration for V5 and exposes
a single run_v5() function. All heavy lifting (data prep, model building,
training, evaluation, save/load) is done by pcdl.py.

============================================================
V5 PHYSICS FOCUS — WHY THIS CONFIGURATION?
============================================================

Variant V5 emphasizes humidity-aware thermal comfort behavior while still
maintaining moderate control over cooling and supply air temperature:

  Active penalty 1 — Cooling_Power vs PMV        (λ = 0.10)
  Active penalty 2 — Offcoil_Temperature vs PMV  (λ = 0.10)
  Active penalty 3 — Return_air_RH vs PMV        (λ = 0.15)
  Active penalty 4 — PMV boundary                (λ = 0.05)

  Total physics weight = 0.10 + 0.10 + 0.15 + 0.05 = 0.40
  (same as V1, V2, V3, and V4 for a fair PCEL comparison)

Why this configuration?
-----------------------
V5 is designed to test whether stronger humidity-aware physical guidance
improves PMV prediction.

In thermal comfort theory, occupants do not perceive temperature alone.
Humidity also affects:
  - sweat evaporation
  - perceived warmth
  - discomfort at the same dry-bulb temperature

Therefore, even when cooling and offcoil behavior are correct, PMV can
still become unrealistic if moisture-related comfort effects are ignored.

V5 gives the highest emphasis to humidity among all variants while still
keeping moderate penalties on:
  - Cooling_Power
  - Offcoil_Temperature

This makes V5 a "comfort-perception specialist" rather than a purely
thermodynamic specialist.

Why use V5 in PCEL?
-------------------
V5 adds a complementary behavior to the ensemble:
  - V1 contributes actuator realism
  - V2 contributes environmental realism
  - V3 contributes balanced all-law consistency
  - V4 contributes strong thermodynamic cooling behavior
  - V5 contributes humidity-sensitive comfort realism

This diversity is valuable because ensemble learning performs best when
each model makes different but meaningful corrections.

Trade-off
---------
V5 may be more responsive to perceived comfort changes, but because its
cooling and offcoil penalties are lighter than V4, it may be slightly
less strict on HVAC thermodynamic transitions. This is acceptable because
PCEL averages V5 with variants that are stronger in those areas.

============================================================
HOW TO USE
============================================================

  Standalone (train V5 only):
    import pcdl_v5
    model, feat_sc, pmv_sc, metrics, _ = pcdl_v5.run_v5(train_df)

  As part of PCEL:
    from pcdl_v5 import VARIANT_V5, run_v5
    model_v5, feat_sc, pmv_sc, metrics_v5, _ = run_v5(train_df)

============================================================
"""

import logging
import Pcdl as pcdl

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("pcdl_v5")

# ── V5 variant configuration ──────────────────────────────────────────────────
VARIANT_V5 = {
    "name": "V5_Humidity_Balanced",

    # Physics penalty weights (lambdas)
    "cooling":  0.10,   # λ1 — ACTIVE: moderate actuator influence
    "offcoil":  0.10,   # λ2 — ACTIVE: moderate supply air temperature effect
    "humidity": 0.15,   # λ3 — ACTIVE: strongest humidity emphasis in all variants
    "bounds":   0.05,   # λ4 — ACTIVE: light safety constraint

    # Total physics weight = 0.40
}


def run_v5(train_df=None, test_df=None, data=None):
    """
    Full pipeline for PCDL Variant 5 (Humidity-Sensitive Comfort Balance).

    Delegates all steps to pcdl.py — no logic is duplicated here.
    The only difference is that the model is built with VARIANT_V5 lambdas:
    cooling=0.10, offcoil=0.10, humidity=0.15, bounds=0.05.

    INPUT:  train_df — pandas DataFrame (training file)
                       Required columns: 7 sensor features + PMV
            test_df  — pandas DataFrame (test file) or None

    OUTPUT: (model, feat_sc, pmv_sc, metrics, forecasts)
              model     — trained Keras PCDL V5 model
              feat_sc   — fitted MinMaxScaler for 7 features
              pmv_sc    — fitted MinMaxScaler for PMV
              metrics   — dict: mae, rmse, r2, mape, violations, violation_pct
              forecasts — list of predicted PMV floats, or None
    """
    log.info("=== PCDL V5 PIPELINE START ===")
    log.info("Variant config | %s | cooling=%.2f  offcoil=%.2f  "
             "humidity=%.2f  bounds=%.2f",
             VARIANT_V5["name"],
             VARIANT_V5["cooling"], VARIANT_V5["offcoil"],
             VARIANT_V5["humidity"], VARIANT_V5["bounds"])

    if data is None:
        log.info("Step 1: Preparing data...")
        data = pcdl.prepare_data(train_df, test_df)
    else:
        log.info("Step 1: Using provided shared data...")

    # ── Step 2: Train with V5 lambdas ─────────────────────────────────────────
    # INPUT:  data dict + VARIANT_V5 config
    # OUTPUT: trained Keras model, Keras History
    # HOW:    builds shared LSTM trunk + PhysicsConstraintLayer(
    #           lambda_cooling=0.10, lambda_offcoil=0.10,
    #           lambda_humidity=0.15, lambda_bounds=0.05 )
    #         total loss = MSE + 0.10×cooling_penalty
    #                          + 0.10×offcoil_penalty
    #                          + 0.15×humidity_penalty
    #                          + 0.05×bounds_penalty
    log.info("Step 2: Training PCDL V5 (humidity-sensitive balanced constraints)...")
    model, history = pcdl.train_variant(data, VARIANT_V5)

    feat_sc = data["feat_scaler"]
    pmv_sc  = data["pmv_scaler"]

    # ── Step 3: Evaluate on validation set ───────────────────────────────────
    # INPUT:  model, data dict
    # OUTPUT: metrics dict — all values in real PMV units
    log.info("Step 3: Evaluating PCDL V5...")
    metrics = pcdl.evaluate(model, data)

    log.info("─" * 50)
    log.info("PCDL V5 (%s) — VALIDATION RESULTS", VARIANT_V5["name"])
    log.info("  MAE        = %.4f PMV units", metrics["mae"])
    log.info("  RMSE       = %.4f PMV units", metrics["rmse"])
    log.info("  R²         = %.4f",           metrics["r2"])
    log.info("  MAPE       = %.2f%%",         metrics["mape"])
    log.info("  Violations = %d (%.1f%%)",
             metrics["violations"], metrics["violation_pct"])
    log.info("─" * 50)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    # FILES: saved_models/PCDL_V5_Humidity_Balanced_model.keras
    #        saved_models/PCDL_V5_Humidity_Balanced_scalers.pkl
    log.info("Step 4: Saving V5 model...")
    pcdl.save_model(model, feat_sc, pmv_sc, model_name=VARIANT_V5["name"])

    # ── Step 5: Rolling forecast on test file (optional) ─────────────────────
    # INPUT:  test_df — same columns as train_df
    # OUTPUT: list of predicted PMV floats (one per test row after warmup)
    forecasts = None
    if test_df is not None:
        log.info("Step 5: Rolling forecast on test data...")
        forecasts = pcdl.rolling_forecast(
            model, train_df, test_df, feat_sc, pmv_sc
        )
        log.info("V5 Forecast complete | %d predictions | "
                 "min=%.3f  max=%.3f  mean=%.3f",
                 len(forecasts),
                 min(forecasts), max(forecasts),
                 sum(forecasts) / len(forecasts))

    log.info("=== PCDL V5 PIPELINE COMPLETE ===")
    return model, feat_sc, pmv_sc, metrics, forecasts, history