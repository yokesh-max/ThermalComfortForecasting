"""
pcdl_v4.py
==========
PCDL Variant 4 — Cooling + Offcoil Emphasis.

This file defines the physics penalty configuration for V4 and exposes
a single run_v4() function. All heavy lifting (data prep, model building,
training, evaluation, save/load) is done by pcdl.py.

============================================================
V4 PHYSICS FOCUS — WHY THIS CONFIGURATION?
============================================================

Variant V4 emphasizes the two most direct HVAC thermodynamic drivers
of thermal comfort:

  Active penalty 1 — Cooling_Power vs PMV        (λ = 0.20)
  Active penalty 2 — Offcoil_Temperature vs PMV  (λ = 0.15)
  Active penalty 3 — Return_air_RH vs PMV        (λ = 0.00)
  Active penalty 4 — PMV boundary                (λ = 0.05)

  Total physics weight = 0.20 + 0.15 + 0.00 + 0.05 = 0.40
  (same as V1, V2, and V3 for a fair PCEL comparison)

Why this configuration?
-----------------------
V4 is designed to test whether stronger enforcement of the two most
important HVAC physical relationships improves PMV prediction:

1. Cooling_Power:
   When cooling power increases, room air should cool and PMV should drop.
   This remains one of the strongest physical laws in the system.

2. Offcoil_Temperature:
   Offcoil temperature reflects the supply air condition delivered to the
   occupied space. Lower offcoil temperature should reduce room warmth and
   lower PMV.

Humidity is intentionally set to zero in V4 so that this variant isolates
the direct thermodynamic effects of cooling and supply air temperature,
without additional comfort-perception correction from moisture.

Why use V4 in PCEL?
-------------------
V4 acts as the "HVAC thermodynamics specialist" in the ensemble.
While:
  - V1 focuses mostly on actuator behavior,
  - V2 focuses on environmental comfort drivers,
  - V3 balances all constraints,

V4 strongly enforces the two most physically direct cooling pathways:
  actuator control + supply air temperature.

This makes V4 useful for reducing physically unrealistic thermal responses,
especially when humidity changes are small or less informative.

Trade-off
---------
V4 may improve physical realism in thermodynamic transitions, but because
humidity is not constrained, it may be slightly less sensitive to perceived
comfort effects in highly humid conditions. This trade-off is acceptable
because PCEL blends V4 with other variants that capture those effects.

============================================================
HOW TO USE
============================================================

  Standalone (train V4 only):
    import pcdl_v4
    model, feat_sc, pmv_sc, metrics, _ = pcdl_v4.run_v4(train_df)

  As part of PCEL:
    from pcdl_v4 import VARIANT_V4, run_v4
    model_v4, feat_sc, pmv_sc, metrics_v4, _ = run_v4(train_df)

============================================================
"""

import logging
import Pcdl as pcdl

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("pcdl_v4")

# ── V4 variant configuration ──────────────────────────────────────────────────
VARIANT_V4 = {
    "name": "V4_Cooling_Offcoil",

    # Physics penalty weights (lambdas)
    "cooling":  0.20,   # λ1 — ACTIVE: strong actuator effect
    "offcoil":  0.15,   # λ2 — ACTIVE: strong supply air temperature effect
    "humidity": 0.00,   # λ3 — inactive in V4 (humidity intentionally excluded)
    "bounds":   0.05,   # λ4 — ACTIVE: light safety constraint

    # Total physics weight = 0.40
}


def run_v4(train_df=None, test_df=None, data=None):
    """
    Full pipeline for PCDL Variant 4 (Cooling + Offcoil Emphasis).

    Delegates all steps to pcdl.py — no logic is duplicated here.
    The only difference is that the model is built with VARIANT_V4 lambdas:
    cooling=0.20, offcoil=0.15, humidity=0.00, bounds=0.05.

    INPUT:  train_df — pandas DataFrame (training file)
                       Required columns: 7 sensor features + PMV
            test_df  — pandas DataFrame (test file) or None

    OUTPUT: (model, feat_sc, pmv_sc, metrics, forecasts)
              model     — trained Keras PCDL V4 model
              feat_sc   — fitted MinMaxScaler for 7 features
              pmv_sc    — fitted MinMaxScaler for PMV
              metrics   — dict: mae, rmse, r2, mape, violations, violation_pct
              forecasts — list of predicted PMV floats, or None
    """
    log.info("=== PCDL V4 PIPELINE START ===")
    log.info("Variant config | %s | cooling=%.2f  offcoil=%.2f  "
             "humidity=%.2f  bounds=%.2f",
             VARIANT_V4["name"],
             VARIANT_V4["cooling"], VARIANT_V4["offcoil"],
             VARIANT_V4["humidity"], VARIANT_V4["bounds"])

    if data is None:
        log.info("Step 1: Preparing data...")
        data = pcdl.prepare_data(train_df, test_df)
    else:
        log.info("Step 1: Using provided shared data...")

    # ── Step 2: Train with V4 lambdas ─────────────────────────────────────────
    # INPUT:  data dict + VARIANT_V4 config
    # OUTPUT: trained Keras model, Keras History
    # HOW:    builds shared LSTM trunk + PhysicsConstraintLayer(
    #           lambda_cooling=0.20, lambda_offcoil=0.15,
    #           lambda_humidity=0.00, lambda_bounds=0.05 )
    #         total loss = MSE + 0.20×cooling_penalty
    #                          + 0.15×offcoil_penalty
    #                          + 0.05×bounds_penalty
    log.info("Step 2: Training PCDL V4 (cooling + offcoil + bounds)...")
    model, history = pcdl.train_variant(data, VARIANT_V4)

    feat_sc = data["feat_scaler"]
    pmv_sc  = data["pmv_scaler"]

    # ── Step 3: Evaluate on validation set ───────────────────────────────────
    # INPUT:  model, data dict
    # OUTPUT: metrics dict — all values in real PMV units
    log.info("Step 3: Evaluating PCDL V4...")
    metrics = pcdl.evaluate(model, data)

    log.info("─" * 50)
    log.info("PCDL V4 (%s) — VALIDATION RESULTS", VARIANT_V4["name"])
    log.info("  MAE        = %.4f PMV units", metrics["mae"])
    log.info("  RMSE       = %.4f PMV units", metrics["rmse"])
    log.info("  R²         = %.4f",           metrics["r2"])
    log.info("  MAPE       = %.2f%%",         metrics["mape"])
    log.info("  Violations = %d (%.1f%%)",
             metrics["violations"], metrics["violation_pct"])
    log.info("─" * 50)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    # FILES: saved_models/PCDL_V4_Cooling_Offcoil_model.keras
    #        saved_models/PCDL_V4_Cooling_Offcoil_scalers.pkl
    log.info("Step 4: Saving V4 model...")
    pcdl.save_model(model, feat_sc, pmv_sc, model_name=VARIANT_V4["name"])

    # ── Step 5: Rolling forecast on test file (optional) ─────────────────────
    # INPUT:  test_df — same columns as train_df
    # OUTPUT: list of predicted PMV floats (one per test row after warmup)
    forecasts = None
    if test_df is not None:
        log.info("Step 5: Rolling forecast on test data...")
        forecasts = pcdl.rolling_forecast(
            model, train_df, test_df, feat_sc, pmv_sc
        )
        log.info("V4 Forecast complete | %d predictions | "
                 "min=%.3f  max=%.3f  mean=%.3f",
                 len(forecasts),
                 min(forecasts), max(forecasts),
                 sum(forecasts) / len(forecasts))

    log.info("=== PCDL V4 PIPELINE COMPLETE ===")
    return model, feat_sc, pmv_sc, metrics, forecasts, history