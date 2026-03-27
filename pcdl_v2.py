"""
pcdl_v2.py
==========
PCDL Variant 2 — Environmental Sensor Constraint.

This file defines the physics penalty configuration for V2 and exposes
a single run_v2() function. All heavy lifting (data prep, model building,
training, evaluation, save/load) is done by pcdl_simple.py.

============================================================
V2 PHYSICS FOCUS — WHY THIS CONFIGURATION?
============================================================

Variant V2 enforces TWO environmental laws and ONE safety bound:

  Active penalty 1 — Offcoil_Temperature vs PMV  (λ = 0.20)
  ─────────────────────────────────────────────────────────
  Physical law:  Offcoil_Temperature is the temperature of supply air
                 leaving the AHU coils and flowing into the room.
                 When it increases → warmer air enters room → room air temp rises
                 → PMV increases.
                 Therefore: ΔOffcoil_Temperature and ΔPMV must have the SAME sign.

  Why this law? ISO 7730 Fanger equation maps Offcoil_Temperature directly
  to 'ta' (air temperature), the dominant PMV driver.
  1°C increase in air temperature ≈ +0.5 PMV change under typical office conditions.
  This is the most direct link between a sensor and perceived comfort.

  Active penalty 2 — Return_air_RH vs PMV  (λ = 0.10)
  ──────────────────────────────────────────────────────
  Physical law:  Higher room humidity reduces evaporative cooling from the skin.
                 Sweat cannot evaporate as efficiently in humid air.
                 → Body retains more heat → room feels warmer → PMV increases.
                 Therefore: ΔReturn_air_RH and ΔPMV must have the SAME sign.

  Why this law? In the Fanger equation, humidity enters via 'pa' (partial
  water vapour pressure): pa = (RH/100) × 10 × exp(16.65 - 4030/(ta+235)).
  Higher pa → reduced skin latent heat loss (hl1) → higher PMV.

  Active penalty 3 — PMV boundary  (λ = 0.10)
  ─────────────────────────────────────────────
  Always included across all variants as a safety guardrail.

  Inactive penalty — cooling (0.00)
  ─────────────────────────────────
  V2 is intentionally unconstrained on the cooling actuator channel.
  V1 covers that — so in the PCEL ensemble:
    V1 = actuator accuracy  |  V2 = environmental accuracy

  Total physics weight = 0.20 + 0.10 + 0.10 = 0.40
  (same as V1 and V3 for fair PCEL comparison)

============================================================
"""

import logging
import Pcdl as pcdl

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("pcdl_v2")

# ── V2 variant configuration ──────────────────────────────────────────────────
VARIANT_V2 = {
    "name":     "V2_Environment",  # used in model name + saved file names

    # Physics penalty weights (lambdas)
    "cooling":  0.00,   # λ1 — inactive in V2 (covered by V1)
    "offcoil":  0.20,   # λ2 — ACTIVE: supply air temp is dominant comfort driver
    "humidity": 0.10,   # λ3 — ACTIVE: humidity affects perceived warmth via sweat
    "bounds":   0.10,   # λ4 — ACTIVE: safety constraint, always included

    # Total physics weight = 0.40
}


def run_v2(train_df=None, test_df=None, data=None):
    """
    Full pipeline for PCDL Variant 2 (Environmental Constraint).

    Delegates all steps to pcdl_simple.py — no logic is duplicated here.
    The only difference from pcdl_simple.run_example() is that the model
    is built with VARIANT_V2 lambdas: offcoil=0.20, humidity=0.10, bounds=0.10.

    NOTE: If calling this from main.py for PCEL, pass the same train_df
    that was used for V1 so all three variants share the same scalers.
    prepare_data() is called inside here — if you already have a data dict,
    use pcdl.train_variant(data, VARIANT_V2) directly instead.

    INPUT:  train_df — pandas DataFrame (training file)
                       Required columns: 7 sensor features + PMV
            test_df  — pandas DataFrame (test file) or None

    OUTPUT: (model, feat_sc, pmv_sc, metrics, forecasts)
              model     — trained Keras PCDL V2 model
              feat_sc   — fitted MinMaxScaler for 7 features
              pmv_sc    — fitted MinMaxScaler for PMV
              metrics   — dict: mae, rmse, r2, mape, violations, violation_pct
              forecasts — list of predicted PMV floats, or None
    """
    log.info("=== PCDL V2 PIPELINE START ===")
    log.info("Variant config | %s | cooling=%.2f  offcoil=%.2f  "
             "humidity=%.2f  bounds=%.2f",
             VARIANT_V2["name"],
             VARIANT_V2["cooling"], VARIANT_V2["offcoil"],
             VARIANT_V2["humidity"], VARIANT_V2["bounds"])

    if data is None:
        log.info("Step 1: Preparing data...")
        data = pcdl.prepare_data(train_df, test_df)
    else:
        log.info("Step 1: Using provided shared data...")

    # ── Step 2: Train with V2 lambdas ─────────────────────────────────────────
    # INPUT:  data dict + VARIANT_V2 config
    # OUTPUT: trained Keras model, Keras History
    # HOW:    builds shared LSTM trunk + PhysicsConstraintLayer(
    #           lambda_cooling=0.00, lambda_offcoil=0.20,
    #           lambda_humidity=0.10, lambda_bounds=0.10 )
    #         total loss = MSE + 0.20×offcoil_penalty + 0.10×humidity_penalty
    #                          + 0.10×bounds_penalty
    log.info("Step 2: Training PCDL V2 (offcoil + humidity + bounds)...")
    model, history = pcdl.train_variant(data, VARIANT_V2)

    feat_sc = data["feat_scaler"]
    pmv_sc  = data["pmv_scaler"]

    # ── Step 3: Evaluate on validation set ───────────────────────────────────
    # INPUT:  model, data dict
    # OUTPUT: metrics dict — all values in real PMV units
    log.info("Step 3: Evaluating PCDL V2...")
    metrics = pcdl.evaluate(model, data)

    log.info("─" * 50)
    log.info("PCDL V2 (%s) — VALIDATION RESULTS", VARIANT_V2["name"])
    log.info("  MAE        = %.4f PMV units", metrics["mae"])
    log.info("  RMSE       = %.4f PMV units", metrics["rmse"])
    log.info("  R²         = %.4f",           metrics["r2"])
    log.info("  MAPE       = %.2f%%",          metrics["mape"])
    log.info("  Violations = %d (%.1f%%)",
             metrics["violations"], metrics["violation_pct"])
    log.info("─" * 50)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    # FILES: saved_models/PCDL_V2_Environment_model.keras
    #        saved_models/PCDL_V2_Environment_scalers.pkl
    log.info("Step 4: Saving V2 model...")
    pcdl.save_model(model, feat_sc, pmv_sc, model_name=VARIANT_V2["name"])

    # ── Step 5: Rolling forecast (optional) ──────────────────────────────────
    forecasts = None
    if test_df is not None:
        log.info("Step 5: Rolling forecast on test data...")
        forecasts = pcdl.rolling_forecast(
            model, train_df, test_df, feat_sc, pmv_sc
        )
        log.info("V2 Forecast complete | %d predictions | "
                 "min=%.3f  max=%.3f  mean=%.3f",
                 len(forecasts),
                 min(forecasts), max(forecasts),
                 sum(forecasts) / len(forecasts))

    log.info("=== PCDL V2 PIPELINE COMPLETE ===")
    return model, feat_sc, pmv_sc, metrics, forecasts, history