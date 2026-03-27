"""
pcdl_v3.py
==========
PCDL Variant 3 — All Constraints Combined (Physics Anchor).

This file defines the physics penalty configuration for V3 and exposes
a single run_v3() function. All heavy lifting (data prep, model building,
training, evaluation, save/load) is done by pcdl_simple.py.

============================================================
V3 PHYSICS FOCUS — WHY THIS CONFIGURATION?
============================================================

Variant V3 enforces ALL FOUR physics penalties simultaneously.
This is the strictest PCDL variant — its lambda weights are identical
to the default configuration in pcdl_simple.py.

  Active penalty 1 — Cooling_Power vs PMV   (λ = 0.15)
  Active penalty 2 — Offcoil_Temperature vs PMV  (λ = 0.10)
  Active penalty 3 — Return_air_RH vs PMV   (λ = 0.10)
  Active penalty 4 — PMV boundary           (λ = 0.05)

  Total physics weight = 0.15 + 0.10 + 0.10 + 0.05 = 0.40
  (same as V1 and V2 for a fair PCEL comparison)

Why lower individual weights than V1 and V2?
  All four penalties compete for influence in the gradient.
  If each had 0.30, the combined physics weight would be 1.20 — far too
  high, leaving only ~45% for data accuracy (MSE). Spreading 0.40 across
  all four keeps the data loss dominant while enforcing all physics laws.
  λ1=0.15 (cooling still highest — reflects 75% feature importance).
  λ4=0.05 (bounds lightest — predictions rarely exceed the range, so a
  small weight is sufficient to deter outliers without penalising normal cases).

Role of V3 in the PCEL ensemble
  V3 is the "physics anchor". When V1's cooling penalty is inactive
  (flat cooling periods) and V2's temperature/humidity penalty is inactive
  (stable environmental conditions), V3 still enforces all four laws.
  This guarantees that at least some physics enforcement is present in the
  ensemble at every timestep, regardless of sensor behaviour.

Trade-off
  V3 produces the most physically consistent predictions but may have
  slightly higher MAE than V1 or V2, because four competing penalties
  create more friction against purely data-driven fitting. This is expected
  and acceptable — the ensemble averages V3 with V1 and V2, which are less
  constrained and more accurate in their respective channels.

============================================================
"""

import logging
import Pcdl as pcdl

# ── Logging ───────────────────────────────────────────────────────────────────
log = logging.getLogger("pcdl_v3")

# ── V3 variant configuration ──────────────────────────────────────────────────
VARIANT_V3 = {
    "name":     "V3_Combined",  # used in model name + saved file names

    # Physics penalty weights (lambdas)
    # All four penalties active — same as pcdl_simple.py defaults
    "cooling":  0.15,   # λ1 — ACTIVE: cooling highest (75% feature importance)
    "offcoil":  0.10,   # λ2 — ACTIVE: supply air temp direct PMV driver
    "humidity": 0.10,   # λ3 — ACTIVE: humidity affects perceived warmth
    "bounds":   0.05,   # λ4 — ACTIVE: safety constraint (lightest — rarely fires)

    # Total physics weight = 0.40
}


def run_v3(train_df=None, test_df=None, data=None):
    """
    Full pipeline for PCDL Variant 3 (All Constraints Combined).

    Delegates all steps to pcdl_simple.py — no logic is duplicated here.
    The lambda configuration is identical to pcdl_simple.py defaults, but
    the model is saved under the V3 name for clean PCEL bookkeeping.

    INPUT:  train_df — pandas DataFrame (training file)
                       Required columns: 7 sensor features + PMV
            test_df  — pandas DataFrame (test file) or None

    OUTPUT: (model, feat_sc, pmv_sc, metrics, forecasts)
              model     — trained Keras PCDL V3 model
              feat_sc   — fitted MinMaxScaler for 7 features
              pmv_sc    — fitted MinMaxScaler for PMV
              metrics   — dict: mae, rmse, r2, mape, violations, violation_pct
              forecasts — list of predicted PMV floats, or None
    """
    log.info("=== PCDL V3 PIPELINE START ===")
    log.info("Variant config | %s | cooling=%.2f  offcoil=%.2f  "
             "humidity=%.2f  bounds=%.2f",
             VARIANT_V3["name"],
             VARIANT_V3["cooling"], VARIANT_V3["offcoil"],
             VARIANT_V3["humidity"], VARIANT_V3["bounds"])

    if data is None:
        log.info("Step 1: Preparing data...")
        data = pcdl.prepare_data(train_df, test_df)
    else:
        log.info("Step 1: Using provided shared data...")

    # ── Step 2: Train with V3 lambdas ─────────────────────────────────────────
    # INPUT:  data dict + VARIANT_V3 config
    # OUTPUT: trained Keras model, Keras History
    # HOW:    builds shared LSTM trunk + PhysicsConstraintLayer(
    #           lambda_cooling=0.15, lambda_offcoil=0.10,
    #           lambda_humidity=0.10, lambda_bounds=0.05 )
    #         total loss = MSE + all 4 penalties
    log.info("Step 2: Training PCDL V3 (all four penalties)...")
    model, history = pcdl.train_variant(data, VARIANT_V3)

    feat_sc = data["feat_scaler"]
    pmv_sc  = data["pmv_scaler"]

    # ── Step 3: Evaluate on validation set ───────────────────────────────────
    # INPUT:  model, data dict
    # OUTPUT: metrics dict — all values in real PMV units
    log.info("Step 3: Evaluating PCDL V3...")
    metrics = pcdl.evaluate(model, data)

    log.info("─" * 50)
    log.info("PCDL V3 (%s) — VALIDATION RESULTS", VARIANT_V3["name"])
    log.info("  MAE        = %.4f PMV units", metrics["mae"])
    log.info("  RMSE       = %.4f PMV units", metrics["rmse"])
    log.info("  R²         = %.4f",           metrics["r2"])
    log.info("  MAPE       = %.2f%%",          metrics["mape"])
    log.info("  Violations = %d (%.1f%%)",
             metrics["violations"], metrics["violation_pct"])
    log.info("─" * 50)

    # ── Step 4: Save ──────────────────────────────────────────────────────────
    # FILES: saved_models/PCDL_V3_Combined_model.keras
    #        saved_models/PCDL_V3_Combined_scalers.pkl
    log.info("Step 4: Saving V3 model...")
    pcdl.save_model(model, feat_sc, pmv_sc, model_name=VARIANT_V3["name"])

    # ── Step 5: Rolling forecast (optional) ──────────────────────────────────
    forecasts = None
    if test_df is not None:
        log.info("Step 5: Rolling forecast on test data...")
        forecasts = pcdl.rolling_forecast(
            model, train_df, test_df, feat_sc, pmv_sc
        )
        log.info("V3 Forecast complete | %d predictions | "
                 "min=%.3f  max=%.3f  mean=%.3f",
                 len(forecasts),
                 min(forecasts), max(forecasts),
                 sum(forecasts) / len(forecasts))

    log.info("=== PCDL V3 PIPELINE COMPLETE ===")
    return model, feat_sc, pmv_sc, metrics, forecasts, history