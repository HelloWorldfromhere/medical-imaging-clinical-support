"""
Step 3: Patch main.py to handle updated cnn_inference.py
========================================================
Run from project root:
    python fix_3_patch_main.py

This patches api/main.py to work with the new predict_conditions()
return format (dict instead of list) and adds the
"needs_manual_selection" handling.
"""

import re

MAIN_PATH = "api/main.py"

with open(MAIN_PATH, "r", encoding="utf-8") as f:
    content = f.read()

changes = 0

# ── Fix 1: /predict endpoint ──────────────────────────────────────────────
# Old: predictions = predict_conditions(image_bytes)
#      return PredictResponse(predictions=[ConditionPrediction(**p) for p in predictions], ...)
# New: result = predict_conditions(image_bytes)  # returns dict now
#      return PredictResponse(predictions=[ConditionPrediction(**p) for p in result["predictions"]], ...)

# Find the predict endpoint and replace the predict_conditions call
old_predict = "predictions = predict_conditions(image_bytes)"
new_predict = "result = predict_conditions(image_bytes)"

if old_predict in content:
    content = content.replace(old_predict, new_predict, 1)
    changes += 1
    print(f"  [OK] Replaced predict_conditions call in /predict")

    # Fix the return statement in /predict
    old_return = """return PredictResponse(
        predictions=[ConditionPrediction(**p) for p in predictions],
        model_loaded=is_model_loaded(),
        inference_latency_ms=round(latency, 2),
    )"""

    new_return = """return PredictResponse(
        predictions=[ConditionPrediction(**p) for p in result["predictions"]],
        model_loaded=result["model_loaded"],
        needs_manual_selection=result.get("needs_manual_selection", False),
        inference_latency_ms=round(latency, 2),
    )"""

    if old_return in content:
        content = content.replace(old_return, new_return, 1)
        changes += 1
        print(f"  [OK] Updated PredictResponse construction")
    else:
        print(f"  [WARN] Could not find exact PredictResponse return — check manually")

# ── Fix 2: /analyze endpoint ──────────────────────────────────────────────
# The /analyze endpoint also calls predict_conditions
old_analyze = """    image_bytes = await file.read()

    # Step 1: CNN prediction
    cnn_start = time.perf_counter()
    predictions = predict_conditions(image_bytes)
    cnn_ms = (time.perf_counter() - cnn_start) * 1000"""

new_analyze = """    image_bytes = await file.read()

    # Step 1: CNN prediction
    cnn_start = time.perf_counter()
    cnn_result = predict_conditions(image_bytes)
    predictions = cnn_result["predictions"]
    needs_manual = cnn_result.get("needs_manual_selection", False)
    cnn_ms = (time.perf_counter() - cnn_start) * 1000"""

if old_analyze in content:
    content = content.replace(old_analyze, new_analyze, 1)
    changes += 1
    print(f"  [OK] Updated /analyze CNN prediction call")
else:
    # Try a more flexible match
    if "predictions = predict_conditions(image_bytes)" in content:
        # There might be a second occurrence in /analyze
        # Replace only the second one
        first_idx = content.index("predictions = predict_conditions(image_bytes)")
        remaining = content[first_idx + 1:]
        if "predict_conditions(image_bytes)" in remaining:
            print(f"  [INFO] Found second predict_conditions call — needs manual fix in /analyze")
        else:
            print(f"  [INFO] Only one predict_conditions call found — /predict already fixed")
    print(f"  [WARN] Could not find exact /analyze pattern — check manually")

# ── Fix 3: Add needs_manual_selection to PredictResponse in schemas.py ────
print(f"\n  Remember to add to PredictResponse in schemas.py:")
print(f"    needs_manual_selection: bool = False")

# ── Save ──────────────────────────────────────────────────────────────────
if changes > 0:
    with open(MAIN_PATH, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n  Saved {changes} changes to {MAIN_PATH}")
else:
    print(f"\n  No automatic changes applied — your main.py may have a different format.")
    print(f"  Apply these changes manually:")
    print(f"    1. predict_conditions() now returns a dict, not a list")
    print(f"    2. Access predictions via result['predictions']")
    print(f"    3. Access needs_manual_selection via result['needs_manual_selection']")

print(f"\nDone! Test with: uvicorn api.main:app --reload --port 8080")
