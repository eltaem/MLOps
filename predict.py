#!/usr/bin/env python3
import sys
import json
from pathlib import Path

MODEL_JOBLIB = Path(__file__).parent / "model_and_encoder.joblib"
MODEL_JSON = Path(__file__).parent / "model.json"


def err(msg, details=None):
    out = {"error": msg}
    if details is not None:
        out["details"] = details
    print(json.dumps(out))
    sys.exit(1)


def main():
    try:
        import joblib
        import numpy as np
        import xgboost as xgb
    except Exception as e:
        err("missing python dependency", str(e))

    # Parse payload
    payload = {}
    if len(sys.argv) > 1:
        try:
            payload = json.loads(sys.argv[1])
        except Exception as e:
            err("invalid json input", str(e))

    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        err("features must be a list")

    X = np.array(features, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    model = None

    # ---------- Try joblib first ----------
    if MODEL_JOBLIB.exists():
        try:
            model = joblib.load(MODEL_JOBLIB)
        except Exception as e:
            # fall back to JSON model
            sys.stderr.write(f"Failed to load joblib model: {e}\n")

    # ---------- Try XGBoost JSON ----------
    if model is None and MODEL_JSON.exists():
        try:
            booster = xgb.Booster()
            booster.load_model(MODEL_JSON)
            dmat = xgb.DMatrix(X)
            preds = booster.predict(dmat)
            print(json.dumps({"prediction": preds.tolist()}))
            return
        except Exception as e:
            err("failed to load XGBoost JSON model", str(e))

    # ---------- If model was joblib and has predict() ----------
    if model is not None:
        try:
            # case: model was saved as dict with {'model': ..., 'encoder': ...}
            if isinstance(model, dict) and "model" in model and "encoder" in model:
                raw_model = model["model"]
                encoder = model["encoder"]

                proba = None
                preds = raw_model.predict(X).tolist()
                out = {"prediction": preds}

                # convert to original string labels
                try:
                    out["class_labels"] = encoder.inverse_transform(preds).tolist()
                except Exception:
                    pass
                
                # handle probabilities
                if hasattr(raw_model, "predict_proba"):
                    probs = raw_model.predict_proba(X)[0]  # get first sample
                    classes = encoder.classes_
                    out["predicted_class"] = encoder.inverse_transform(preds)[0]
                    out["probabilities"] = {
                        cls: float(p) for cls, p in zip(classes, probs)
                    }

                print(json.dumps(out))
                return

            # case: plain model
            if hasattr(model, "predict"):
                proba = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X).tolist()

                preds = model.predict(X).tolist()
                out = {"prediction": preds}
                if proba is not None:
                    out["probabilities"] = proba

                print(json.dumps(out))
                return

        except Exception as e:
            err("pickle model prediction failed", str(e))

    # ---------- Fallback ----------
    err("prediction failed", "no usable model found (expected model.pkl or model.json)")


if __name__ == "__main__":
    main()
