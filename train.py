#!/usr/bin/env python3
"""
Train script with multiple models and MLflow integration.

Trains and logs different models (XGBoost, RandomForest, LogisticRegression)
on the Iris dataset. Logs all runs to MLflow, including detailed metrics from
classification_report, but only saves the best model (based on weighted F1)
+ encoder as a single joblib file.
"""

import argparse
import sys
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from datetime import datetime
import dvc.api


def parse_args():
    parser = argparse.ArgumentParser(description='Train classifiers on Iris dataset with MLflow logging')
    parser.add_argument('--target', type=str, default='variety', help='Target column name')
    parser.add_argument('--out', type=str, default='model_and_encoder.joblib', help='Output joblib path')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of estimators for tree models')
    parser.add_argument('--mlflow-exp', type=str, default='Iris-Experiments', help='MLflow experiment name')
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from mlflow.models.signature import infer_signature
        from sklearn.metrics import classification_report
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        try:
            from xgboost import XGBClassifier
        except Exception:
            XGBClassifier = None
        from joblib import dump
        import json
    except Exception as e:
        print('Missing dependency:', e, file=sys.stderr)
        sys.exit(2)

    # Load dataset
    # Ambil URL dataset versi tertentu
    data_url = dvc.api.get_url(
        path="iris.csv",
        repo=".",       # repo path
        rev="HEAD"      # bisa ganti commit hash
    )
    print("Dataset URL:", data_url)
    df = pd.read_csv(data_url)
    dataset = mlflow.data.from_pandas(df, name="iris_dataset")

    if args.target not in df.columns:
        print(f"Target column '{args.target}' not found", file=sys.stderr)
        sys.exit(2)

    X = df.drop(args.target, axis=1)
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Candidate models
    models = {}

    if XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=args.n_estimators,
            random_state=args.random_state,
            eval_metric="logloss"
        )

    models["RandomForest"] = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state
    )

    models["LogisticRegression"] = LogisticRegression(
        max_iter=500,
        random_state=args.random_state
    )

    # MLflow experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow.set_experiment(f"{args.mlflow_exp}_{timestamp}")

    best_f1 = -1.0
    best_model = None

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            # Log dataset
            mlflow.log_input(dataset, context="training", tags={"dataset_name": "iris_dataset"})
            print(f"Training {model_name}...")

            # Log common parameters
            mlflow.log_param("random_state", args.random_state)
            mlflow.log_param("test_size", args.test_size)
            mlflow.log_param("metric_for_selection", "f1_weighted")  # metric used for best model

            # Log model-specific parameters
            if hasattr(model, "n_estimators"):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, "max_iter"):
                mlflow.log_param("max_iter", model.max_iter)

            # Train and predict
            model.fit(X_train, y_train_enc)
            preds = model.predict(X_test)

            # Generate classification report and log metrics
            report_dict = classification_report(
                y_test_enc, preds, target_names=le.classes_, output_dict=True
            )

            if "accuracy" in report_dict:
                mlflow.log_metric("accuracy", report_dict["accuracy"])

            for key, value in report_dict.items():
                if key not in ["accuracy", "macro avg", "weighted avg"]:
                    mlflow.log_metric(f"{key}_precision", value["precision"])
                    mlflow.log_metric(f"{key}_recall", value["recall"])
                    mlflow.log_metric(f"{key}_f1", value["f1-score"])
                elif key == "weighted avg":
                    mlflow.log_metric("precision_weighted", value["precision"])
                    mlflow.log_metric("recall_weighted", value["recall"])
                    mlflow.log_metric("f1_weighted", value["f1-score"])

            # Determine best model based on weighted F1
            f1_weighted = report_dict["weighted avg"]["f1-score"]
            if f1_weighted > best_f1:
                best_f1 = f1_weighted
                best_model = model
                best_model_name = model_name

            # Infer signature and log model
            signature = infer_signature(X_train, model.predict(X_train[:5]))
            input_example = X_train.iloc[:5]

            if model_name == "XGBoost":
                mlflow.xgboost.log_model(
                    model,
                    name=f"model_{model_name}",
                    signature=signature,
                    input_example=input_example,
                    model_format="json"
                )
            else:
                mlflow.sklearn.log_model(
                    model,
                    name=f"model_{model_name}",
                    signature=signature,
                    input_example=input_example
                )

            print(f"{model_name} metrics and parameters logged to MLflow.")

    # Save best model + encoder as one joblib
    if best_model is not None:
        dump({"model": best_model, "encoder": le}, args.out)
        print(f"Best model ({best_model_name}) saved as {args.out} with weighted F1-score={best_f1:.4f}")


if __name__ == '__main__':
    main()
