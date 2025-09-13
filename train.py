#!/usr/bin/env python3
"""Train script converted from `train_model.ipynb`.

Saves a joblib file containing {'model': model, 'encoder': label_encoder} by default.
"""
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Train an XGBoost classifier on a CSV dataset')
    parser.add_argument('--csv', type=str, default='iris.csv', help='Path to input CSV file')
    parser.add_argument('--target', type=str, default='variety', help='Name of the target column')
    parser.add_argument('--out', type=str, default='model_and_encoder.joblib', help='Output joblib path')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set fraction')
    parser.add_argument('--random-state', type=int, default=42, help='Random seed')
    parser.add_argument('--n-estimators', type=int, default=100, help='Number of estimators for XGB')
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        try:
            # prefer xgboost if available
            from xgboost import XGBClassifier
        except Exception:
            XGBClassifier = None
        from joblib import dump
    except Exception as e:
        print('Missing dependency:', e, file=sys.stderr)
        sys.exit(2)

    df = pd.read_csv(args.csv)
    if args.target not in df.columns:
        print(f"Target column '{args.target}' not found in {args.csv}", file=sys.stderr)
        sys.exit(2)

    X = df.drop(args.target, axis=1)
    y = df[args.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    if XGBClassifier is None:
        print('xgboost is not installed. Install xgboost to use XGBClassifier.', file=sys.stderr)
        sys.exit(2)

    model = XGBClassifier(n_estimators=args.n_estimators, random_state=args.random_state, eval_metric='logloss')
    model.fit(X_train, y_train_enc)

    dump({'model': model, 'encoder': le}, args.out)
    print(f'Model trained and saved as {args.out}')


if __name__ == '__main__':
    main()
