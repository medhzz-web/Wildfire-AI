import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# Relative import compatibility
try:
    from src.data_processing import load_data, preprocess_data, FEATURE_COLS
except ImportError:
    from data_processing import load_data, preprocess_data, FEATURE_COLS


def train_and_evaluate_model(data_path: str = None, model_path: str = None):
    """
    Trains a RandomForestClassifier model on WildFire dataset and evaluates performance.
    Saves the trained model dictionary to joblib.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    if data_path is None:
        data_path = os.path.join(base_dir, 'data', 'sample_data.csv')
    if model_path is None:
        model_path = os.path.join(base_dir, 'models', 'model.joblib')

    # Load & preprocess
    df = load_data(data_path)
    df_clean, X, y = preprocess_data(df)

    # Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Model training with reproduciblity
    rf_model = RandomForestClassifier(
        n_estimators=120,
        max_depth=10,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # Predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    y_prob_test = rf_model.predict_proba(X_test)[:, 1]

    # Metrics calculation
    metrics = {
        'train_accuracy': float(accuracy_score(y_train, y_pred_train)),
        'test_accuracy': float(accuracy_score(y_test, y_pred_test)),
        'precision': float(precision_score(y_test, y_pred_test, zero_division=0)),
        'recall': float(recall_score(y_test, y_pred_test, zero_division=0)),
        'f1_score': float(f1_score(y_test, y_pred_test, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, y_prob_test)),
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test))
    }

    cm = confusion_matrix(y_test, y_pred_test).tolist()

    # Feature Importance
    feature_importances = dict(
        zip(FEATURE_COLS, np.round(rf_model.feature_importances_, 4))
    )

    # Model Artifact Bundle
    model_bundle = {
        'model': rf_model,
        'feature_cols': FEATURE_COLS,
        'metrics': metrics,
        'confusion_matrix': cm,
        'feature_importances': feature_importances,
        'X_test': X_test,
        'y_test': y_test,
        'y_prob_test': y_prob_test
    }

    # Ensure output directory exists
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model_bundle, model_path)
    print(f"Model successfully saved to {model_path}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.2%}, F1-Score: {metrics['f1_score']:.4f}")

    return model_bundle


def load_saved_model(model_path: str = None):
    """
    Loads saved model bundle from joblib. If file doesn't exist, auto-trains model.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if model_path is None:
        model_path = os.path.join(base_dir, 'models', 'model.joblib')

    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}. Training new model...")
        return train_and_evaluate_model(model_path=model_path)

    try:
        model_bundle = joblib.load(model_path)
        print("Model bundle loaded successfully.")
        return model_bundle
    except Exception as e:
        print(f"Error loading model: {e}. Retraining model...")
        return train_and_evaluate_model(model_path=model_path)


if __name__ == '__main__':
    train_and_evaluate_model()
