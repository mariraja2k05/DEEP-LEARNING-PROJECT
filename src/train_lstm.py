from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout, LSTM, Dense

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.feature_engineering import build_feature_matrix

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / 'models'
OUTPUT_METRICS_DIR = ROOT_DIR / 'outputs' / 'metrics'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    bundle = build_feature_matrix()
    X_seq = bundle['X_seq']
    y = bundle['y']
    label_order = bundle['label_order']

    X_train, X_test, y_train, y_test = train_test_split(
        X_seq,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    model = Sequential([
        keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(len(label_order), activation='softmax'),
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=64,
        callbacks=[early_stop],
        verbose=0,
    )

    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision_macro': float(precision_score(y_test, y_pred, average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y_test, y_pred, average='macro', zero_division=0)),
        'f1_macro': float(f1_score(y_test, y_pred, average='macro', zero_division=0)),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred, labels=[0, 1, 2], target_names=label_order, output_dict=True),
        'label_order': label_order,
        'history': {k: [float(v) for v in vals] for k, vals in history.history.items()},
    }

    model.save(MODEL_DIR / 'lstm_engagement.keras')
    (MODEL_DIR / 'lstm_label_order.json').write_text(json.dumps(label_order, indent=2), encoding='utf-8')
    (OUTPUT_METRICS_DIR / 'lstm_metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')

    print('LSTM training complete')
    print(json.dumps({
        'accuracy': metrics['accuracy'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro': metrics['recall_macro'],
        'f1_macro': metrics['f1_macro'],
    }, indent=2))


if __name__ == '__main__':
    main()
