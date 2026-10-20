from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT_DIR / 'outputs' / 'metrics'
MODEL_DIR = ROOT_DIR / 'models'
METRICS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_metric_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def build_comparison_table() -> pd.DataFrame:
    files = [
        MODEL_DIR / 'ann_label_order.json',
        MODEL_DIR / 'lstm_label_order.json',
        MODEL_DIR / 'gru_label_order.json',
    ]
    data = []
    for model_name in ['ann', 'lstm', 'gru']:
        metric_path = METRICS_DIR / f'{model_name}_metrics.json'
        if not metric_path.exists():
            continue
        metrics = load_metric_file(metric_path)
        row = {
            'model': model_name.upper(),
            'accuracy': metrics.get('accuracy', 0.0),
            'precision_macro': metrics.get('precision_macro', 0.0),
            'recall_macro': metrics.get('recall_macro', 0.0),
            'f1_macro': metrics.get('f1_macro', 0.0),
            'label_order': metrics.get('label_order', ['Low', 'Medium', 'High']),
        }
        data.append(row)

    if not data:
        raise FileNotFoundError('No metrics files were found; train the models before evaluating them.')

    df = pd.DataFrame(data)
    comparison_path = METRICS_DIR / 'model_comparison.csv'
    df.to_csv(comparison_path, index=False)
    print(df.to_string(index=False))
    return df


if __name__ == '__main__':
    build_comparison_table()
