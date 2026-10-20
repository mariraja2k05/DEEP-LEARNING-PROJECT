from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_processing import build_student_summary, build_weekly_clicks, ensure_oulad_data

ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT_DIR / 'models'
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def quantile_engagement_label(values: np.ndarray) -> str:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 'Low'
    value_total = float(values.sum())
    if np.allclose(values, values[0]):
        return 'Medium'
    return 'Low'


def build_student_activity_sequences(data_dir: Path | str | None = None, input_weeks: Tuple[int, ...] = (0, 1, 2, 3), target_weeks: Tuple[int, ...] = (4, 5, 6)) -> Dict[str, object]:
    data_dir = Path(data_dir) if data_dir is not None else ensure_oulad_data()
    weekly = build_weekly_clicks(data_dir)
    student_summary = build_student_summary(data_dir)

    pivot = weekly.pivot_table(
        index=['code_module', 'code_presentation', 'id_student'],
        columns='week',
        values='total_clicks',
        aggfunc='sum',
        fill_value=0,
    ).reset_index()

    if pivot.empty:
        raise ValueError('No valid week activity data found in studentVle.csv')

    # Normalize week columns to strings so the model uses the actual OULAD period layout.
    week_columns = [col for col in pivot.columns if col not in ['code_module', 'code_presentation', 'id_student']]
    for week in week_columns:
        pivot[str(week)] = pivot[week].astype(float)
    pivot = pivot[['code_module', 'code_presentation', 'id_student'] + [str(w) for w in sorted(int(w) for w in week_columns)]].copy()

    max_week = max(int(col) for col in pivot.columns if col not in ['code_module', 'code_presentation', 'id_student'])
    effective_input_weeks = tuple(w for w in input_weeks if w <= max_week)
    effective_target_weeks = tuple(w for w in target_weeks if w <= max_week and w > max(effective_input_weeks, default=-1))

    if not effective_input_weeks or not effective_target_weeks:
        raise ValueError('Insufficient weekly coverage for the selected input/target windows.')

    input_columns = [str(w) for w in effective_input_weeks]
    target_columns = [str(w) for w in effective_target_weeks]

    sequence_df = pivot.copy()
    sequence_df = sequence_df.merge(
        student_summary[['code_module', 'code_presentation', 'id_student', 'final_result']],
        on=['code_module', 'code_presentation', 'id_student'],
        how='left',
    )

    rows = []
    for _, row in sequence_df.iterrows():
        input_vector = row[input_columns].astype(float).to_numpy(dtype=float)
        target_vector = row[target_columns].astype(float).to_numpy(dtype=float)
        if input_vector.size == 0 or target_vector.size == 0:
            continue
        if np.sum(input_vector) == 0 and np.sum(target_vector) == 0:
            continue

        rows.append({
            'code_module': row['code_module'],
            'code_presentation': row['code_presentation'],
            'id_student': int(row['id_student']),
            'input_sequence': input_vector,
            'target_sequence': target_vector,
            'total_clicks_input': float(input_vector.sum()),
            'total_clicks_target': float(target_vector.sum()),
            'mean_clicks_input': float(input_vector.mean()),
            'mean_clicks_target': float(target_vector.mean()),
            'active_weeks_input': int((input_vector > 0).sum()),
            'active_weeks_target': int((target_vector > 0).sum()),
            'max_week_clicks': float(input_vector.max()),
        })

    labels_df = pd.DataFrame(rows)
    if labels_df.empty:
        raise ValueError('No valid student-course sequences were produced from OULAD activity data.')

    label_order = ['Low', 'Medium', 'High']
    q33, q66 = np.quantile(labels_df['total_clicks_target'].to_numpy(dtype=float), [0.33, 0.66])
    labels_df['engagement_label'] = labels_df['total_clicks_target'].apply(
        lambda total: 'Low' if total <= q33 else ('Medium' if total <= q66 else 'High')
    )
    label_map = {label: idx for idx, label in enumerate(label_order)}
    labels_df['label_index'] = labels_df['engagement_label'].map(label_map)
    labels_df = labels_df[labels_df['label_index'].notna()].copy()
    if labels_df.empty:
        raise ValueError('No valid engagement labels were produced from the OULAD target window.')

    ann_features = labels_df[
        ['total_clicks_input', 'mean_clicks_input', 'active_weeks_input', 'max_week_clicks']
    ].copy()

    model_bundle = {
        'labels': labels_df,
        'input_weeks': list(effective_input_weeks),
        'target_weeks': list(effective_target_weeks),
        'feature_columns': list(ann_features.columns),
        'sequence_length': len(effective_input_weeks),
        'label_order': label_order,
    }
    return model_bundle


def save_model_bundle(bundle: Dict[str, object], output_name: str) -> Path:
    output_path = MODEL_DIR / f'{output_name}.json'
    payload = {
        'input_weeks': bundle['input_weeks'],
        'target_weeks': bundle['target_weeks'],
        'sequence_length': bundle['sequence_length'],
        'label_order': bundle['label_order'],
        'feature_columns': bundle['feature_columns'],
        'n_records': int(len(bundle['labels'])),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return output_path


def build_feature_matrix() -> Dict[str, object]:
    bundle = build_student_activity_sequences()
    labels = bundle['labels'].copy()
    X = labels[['total_clicks_input', 'mean_clicks_input', 'active_weeks_input', 'max_week_clicks']].to_numpy(dtype=float)
    y = labels['label_index'].to_numpy(dtype=int)
    input_sequences = np.stack(labels['input_sequence'].to_numpy())
    X_seq = input_sequences.reshape(-1, input_sequences.shape[1], 1)
    return {
        'X_ann': X,
        'y': y,
        'X_seq': X_seq,
        'labels': labels,
        'feature_columns': list(labels[['total_clicks_input', 'mean_clicks_input', 'active_weeks_input', 'max_week_clicks']].columns),
        'label_order': bundle['label_order'],
    }
