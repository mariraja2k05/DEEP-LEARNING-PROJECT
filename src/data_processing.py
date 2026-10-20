from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data" / "oulad"
REQUIRED_FILES = [
    "studentInfo.csv",
    "studentRegistration.csv",
    "studentAssessment.csv",
    "studentVle.csv",
    "assessments.csv",
    "courses.csv",
    "vle.csv",
]


def ensure_oulad_data() -> Path:
    data_dir = DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"OULAD directory not found: {data_dir}")

    missing = [name for name in REQUIRED_FILES if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required OULAD files: {missing}")
    return data_dir


def load_oulad_tables(data_dir: Path | str | None = None) -> Dict[str, pd.DataFrame]:
    data_dir = Path(data_dir) if data_dir is not None else ensure_oulad_data()
    tables = {}
    for name in REQUIRED_FILES:
        path = data_dir / name
        tables[name.replace('.csv', '')] = pd.read_csv(path)
    return tables


def summarize_dataframe(name: str, df: pd.DataFrame) -> dict:
    return {
        "file": name,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "columns": list(df.columns),
        "dtypes": {k: str(v) for k, v in df.dtypes.to_dict().items()},
        "missing_values": {k: int(v) for k, v in df.isna().sum().to_dict().items() if v > 0},
        "duplicate_rows": int(df.duplicated().sum()),
    }


def write_summary_json(summary: List[dict], output_path: Path | str) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding='utf-8')
    return output_path


def build_weekly_clicks(data_dir: Path | str | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir is not None else ensure_oulad_data()
    tables = load_oulad_tables(data_dir)

    student_vle = tables['studentVle'].copy()
    student_vle = student_vle[['code_module', 'code_presentation', 'id_student', 'id_site', 'date', 'sum_click']].copy()
    student_vle['date'] = pd.to_numeric(student_vle['date'], errors='coerce')
    student_vle['sum_click'] = pd.to_numeric(student_vle['sum_click'], errors='coerce').fillna(0)
    student_vle = student_vle.dropna(subset=['date']).copy()
    student_vle['week'] = student_vle['date'].clip(lower=0).astype(int) // 7

    weekly = (
        student_vle.groupby(['code_module', 'code_presentation', 'id_student', 'week'], as_index=False)['sum_click']
        .sum()
        .rename(columns={'sum_click': 'total_clicks'})
    )
    return weekly


def build_student_summary(data_dir: Path | str | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir is not None else ensure_oulad_data()
    tables = load_oulad_tables(data_dir)
    student_info = tables['studentInfo'].copy()
    student_info = student_info[['code_module', 'code_presentation', 'id_student', 'final_result']].copy()

    weekly = build_weekly_clicks(data_dir)
    student_week_summary = (
        weekly.groupby(['code_module', 'code_presentation', 'id_student'], as_index=False)
        .agg(
            total_clicks=('total_clicks', 'sum'),
            active_weeks=('week', 'nunique'),
            weeks_with_activity=('week', 'count'),
            max_week_clicks=('total_clicks', 'max'),
            mean_week_clicks=('total_clicks', 'mean'),
            std_week_clicks=('total_clicks', 'std')
        )
    )
    student_week_summary['std_week_clicks'] = student_week_summary['std_week_clicks'].fillna(0.0)

    student_summary = student_info.merge(student_week_summary, on=['code_module', 'code_presentation', 'id_student'], how='left')
    student_summary['total_clicks'] = student_summary['total_clicks'].fillna(0.0)
    student_summary['active_weeks'] = student_summary['active_weeks'].fillna(0)
    student_summary['mean_week_clicks'] = student_summary['mean_week_clicks'].fillna(0.0)
    student_summary['max_week_clicks'] = student_summary['max_week_clicks'].fillna(0.0)
    student_summary['std_week_clicks'] = student_summary['std_week_clicks'].fillna(0.0)
    return student_summary


def build_processed_dataset(data_dir: Path | str | None = None) -> pd.DataFrame:
    student_summary = build_student_summary(data_dir)
    return student_summary.sort_values(['code_module', 'code_presentation', 'id_student']).reset_index(drop=True)
