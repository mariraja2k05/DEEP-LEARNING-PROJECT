# EduTrack — Student Engagement Prediction Using Deep Learning

This project uses the Open University Learning Analytics Dataset (OULAD) to predict student engagement using ANN, LSTM, and GRU models. The working application keeps the original project structure, preserves the dashboard styling, and integrates the actual saved deep-learning models and evaluation artifacts from the project outputs folder.

## Dataset

The primary dataset is the OULAD collection stored under `data/oulad/` and derived from the verified source files:

- `studentInfo.csv`
- `studentRegistration.csv`
- `studentAssessment.csv`
- `studentVle.csv`
- `assessments.csv`
- `courses.csv`
- `vle.csv`

These files are read through the processing utilities in `src/data_processing.py`.

## Data processing

The pipeline performs the following steps:

1. Loads the seven OULAD tables.
2. Aggregates clicks into weekly student activity by `id_student`, `code_module`, `code_presentation`, and `week`.
3. Builds a student-level summary for activity volume and active weeks.
4. Creates input sequences from the first four study weeks.
5. Constructs a future-engagement target from the later activity window.
6. Uses a stratified hold-out split for evaluation.

## Feature engineering

The models use:

- `total_clicks_input`
- `mean_clicks_input`
- `active_weeks_input`
- `max_week_clicks`

For the LSTM and GRU models, the weekly click totals are reshaped into a 4-step sequence with one feature per week.

## Engagement label definition

The engagement label is a three-class target: `Low`, `Medium`, and `High`.

The label is generated from the later target-window total clicks using quantile thresholds, which produces a class distribution suitable for comparative deep-learning evaluation. This is intended as an engagement intensity forecast from earlier activity, not as a direct copy of the same activity window.

## Model architecture

- ANN: a dense feed-forward network with ReLU layers and softmax output
- LSTM: a recurrent model operating on 4 weekly steps
- GRU: a recurrent model operating on 4 weekly steps

All three models are trained on the same OULAD feature representation and the same class labels.

## Evaluation methodology

The evaluation uses a stratified 80/20 train-test split and reports:

- accuracy
- macro precision
- macro recall
- macro F1
- confusion matrix

The verified saved metrics can be found in `outputs/metrics/`.

## Verified results

The actual saved metrics as of this project revision are:

- ANN: accuracy = 0.65117, macro F1 = 0.65263
- LSTM: accuracy = 0.65513, macro F1 = 0.65718
- GRU: accuracy = 0.65857, macro F1 = 0.65858

These values were read directly from the saved JSON evaluation files and are the numbers used by the dashboard.

## Project scripts

- `src/data_processing.py` — OULAD loading and table preparation
- `src/feature_engineering.py` — weekly sequence construction and label creation
- `src/train_ann.py` — ANN training and artifact export
- `src/train_lstm.py` — LSTM training and artifact export
- `src/train_gru.py` — GRU training and artifact export
- `streamlit/EduTrack_App.py` — dashboard integration

## Run the dashboard

```bash
cd /Users/marirajasankar/Downloads/SEP
source .venv/bin/activate
streamlit run streamlit/EduTrack_App.py --server.headless true --server.port 8501
```

## Known limitations

- The engagement classes are derived from click intensity and are best treated as behavioral engagement bands rather than independently measured ground-truth outcomes.
- The model performance is moderate, which is expected for a real-world educational dataset with high class overlap.
- The project was intentionally built around the verified OULAD files and does not rely on the older synthetic demo artifacts.
