# Adaptive Explainable Intrusion Detection System

Streamlit deployment for three IDS datasets (NSL-KDD, Farm-Flow, and IoMT-AI-IDS), with DNN/XGBoost inference, SHAP/LIME local explanations, and optional Gemini-generated narrative explanations.

## Important: create local background samples before deploying

Raw datasets are intentionally excluded from this repository. From the original project root (the directory containing `0_Datasets`), run:

```bash
python adaptive-ids-deploy/prepare_backgrounds.py
```

This creates 100-row, feature-only CSV files in `adaptive-ids-deploy/backgrounds/` for SHAP/LIME reference data. They do not contain labels and are not full raw datasets. Review the files before committing. If your source CSVs use different paths, update the `DATASETS` mapping in the script.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

For local Gemini use, place `GEMINI_API_KEY=...` in a local `.env` file (never commit it). On Streamlit Community Cloud, add `GEMINI_API_KEY` in App settings → Secrets. Optional `GEMINI_MODEL` defaults to `gemini-2.5-flash`.

## Deploy

Push this folder as the GitHub repository root. In Streamlit Community Cloud, select `app.py` as the entry point and configure secrets. The app expects model files in `models/`, feature definitions in `model_registry.json`, and small reference samples in `backgrounds/`.

## Data handling

Do not commit raw training/test datasets, notebooks, `.env`, prediction exports, or private credentials. The small background samples are derived feature-only artifacts needed for local explanation and should be treated as data artifacts too; ensure they are permitted for publication before pushing.
