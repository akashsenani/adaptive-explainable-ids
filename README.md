
# 🛡️ Adaptive Explainable Intrusion Detection System (X-AI IDS)

### Adaptive Model Selection • Intrusion Detection • Explainable AI • LLM-Based Security Analysis

<p align="center">
  <a href="https://adaptive-explainable-ids.streamlit.app/">
    <strong>🚀 Launch Live Application</strong>
  </a>
  &nbsp; | &nbsp;
  <a href="https://github.com/akashsenani/adaptive-explainable-ids">
    <strong>💻 GitHub Repository</strong>
  </a>
</p>

---

## 📌 Overview

The **Adaptive Explainable Intrusion Detection System (X-AI IDS)** is a machine learning-based cybersecurity project designed to detect potentially malicious network activity, select between candidate predictive models, and explain model decisions through Explainable Artificial Intelligence (XAI).

Traditional intrusion detection models can produce a prediction without making it easy for an analyst to understand which input features influenced that decision. Furthermore, the best-performing model may vary depending on the dataset and its feature distribution.

This project addresses these challenges through an integrated workflow combining:

- **Adaptive Model Selection:** Uses validation evaluation results to identify a preferred model for each supported dataset.
- **Deep Neural Network (DNN):** Learns nonlinear relationships in dataset features.
- **XGBoost:** Provides a gradient-boosted tree-based classification alternative.
- **SHAP:** Estimates feature contributions to model predictions.
- **LIME:** Generates local, model-agnostic explanations.
- **Google Gemini:** Converts structured prediction and explainability evidence into a human-readable threat assessment.
- **Streamlit:** Provides an interactive interface for configuring the analysis and viewing results.

The system brings model evaluation, inference, explainability, and natural-language interpretation together in a single application.

## 🔗 Project Links

| Resource | Link |
|---|---|
| Live Streamlit Application | https://adaptive-explainable-ids.streamlit.app/ |
| GitHub Source Code | https://github.com/akashsenani/adaptive-explainable-ids |

## 🎯 Problem Statement

Intrusion detection systems are an important component of cybersecurity, but machine learning-based detection presents several challenges.

1. **Model selection:** Different algorithms can perform differently across datasets.
2. **Interpretability:** A prediction alone does not explain which features contributed to the result.
3. **Analyst usability:** Raw feature-attribution values can be difficult to interpret without additional context.
4. **Dataset diversity:** Different intrusion detection datasets use different feature schemas and data distributions.
5. **Deployment constraints:** Large datasets and machine learning dependencies can make cloud deployment difficult.

X-AI IDS combines candidate-model evaluation, dataset-specific inference, explainability techniques, and LLM-generated interpretation to address these challenges in a unified research prototype.

## 🎯 Project Objectives

- Develop an intrusion detection workflow supporting multiple datasets.
- Evaluate DNN and XGBoost as candidate classification models.
- Use validation metrics to support dataset-specific model selection.
- Generate predictions and class probabilities for custom input records.
- Explain predictions using SHAP and LIME.
- Integrate Gemini to produce structured, readable explanations from XAI evidence.
- Provide an interactive dashboard for experimentation and analysis.
- Keep prediction outputs and explanations structured for future extensions.

---

## 🗂️ Supported Datasets

The project supports three dataset configurations.

| Dataset | Application Context |
|---|---|
| **NSL-KDD** | Network intrusion detection |
| **Farm-Flow** | Dataset-specific flow classification experiments |
| **IoMT-AI-IDS** | Intrusion detection in Internet of Medical Things environments |

Each dataset has its own expected input features and corresponding saved DNN and XGBoost model artifacts.

### Dataset-specific processing

The application loads a small feature-only background dataset for the selected configuration. This background data is used to populate default feature values and support explainability calculations.

Input features must match the feature schema expected by the selected dataset and model. Feature order, numerical representation, and any preprocessing applied during training must remain consistent with the saved models.

**Dataset note:** The dataset names above describe the configurations used in this project. Their exact provenance, licensing, and experimental characteristics should be documented separately when reporting research results.

---

## 🏗️ System Architecture

The system follows a pipeline that connects model selection with inference and explainability.

```text
┌──────────────────────────────────────┐
│          Dataset Selection           │
│       NSL-KDD / Farm-Flow / IoMT      │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│       Load Candidate Models          │
│              DNN / XGBoost           │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        Validation Evaluation         │
│ Accuracy • Precision • Recall • F1   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│       Adaptive Model Selection       │
│    Preferred model for the dataset   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│          Input Preparation           │
│  Feature schema and numeric values   │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│          Model Inference             │
│    Prediction + class probabilities  │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│       Explainable AI Layer           │
│              SHAP + LIME             │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        Gemini Explanation            │
│   Evidence-based natural language    │
└──────────────────┬───────────────────┘
                   │
                   ▼
┌──────────────────────────────────────┐
│        Streamlit Dashboard           │
│ Prediction • Evidence • Report       │
└──────────────────────────────────────┘
```

### Workflow explanation

**Step 1 — Dataset selection**

The user selects one of the supported dataset configurations. The application loads the corresponding feature schema, model configuration, and explainability background data.

**Step 2 — Candidate-model evaluation and selection**

The adaptive selection component uses the configured validation results to determine the preferred candidate model for the selected dataset.

The model preference is dataset-specific rather than assuming that DNN or XGBoost will always perform best.

**Step 3 — Input preparation**

The application presents dataset-specific feature inputs. The user can modify the feature values and submit a custom flow record for analysis.

**Step 4 — Prediction**

The selected model processes the input and returns a predicted class and associated class probabilities.

**Step 5 — Explainability**

When enabled, SHAP and LIME generate feature-level evidence to help explain the model's prediction.

**Step 6 — LLM interpretation**

When enabled, Gemini uses the available SHAP/LIME evidence and model information to produce a structured, natural-language threat assessment.

**Step 7 — Results presentation**

The dashboard presents the prediction, explanation evidence, and generated report. The user can inspect the submitted input and available raw explanation output.

> **Important implementation detail:** The application also allows the user to select DNN or XGBoost manually. If the manually selected model differs from the adaptive selector's preference, the dashboard displays a warning. Therefore, the adaptive preference and the model actually used for inference should be distinguished when interpreting results.

---

## 🧠 Machine Learning Models

### 1. Deep Neural Network (DNN)

The Deep Neural Network learns nonlinear relationships among the input features through successive neural-network layers.

**Role in this project:**

- Classifies input records using the saved Keras model.
- Produces a class-1 probability, with the complementary probability used for class 0.
- Participates in candidate-model evaluation and adaptive selection.
- Supports feature-level explanations through the configured explainability workflow.

The DNN is useful as a neural-network-based candidate against which the tree-based model can be evaluated.

### 2. XGBoost

XGBoost is a gradient-boosted decision-tree algorithm that combines multiple trees to produce a classification result.

**Role in this project:**

- Classifies input records using the saved XGBoost model.
- Produces class probabilities through `predict_proba`.
- Participates in model evaluation and adaptive selection.
- Supports tree-based feature attribution when the model format and installed SHAP version are compatible.

### 3. Why use both models?

Using two different model families enables an empirical comparison between a neural-network approach and a boosted-tree approach.

The comparison is useful because model performance, inference behavior, and explanation compatibility can differ across datasets.

The project does not assume that one model is universally superior. The preferred model should be determined from appropriate validation results.

---

## 🔀 Adaptive Model Selection

The adaptive selection component maintains a dataset-specific model preference based on configured evaluation results.

The candidate models are assessed using classification metrics such as:

| Metric | Meaning |
|---|---|
| **Accuracy** | Fraction of predictions that are correct |
| **Precision** | Fraction of predicted positive cases that are truly positive |
| **Recall** | Fraction of actual positive cases identified |
| **F1-score** | Harmonic mean of precision and recall |

The selection process uses these evaluation results and the configured selection logic to identify a preferred model.

### Why adaptive selection matters

A single fixed model may not provide the best results across all datasets. A dataset-specific selection process provides a systematic way to choose between the available candidates using measured validation performance.

### Selection versus live inference

The project uses a stored selection preference to initialize the dashboard's model choice. The user can then select a model manually.

Consequently, the stored adaptive preference is not the same as a fresh evaluation on every submitted input. A new evaluation would be needed if the system were extended to dynamically reselect the model at runtime.

**Research consideration:** When reporting model-selection quality, use validation results to choose a model and an independent test set to assess the final performance. Avoid selecting and evaluating a model on the same data.

---

## 🔍 Explainable AI (XAI)

Explainability is a core part of the project. It helps users inspect the features associated with a prediction rather than relying only on the predicted class.

### 1. SHAP — SHapley Additive exPlanations

SHAP assigns feature contributions to a prediction relative to a reference or background distribution.

It helps answer questions such as:

- Which features contributed most to this prediction?
- Which features pushed the model output toward the malicious class?
- Which features pushed the output away from the malicious class?

For XGBoost, the application attempts to use TreeSHAP when compatible with the installed libraries and saved model format. A model-agnostic Kernel SHAP fallback can be used when direct tree-based explanation fails.

For the DNN, Kernel SHAP uses background samples and a prediction function to estimate feature contributions.

**Interpretation:** SHAP contributions describe model behavior relative to the chosen explanation setup. They do not independently prove that a feature caused an attack.

### 2. LIME — Local Interpretable Model-Agnostic Explanations

LIME explains a specific prediction by approximating the model's behavior around the selected input record.

It helps identify which input features support or oppose the local prediction under LIME's approximation.

LIME complements SHAP because the methods use different explanation strategies. Their outputs may not always agree, particularly when features are correlated or the model behaves nonlinearly.

### 3. SHAP and LIME together

| SHAP | LIME |
|---|---|
| Estimates feature contributions relative to a background distribution | Approximates behavior around an individual input |
| Can provide a consistent feature-attribution framework | Provides a local, interpretable approximation |
| Can use tree-specific explainers for compatible tree models | Is model-agnostic when supplied with a suitable prediction function |

Using both methods offers complementary perspectives. Agreement between the methods can provide useful supporting evidence, but disagreement should be investigated rather than hidden.

---

## 🤖 Gemini-Based Natural-Language Explanation

The project integrates Google Gemini to transform structured explainability evidence into a more accessible threat assessment.

### Explanation workflow

1. The model generates a prediction and class probabilities.
2. SHAP and LIME produce feature-level evidence when explanation generation is enabled.
3. The application supplies the available evidence and model context to Gemini.
4. Gemini generates a structured explanation.
5. The dashboard presents the interpretation, supporting evidence, and recommended action.

The report can include:

- Threat assessment and predicted label.
- Dataset and selected model information.
- Evidence supporting the prediction.
- Evidence opposing the prediction.
- LIME feature evidence.
- Natural-language interpretation.
- Recommended action.

### Why use an LLM?

SHAP and LIME produce numerical or feature-oriented evidence that may be difficult for a non-specialist to interpret. An LLM can help translate that evidence into a readable explanation for review.

### Important safeguards

Gemini is an interpretation layer, not the source of the model's authoritative prediction. Its generated text may be incomplete or incorrect.

The application should be interpreted using the live model output and underlying XAI evidence. A generated explanation is not proof of malicious activity, and recommended actions should be checked against additional telemetry and security policies.

Gemini functionality also depends on a valid API key, network access, and the availability of the configured model.

---

## 🖥️ Streamlit Application

The interactive dashboard combines the main components into a single interface.

### Available workflow features

- Select a dataset.
- View the adaptive model preference.
- Select DNN or XGBoost for inference.
- Enter or adjust dataset-specific feature values.
- Run threat analysis on a custom record.
- View the predicted class and class probabilities.
- Enable SHAP and LIME explanation generation.
- Enable Gemini-based natural-language reporting.
- Inspect submitted input values.
- Review the generated threat assessment and raw LLM JSON where available.
- Switch between interface themes.

The dashboard is designed for experimentation and analysis. Predictions made from manually entered records should be treated as model outputs rather than verified ground truth.

---

## 🛠️ Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application logic and ML integration |
| Pandas | Tabular data handling |
| NumPy | Numerical operations |
| TensorFlow / Keras | DNN model loading and inference |
| XGBoost | Gradient-boosted classification |
| Scikit-learn | Classification utilities and evaluation |
| Joblib | Loading serialized XGBoost models |
| SHAP | Feature-attribution explanations |
| LIME | Local model-agnostic explanations |
| Streamlit | Interactive web interface |
| Google Gemini API | Natural-language interpretation |
| JSON | Model configuration and structured explanation data |
| Git / GitHub | Version control and source hosting |

---

## 📁 Repository and Model Artifacts

The project uses saved model artifacts, dataset-specific feature configurations, and small feature-only background files to support deployment.

Representative model artifacts used by the workflow include:

| Dataset | DNN artifact | XGBoost artifact |
|---|---|---|
| NSL-KDD | `NSL-KDD_DNN.keras` | `NSL-KDD_XGBoost.joblib` |
| Farm-Flow | `Farm-Flow_DNN.keras` | `Farm-Flow_XGBoost.joblib` |
| IoMT-AI-IDS | `IoMT_AI_IDS_DNN.keras` | `IoMT_AI_IDS_XGBoost.joblib` |

The application also relies on model-selection configuration, feature schemas, and dataset-specific explainability background files.

**Deployment note:** The complete raw datasets are not bundled with the cloud deployment because of their size. The application uses compact background samples for its configured inference and explanation workflow. The full training and evaluation data may need to be obtained and prepared separately to reproduce the original experiments.

The exact directory layout should be taken from the current repository, since files may be reorganized during development.

---

## ⚙️ Run Locally

### Prerequisites

- Python 3.11 is recommended for the current deployment setup.
- Git.
- The project's required Python dependencies.
- The required model artifacts, registry/configuration files, and background CSV files.
- A Gemini API key if LLM explanations are required.

### Step 1 — Clone the repository

```bash
git clone https://github.com/akashsenani/adaptive-explainable-ids.git
cd adaptive-explainable-ids
```

### Step 2 — Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux or macOS:

```bash
source .venv/bin/activate
```

### Step 3 — Install dependencies

If a `requirements.txt` file is present:

```bash
pip install -r requirements.txt
```

Use the dependency versions defined by the project. TensorFlow, XGBoost, SHAP, and other ML libraries must be compatible with the saved models and the application code.

### Step 4 — Configure Gemini

For local use, supply the API key using the configuration method supported by the application.

For example, using Windows PowerShell:

```powershell
$env:GEMINI_API_KEY="your_api_key_here"
```

Alternatively, use an appropriately configured local environment file if supported by the application.

Never commit a real API key or a secrets file to GitHub.

### Step 5 — Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL shown in the terminal.

> These commands assume that `app.py` and `requirements.txt` exist at the repository root. If your current repository uses different paths or filenames, use those actual paths. The required model and background artifacts must also be available before the application can run.

---

## ☁️ Live Deployment

The project is deployed using Streamlit Community Cloud.

**Launch the application:**  
https://adaptive-explainable-ids.streamlit.app/

For deployment or redeployment:

1. Connect the GitHub repository to Streamlit Community Cloud.
2. Select the application's entry-point script.
3. Configure a compatible Python version and dependencies.
4. Ensure that required model artifacts, registry files, and background CSVs are available.
5. Add the Gemini API key through Streamlit's secrets settings.
6. Test each dataset configuration and both candidate models.
7. Test SHAP, LIME, and Gemini separately so that failures in one optional explanation layer can be identified.

### Deployment considerations

- Large raw datasets should not be committed merely to support inference.
- Saved model files must be available at the paths expected by the application.
- Feature schemas must match the saved models.
- Library versions should be tested against the model formats.
- SHAP computation can be slower than prediction, especially with model-agnostic methods.
- LLM reporting requires a working API configuration.

---

## 📊 Evaluation and Reporting

The adaptive selection workflow uses validation metrics to compare candidate models.

For a rigorous experimental report, evaluate the selected model on a separate test set and report at least:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- False-positive rate and false-negative rate, where applicable
- Per-dataset results
- Per-model results
- Inference and explanation runtime, where relevant

Because the three datasets can differ in feature representation, label distribution, and difficulty, performance results should be reported separately for each dataset unless a justified common evaluation protocol is used.

**No universal performance figures are claimed in this README.** Add numerical results from the saved evaluation metrics or reproducible test runs rather than estimating or inventing them.

---

## 🔐 Security, Limitations, and Responsible Use

This project is a research and educational prototype. It is not, by itself, a validated production-grade intrusion prevention system.

- **Prediction uncertainty:** A predicted class or high probability does not guarantee that an attack has occurred.
- **False positives and negatives:** Legitimate activity may be flagged, and malicious activity may be missed.
- **Feature compatibility:** Incorrect feature order, preprocessing, or input ranges can lead to unreliable results.
- **Explanation limitations:** SHAP and LIME explain model behavior; they do not prove causality or ground truth.
- **LLM limitations:** Gemini may produce misleading or incomplete interpretations.
- **Dataset limitations:** Model performance depends on dataset quality, coverage, and distribution.
- **API security:** Keep API keys and credentials private.
- **Operational use:** Validate detections using logs, endpoint telemetry, network context, and established security procedures before taking action.

The system should be tested on representative data and reviewed before any real-world operational use.

---

## 🚀 Future Improvements

Potential extensions include:

- Dynamic model reselection based on new validation or monitoring results.
- Broader evaluation on independent datasets and unseen traffic.
- Calibration and monitoring of prediction probabilities.
- Drift detection and periodic model evaluation.
- More robust input validation and schema checks.
- Comparison of SHAP and LIME explanations.
- Explanation consistency and reliability evaluation.
- Structured security-report export.
- Latency and resource optimization for cloud deployment.
- Integration with additional security telemetry and analyst workflows.

These are potential enhancements and should not be interpreted as features already implemented.

---

## 👨‍💻 Author

**Akash Kumar Senani**  
B.Tech — Electronics and Computer Engineering  
VIT Chennai

GitHub: [@akashsenani](https://github.com/akashsenani)

## 📄 License

No open-source license is specified here. Add a `LICENSE` file if you intend to permit reuse or redistribution under a particular license. Without an applicable license, reuse remains subject to the relevant copyright law.

---

⭐ Explore the [live application](https://adaptive-explainable-ids.streamlit.app/) or visit the [GitHub repository](https://github.com/akashsenani/adaptive-explainable-ids) to review the project implementation.
