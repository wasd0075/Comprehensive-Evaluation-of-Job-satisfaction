# Comprehensive Evaluation of Job Satisfaction Using Deep Learning

> **Predicting employee satisfaction scores from Glassdoor reviews using DistilBERT + SHAP and Bi-LSTM, with explainable AI for transparent sentiment analysis.**

---

## Overview

This project evaluates the capability of deep learning models to predict employee job satisfaction from textual reviews posted on Glassdoor. Rather than relying solely on numeric ratings, the system analyses the full context of reviews — including pros, cons, headlines, and structured sub-ratings — to generate a satisfaction score on a scale of 1 to 5.

Two models are compared: a **Bidirectional LSTM (Bi-LSTM)** and a **DistilBERT** transformer model enhanced with **SHAP (Shapley Additive Explanations)** for interpretability. The research follows a **CRISP-DM** methodology and is designed to support job seekers, HR departments, and organisations in making data-driven decisions about workplace culture.

---

## Research Question

> *"How accurately can we predict employee job satisfaction from textual reviews using deep learning models, and what linguistic patterns and temporal trends correlate most strongly with high or low satisfaction?"*

---

## Key Features

- **5-Class Satisfaction Scoring** — Predicts employee satisfaction on a 1–5 scale, not just binary positive/negative
- **Dual-Model Comparison** — Bi-LSTM vs. DistilBERT evaluated head-to-head on the same dataset
- **Explainable AI via SHAP** — Word-level contribution scores reveal which linguistic patterns drive satisfaction predictions
- **Hybrid Input** — Combines textual reviews (headline, pros, cons) with structured metadata (sub-ratings, role, location, outlook)
- **Temporal Analysis** — Tracks satisfaction trends over time and across job titles, including pre/post-2020 shifts
- **Class Imbalance Handling** — Computed class weights applied during training

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.10.0 |
| Deep Learning Framework | TensorFlow 2.11.0 / Keras |
| Transformer Model | `DistilBERT` via Hugging Face Transformers |
| Sequential Model | Bidirectional LSTM (Bi-LSTM) |
| Explainability | SHAP (Shapley Additive Explanations) |
| Data Processing | `pandas`, `numpy` (<2.0) |
| Visualisation | `matplotlib`, `seaborn`, `wordcloud` |
| Environment | Google Colab / Jupyter Notebook |

---

## Model Architectures

### 1. Bi-LSTM
A multi-input Recurrent Neural Network that reads text in both forward and backward directions for richer contextual understanding.

```
Text Input ──► Embedding ──► Bi-LSTM ──┐
Job/Location ──► Embedding ──► LSTM ───┼──► Dense Layers + Dropout ──► Softmax (5 classes)
Structured Ratings ─────────────────────┘
```

- Tokenisation with padding at 95th percentile sequence length
- Manual grid search over: LSTM units, learning rate, dense layer size
- Early stopping + model checkpointing

### 2. DistilBERT + SHAP (BERTFusion Model)
A custom Keras model fusing DistilBERT contextual embeddings with structured feature inputs.

```
Text Input ──► DistilBERT Tokeniser ──► DistilBERT Encoder ──┐
Structured Features ──────────────────► Dense Projection ────┼──► Concatenate ──► LayerNorm ──► Dense + Dropout ──► Softmax (5 classes)
```

- Dynamic truncation and padding via DistilBERT tokeniser
- Manual grid search over: dense units (64/128), dropout (0.2/0.3), batch size (16/32)
- Fine-tuning: layers initially frozen, then unfrozen for domain adaptation
- SHAP applied post-training for per-word contribution scores

---

## Model Performance

### Bi-LSTM

| Metric | Score |
|---|---|
| Test Accuracy | 63.97% |
| Macro F1-Score | 0.62 |
| Weighted F1-Score | 0.65 |

*Strong on extreme classes (1 and 5); struggles with mid-scale ratings (2–4).*

### DistilBERT

| Metric | Score |
|---|---|
| Rounded Accuracy | 60.54% |
| Macro F1-Score | 0.45 |
| Weighted F1-Score | 0.51 |
| Best Class F1 (Class 0 & 4) | 0.72 / 0.74 |

*More consistent per-class accuracy across all rating levels compared to Bi-LSTM; better at capturing subtle linguistic signals.*

### Head-to-Head Summary

| Capability | Bi-LSTM | DistilBERT + SHAP |
|---|---|---|
| Extreme class accuracy | ✅ Strong | ✅ Strong |
| Mid-scale accuracy | ⚠️ Weak | ✅ More consistent |
| Model explainability | ❌ None | ✅ SHAP word-level |
| Contextual understanding | ⚠️ Sequential only | ✅ Bidirectional transformer |

---

## Exploratory Data Analysis

The notebook includes the following EDA outputs:

- **Word Cloud** — Frequent words in pros/cons sections (e.g., *flexibility*, *teamwork* vs. *workload*)
- **Correlation Heatmap** — Overall rating most strongly correlated with culture & values
- **Average Sub-ratings by Role** — Role-specific satisfaction strengths and weaknesses
- **Sub-ratings by Outlook** — Employees with positive outlook consistently rate higher
- **Missing Value Plot** — Feature-level missingness visualised before imputation

---

## SHAP Explainability

SHAP was integrated with the DistilBERT model to provide word-level transparency:

- **Red tokens** → positive contributors to satisfaction score
- **Blue tokens** → negative contributors
- **Top contributing words** include: *environment*, *working*, *maintain*, *training*, *standard*
- Enables HR teams to identify which specific phrases in reviews drive satisfaction predictions

---

## Temporal Insights

- Satisfaction scores show a **notable increase after 2020**, potentially linked to the adoption of remote/hybrid working during the COVID-19 era
- Satisfaction trends tracked across multiple job titles over time reveal role-specific fluctuations and long-term patterns

---

## Dataset

- **Source:** Glassdoor job reviews dataset (~50,000 records)
- **Key Fields Used:**
  - `headline`, `pros`, `cons` — textual inputs
  - `overall_rating` — target variable (1–5)
  - `work_life_balance`, `culture_values`, `diversity_inclusion`, `career_opp`, `comp_benefits`, `senior_mgmt` — structured sub-ratings
  - `job_title`, `location`, `recommend`, `ceo_approv`, `outlook`, `employment_status` — categorical features

> **Note:** The dataset is not included in this repository. Supply your own Glassdoor reviews CSV and update the file path in the notebook accordingly.

---

## Getting Started

### Prerequisites

```bash
pip install transformers tensorflow==2.11.0 "numpy<2.0" shap pandas matplotlib seaborn wordcloud scikit-learn
```

> ⚠️ NumPy must be **below version 2.0** for TensorFlow 2.11.0 compatibility.

### Running the Notebook

1. Clone this repository
2. Upload your Glassdoor reviews CSV to Google Drive
3. Open `Glassdoor_50k_Final_Deep_Learning.ipynb` in Google Colab
4. Update the dataset path in the data loading cell
5. Run all cells sequentially — models train in order: Bi-LSTM → DistilBERT → SHAP

---

## Methodology

This project follows the **CRISP-DM** (Cross-Industry Standard Process for Data Mining) framework:

1. **Business Understanding** — Define the satisfaction prediction problem and stakeholder needs
2. **Data Understanding** — EDA, word clouds, correlation analysis, missing value inspection
3. **Data Preparation** — Text cleaning, tokenisation, label encoding, feature normalisation, class weight computation
4. **Modelling** — Bi-LSTM and DistilBERT + SHAP training with hyperparameter tuning
5. **Evaluation** — Classification reports, confusion matrices, per-class accuracy, SHAP summaries
6. **Deployment** — Framework designed for integration into HR tools and employee feedback systems

---

## Future Work

- Explore full BERT and RoBERTa models (resource constraints limited this study to DistilBERT)
- Develop self-learning agents that continuously retrain on new review data
- Expand temporal analysis with finer-grained time windows
- Integrate cross-platform review data (LinkedIn, Indeed) for broader generalisation

---

## References

1. Gaye et al., *Sentiment Classification for Employees Reviews Using RV-SGDC*, PeerJ Computer Science, 2021
2. Hussein, *A survey on sentiment analysis challenges*, Computational Intelligence, 2020
3. Sivakumar et al., *Sentiment Analysis of Glassdoor Reviews Using Supervised ML*, IEEE, 2023
4. Sibikanna et al., *Predicting Employee Job Satisfaction Using Vector Space Model*, Springer, 2024
5. Dhanalakshmi & Devi, *Adaptive Cognitive Intelligence in Analyzing Employee Feedback Using LSTM*, JIFS, 2020
6. Shilpa & Sudha, *Hybrid ML and DL Models for Sentiment Analysis in HR Reviews*, ITSE, 2024
7. Talaat, *Sentiment analysis classification system using hybrid BERT models*, Journal of Big Data, 2023
8. Lin & Nuha, *Sentiment Analysis Using Transformer-Based Hybrid Models*, IEEE, 2023
9. Kusal et al., *Sentiment Analysis of Product Reviews: Deep Learning vs Transformers*, Springer, 2024
10. Bhuvaneshwari & Seethalakshmi, *Sentiment analysis using Bi-LSTM self-attention CNN*, Multimedia Tools, 2023

---
