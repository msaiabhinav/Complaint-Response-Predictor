# Complaint Response Predictor

Machine-learning pipeline that predicts how financial institutions respond to consumer complaints using **CFPB complaint metadata + complaint narratives**.

The project combines NLP and structured-data modeling: complaint narratives are cleaned and represented with TF-IDF, operational/categorical features are engineered and encoded, and an XGBoost multiclass classifier predicts the company's response.

[Open the original Colab notebook](https://colab.research.google.com/drive/1peeevT6uUVuVnBj28bbX_VYqwfaiLpDU)

## Problem

Consumer-finance organizations receive large volumes of complaints covering products, issues, companies, locations and free-text narratives. The goal is to predict the **company response to the consumer** from information available in each complaint.

The supplied CFPB sample contains **100,001 complaints** using a fixed train/test partition:

- Training: 80,000 complaints
- Test: 20,001 complaints
- Original target: eight company-response categories
- Modeling target: three consolidated response classes

### Final target classes

1. `Closed with explanation`
2. `Closed with non-monetary relief`
3. `Grouped` — combines Closed, Closed with monetary relief, In progress and Untimely response

## ML Pipeline

```text
CFPB Complaint Data
        |
        v
EDA + Missing-Value Analysis
        |
        v
Data Cleaning
        |
        +--> categorical missing flags / Unknown values
        +--> complaint narrative cleanup
        +--> date normalization
        |
        v
Feature Engineering
        |
        +--> narrative length
        +--> complaint season
        +--> company response time
        +--> date components
        +--> grouped sub-products
        |
        v
NLP
        |
        +--> lowercase / regex cleaning
        +--> stop-word removal
        +--> Porter stemming
        +--> TF-IDF (5,000 features)
        |
        v
Structured Feature Encoding + Scaling
        |
        v
Sparse Structured + TF-IDF Feature Matrix
        |
        +----------------------+
        |                      |
        v                      v
     XGBoost               SMOTE
                               |
                               v
                         XGBoost + SMOTE
```

## Exploratory Analysis

The notebook investigates:

- Target-class distribution
- Complaint narrative lengths
- Most frequent complaint issues
- Companies receiving the most complaints
- Products generating the most complaints
- Missing-value patterns
- Companies with the fastest average response times
- Most common narrative keywords using TF-IDF

## Feature Engineering

### Text features

Complaint narratives are normalized by removing punctuation, numbers and special characters, removing English stop words, and applying Porter stemming. A `TfidfVectorizer` generates up to **5,000 text features**.

### Structured features

The pipeline derives or processes:

- Narrative word count
- State and sub-product missing indicators
- Complaint season
- Company response time
- Received-date year/month/day
- Grouped sub-products
- Encoded categorical attributes
- Standardized response-time and narrative-length features

Structured and TF-IDF matrices are combined as a SciPy sparse matrix before modeling.

## Model Results

### XGBoost baseline

| Metric | Train | Test |
|---|---:|---:|
| Accuracy | **85.15%** | **82.02%** |
| Weighted F1 | **0.85** | **0.81** |
| Macro F1 | 0.76 | 0.61 |

The baseline performs strongly overall but has low recall (**15%**) on the minority `Grouped` class.

### XGBoost + SMOTE

SMOTE balances the training distribution from:

```text
Before: [53,766, 23,073, 3,161]
After:  [53,766, 53,766, 53,766]
```

| Metric | Train | Test |
|---|---:|---:|
| Accuracy | **83.56%** | **80.20%** |
| Weighted F1 | **0.83** | **0.80** |
| Macro F1 | 0.75 | **0.64** |

SMOTE sacrifices some overall accuracy but improves minority-class test recall from **15% → 32%** and minority-class F1 from **0.24 → 0.33**.

## Key Takeaway

The experiment demonstrates the trade-off between aggregate predictive performance and minority-class detection. The baseline XGBoost model achieves the highest overall test accuracy and weighted F1, while SMOTE produces more balanced behavior across response classes.

## Repository Structure

```text
Complaint-Response-Predictor/
├── notebooks/              # Original experimentation notebook
├── src/                    # Reusable project code
├── data/                   # Dataset location (ignored by Git)
├── reports/
│   └── figures/            # Generated visualizations
├── requirements.txt
├── .gitignore
└── README.md
```

## Tech Stack

**Python · Pandas · NumPy · scikit-learn · XGBoost · imbalanced-learn · NLTK · TF-IDF · SciPy · Matplotlib · Seaborn · Squarify · Google Colab**

## Installation

```bash
git clone https://github.com/msaiabhinav/Complaint-Response-Predictor.git
cd Complaint-Response-Predictor
python -m venv .venv
pip install -r requirements.txt
```

The original notebook expects `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`, and `df_sample.csv`. Dataset files are intentionally excluded from Git.

## Author

**Sai Abhinav Mullapudi**  
University of Connecticut
