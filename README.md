# Credit Risk Analysis and Expected Loss Modeling 

## Project Overeview 
Advanced credit risk prediction framework implementing PD (Probability of Default), LGD (Loss Given Default), and EAD (Exposure at Default) modeling with comprehensive stress testing scenarios.

## Key Features 
- **Predictive Modeling**: XGBoost classifier achieving 0.95 ROC-AUC score
- **Risk Metrics**: Complete PD, LGD, EAD framework
- **Stress Testing**: Multiple scenario analysis (Baseline, Mild, Moderate, Heavy)
- **Visualizations**: Grade-level risk distribution and stress impact analysis

## Technical Stack'

### Core Technologies
- **Machine Learning**: scikit-learn 1.3.2, XGBoost 2.0.3
- **Data Processing**: pandas 2.1.4, numpy 1.26.2
- **Visualization**: matplotlib 3.8.2
- **Environment**: Python 3.9+

### ML Pipeline Components
```python
Pipeline Architecture:
├── Feature Engineering
│   ├── Custom transformers (grade mapping)
│   ├── SimpleImputer (median)
│   └── OneHotEncoder (categorical features)
├── Preprocessing
│   ├── StandardScaler (for Logistic Regression)
│   └── ColumnTransformer (parallel processing)
└── Models
    ├── Logistic Regression (baseline)
    └── XGBoost (production model)
```

## Model Performance
| Metric | Logistic Regression | XGBoost | Improvement |
|--------|---------------------|-------------|-------------|
| **ROC-AUC** | 0.87 | 0.95 | +9.2% |
| **Precision** | 0.75 | 0.97 | +29.3% |
| **Recall** | 0.52 | 0.73 | +40.4% |
| **F1 Score** | 0.61 | 0.83 | +36.1% |

## Performance Highlights
- **95% ROC-AUC**: Excellent discrimination between defaulters and non-defaulters
- **97% Precision**: Only 3% false positive rate
- **73% Recall**: Captures 73% of actual defaults
- **83% F1 Score**: Optimal balance between precision and recall

## Portfolio Analysis

### Key Metrics
- Portfolio Default Rate (PD): 21.68%
- Average Exposure (EAD): $9,670.39
- Total Expected Loss: $7,351,864.07
- Average Expected Loss per Loan: $1,128.11

### Loss Distribution by Loan Grade
| Grade | Loss Contribution | 
|-------|------------------|
| D | 37.93%| 
| B | 17.77% |  
| C | 15.57% |  
| E | 13.08% |  
| A | 6.22% |  
| F | 5.06% |  
| G | 4.38% | 

### Stress Testing Results 
| Scenario | PD Multiplier | LGD Adjustment | Total Expected Loss | % Increase from Baseline |
|----------|---------------|----------------|---------------------|--------------------------|
| **Baseline** | 1.0× | +0% | $7,351,864 | - |
| **Mild Stress** | 1.2× | +5% | $9,753,931 | +32.7% |
| **Moderate Stress** | 1.5× | +10% | $13,357,031 | +81.7% |
| **Heavy Stress** | 2.0× | +20% | $20,877,152 | +184.0% |

## Key Methodology

### Data Preprocessing
- **Missing Values**: Median imputation (numeric), mode imputation (categorical)
- **Feature Encoding**: OneHotEncoder for `person_home_ownership` and `loan_intent`
- **Custom Transformers**: Grade mapping (A→1, B→2, ..., G→7), binary encoding for defaults
- **Scaling**: StandardScaler for logistic regression (XGBoost handles raw features)

### Model Development
- **Train/Test Split**: 80/20 with stratification to maintain class balance
- **Baseline Model**: Logistic Regression (interpretable, regulatory-friendly)
- **Production Model**: XGBoost with optimized hyperparameters
  - `learning_rate`: 0.1
  - `max_depth`: 4
  - `n_estimators`: 300
  - `eval_metric`: 'logloss'

  ### Risk Calculation Framework
```python
Expected Loss = PD × EAD × LGD

Where:
- PD (Probability of Default): Model predicted probability
- EAD (Exposure at Default): Loan amount
- LGD (Loss Given Default): Grade-specific loss severity

LGD Mapping:
{
  'A': 0.25,  
  'B': 0.35,
  'C': 0.45,
  'D': 0.55,
  'E': 0.65,
  'F': 0.75,
  'G': 0.85   
}
```