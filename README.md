# Credit Risk Analysis and Loan Default Prediction 

## Project Overview 
Advanced credit risk prediction framework implementing PD (Probability of Default), LGD (Loss Given Default), and EAD (Exposure at Default) modeling with comprehensive stress testing scenarios.

## Key Features 
- **Predictive Modeling**: XGBoost classifier achieving 0.95 ROC-AUC score
- **Risk Metrics**: Complete PD, LGD, EAD framework
- **Stress Testing**: Multiple scenario analysis (Baseline, Mild, Moderate, Severe)
- **Visualizations**: Grade-level risk distribution and stress impact analysis

## Technical Stack

### Core Technologies
- **Machine Learning**: scikit-learn, XGBoost 
- **Data Processing**: pandas, numpy
- **Visualization**: matplotlib, seaborn
- **Environment**: Python 3.9+

### ML Pipeline Components
```python
Pipeline Architecture:  
├── Feature Engineering
│   ├── OrdinalEncoder (loan_grade: A < B < C < D < E < F < G)
│   ├── BinaryEncoder (cb_person_default_on_file: N/Y)
│   ├── SimpleImputer (median for numeric, most_frequent for categorical)
│   └── OneHotEncoder (person_home_ownership, loan_intent)
├── Preprocessing
│   ├── StandardScaler (for Logistic Regression only)
│   └── ColumnTransformer
└── Models
    ├── Logistic Regression (baseline)
    └── XGBoost (production model)
```

## Model Performance
| Metric | Logistic Regression | XGBoost | Improvement |
|--------|---------------------|---------|-------------|
| **ROC-AUC** | 0.87 | 0.95 | +9.2% |
| **Precision** | 0.75 | 0.97 | +29.3% |
| **Recall** | 0.52 | 0.72 | +38.5% |
| **F1 Score** | 0.61 | 0.83 | +36.1% |

### Performance Highlights
- **95% ROC-AUC**: Excellent discrimination between defaulters and non-defaulters
- **97% Precision**: Only 3% false positive rate
- **72% Recall**: Captures 72% of actual defaults
- **83% F1 Score**: Optimal balance between precision and recall

## Portfolio Analysis

### Key Metrics
- **Portfolio Default Rate (PD)**: 21.63%
- **Average Exposure (EAD)**: $9,670.39
- **Total Expected Loss**: $6,886,979.34
- **Average Expected Loss per Loan**: $1,056.77
- **Expected Loss Rate**: 10.93%

### Loss Distribution by Loan Grade
| Grade | Loss Contribution | 
|-------|-------------------|
| **D** | 40.4% | 
| **B** | 19.0% |  
| **C** | 16.6% |  
| **E** | 14.0% |  
| **F** | 5.4% |  
| **G** | 4.7% | 
| **A** | 0.0% |  

### LGD (Loss Given Default) by Grade
| Grade | LGD | 
|-------|-----|
| A | 0% |
| B | 35% |
| C | 45% |
| D | 55% |
| E | 65% |
| F | 75% |
| G | 85% |

### Stress Testing Results 
| Scenario | PD Multiplier | LGD Adjustment | Total Expected Loss | % Increase from Baseline |
|----------|---------------|----------------|---------------------|--------------------------|
| **Baseline** | 1.0× | +0% | $6,886,979 | - |
| **Mild Stress** | 1.2× | +5% | $9,195,083 | +33.5% |
| **Moderate Stress** | 1.5× | +10% | $12,657,239 | +83.8% |
| **Severe Stress** | 2.0× | +20% | $19,940,955 | +189.5% |

## Key Methodology

### Data Preprocessing
- **Missing Values**: Median imputation (numeric), most frequent imputation (categorical)
- **Feature Encoding**: 
  - OrdinalEncoder for `loan_grade` (A=0, B=1, ..., G=6) preserving natural ordering
  - BinaryEncoder for `cb_person_default_on_file` (N=0, Y=1)
  - OneHotEncoder with drop='first' for `person_home_ownership` and `loan_intent`
- **Scaling**: StandardScaler applied only in Logistic Regression pipeline (XGBoost handles raw features)

### Model Development
- **Train/Test Split**: 80/20 stratified sampling
- **Baseline Model**: Logistic Regression with scaling
- **Production Model**: XGBoost (learning_rate=0.1, max_depth=4, n_estimators=300)

### Risk Calculation Framework
```
Expected Loss = PD × EAD × LGD
```

Where:
- **PD (Probability of Default)**: Model predicted probability
- **EAD (Exposure at Default)**: Loan amount
- **LGD (Loss Given Default)**: Grade-specific loss severity (0% for A to 85% for G)

## Dataset Information
- **Total Records**: 32,581 loan applications
- **Features**: 12 columns including borrower demographics, loan characteristics, and credit history
- **Target Variable**: `loan_status` (0 = paid, 1 = default)
- **Default Rate**: 21.8%

### Feature Categories
1. **Ordinal**: loan_grade
2. **Binary**: cb_person_default_on_file  
3. **Nominal**: person_home_ownership, loan_intent
4. **Numeric**: person_age, person_income, person_emp_length, loan_amnt, loan_int_rate, loan_percent_income, cb_person_cred_hist_length

## Key Findings

### Model Performance
- **XGBoost** demonstrates superior performance across all metrics
- **8.3% higher ROC-AUC** compared to Logistic Regression
- Captures **20% more defaults** while maintaining high precision
- **22.5% fewer false alarms** reduces unnecessary loan rejections

### Portfolio Risk Profile
- **Grade D loans** contribute the most to expected losses (40.4%) despite not being the largest segment
- **Lower-grade loans** (E, F, G) show significantly higher default probabilities and loss severities
- **Grade A loans** have 0% LGD, indicating high recovery rates

### Stress Testing Insights
- Under **moderate stress** (+50% PD, +10% LGD), expected losses nearly **double** (+83.8%)
- **Severe stress** scenarios could increase losses by nearly **190%**
- Lower-grade portfolios show higher sensitivity to economic stress

## Business Recommendations
1. **Deploy XGBoost model** for real-time credit decisions to maximize risk detection
2. **Review pricing strategy** for grades D-G to ensure adequate risk premium coverage
3. **Monitor economic indicators** for early warning signs of deteriorating credit conditions
4. **Consider additional collateral requirements** or stricter underwriting for Grade F-G applicants

## Contact & Links

**Author**: Nitin Vinayak  
**Email**: nitinvinayak.m@gmail.com  
**LinkedIn**: [linkedin.com/in/nitin-vinayak](https://linkedin.com/in/nitin-vinayak)
