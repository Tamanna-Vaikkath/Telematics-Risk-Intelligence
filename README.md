# Telematics Risk Intelligence Platform

An end-to-end ML powered telematics insurance analytics platform built using Python, Plotly Dash, GLM, and GA2M/EBM models to simulate real-world commercial auto underwriting and pricing workflows.

The project combines synthetic telematics + claims data generation, actuarial pricing models, explainable AI, portfolio analytics, and interactive dashboards to help insurers make smarter underwriting, pricing, and risk management decisions.

---

# Key Features

- Synthetic commercial telematics insurance dataset generation (10,000 policies, 50+ features)
- Exposure-adjusted actuarial frequency modeling using Logistic GLM
- Severity modeling using GA2M / Explainable Boosting Machine (EBM)
- Pure premium + indicated premium pricing logic
- Risk scoring engine with underwriting recommendations
- Explainable AI risk decomposition
- Portfolio loss concentration and claims analytics
- Interactive Plotly Dash dashboard with 5 tabs
- Industry-style underwriting segmentation and pricing workflows

---

# Dashboard Modules

## 1. Executive Summary
- Portfolio KPIs
- Premium and loss trends
- Risk tier segmentation
- Portfolio concentration analysis

## 2. Risk Explorer
- Interactive underwriting segmentation
- Vehicle/driver risk analysis
- OERI vs Expected Loss analysis
- Underwriting assistant recommendations

## 3. Prediction Tool
- Real-time telematics risk scoring
- Claim probability prediction
- Pure premium and indicated premium estimation
- Underwriting decision support

## 4. Risk Decomposition
- Explainable AI feature contribution analysis
- Risk vs protective factor visualization
- Pricing decomposition and interpretability

## 5. Claims Analytics
- Claim severity analysis
- Catastrophic loss monitoring
- Pareto concentration analysis
- Calibration and lift analysis

---

# Tech Stack

- Python
- Plotly Dash
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- InterpretML (EBM / GA2M)
- Plotly
- YAML / JSON Configs

---

# Project Structure

Telematics-Risk-Intelligence/

├── app.py  
├── requirements.txt  
├── README.md  

├── config/  
│   ├── config.yaml  
│   ├── pricing.json  
│   └── feature_selection.json  

├── data/  
│   ├── telematics_raw.csv  
│   └── telematics_clean.csv  

├── models/  
│   ├── glm_frequency.pkl  
│   ├── glm_frequency_sm.pkl  
│   ├── ebm_severity.pkl  
│   ├── preprocessor_freq.pkl  
│   └── preprocessor_sev.pkl  

├── outputs/  
│   ├── model_predictions.csv  
│   ├── model_metrics.json  
│   ├── calibration_freq.csv  
│   └── lift_freq.csv  

├── src/  
│   ├── generate_data.py  
│   ├── clean_data.py  
│   ├── feature_engineering.py  
│   ├── feature_selection.py  
│   ├── train_model.py  
│   ├── risk_scoring.py  
│   └── utils.py  

---

# Modeling Workflow

Synthetic Data Generation  
↓  
EDA & Data Cleaning  
↓  
Feature Engineering  
↓  
Feature Selection  
↓  
Frequency Modeling (GLM)  
↓  
Severity Modeling (GA2M / EBM)  
↓  
Pure Premium Calculation  
↓  
Risk Scoring & Explainability  
↓  
Interactive Plotly Dash Dashboard  

---

# Installation

```bash
pip install -r requirements.txt
