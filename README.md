
# Production Demand Prediction System

A machine learning project that estimates product demand using production, sales, inventory, price, promotion, holiday, and date-related features.

## Project Overview

The system uses historical records to train regression models and provides an interactive Streamlit dashboard for demand prediction and data analysis.

## Features

- Dataset exploration and preprocessing
- Date-based feature engineering
- Linear Regression, Random Forest, and Decision Tree models
- Model evaluation using MAE, RMSE, and R²
- Saved model using Joblib
- Interactive demand prediction
- Product-wise and monthly demand visualizations
- Historical data filtering and CSV download

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Joblib
- Streamlit
- Jupyter Notebook

## Project Files

- `ML Project.ipynb` — model development and experimentation
- `app.py` — Streamlit dashboard
- `production_demand_dataset.csv` — practice dataset
- `best_demand_model.pkl` — saved trained model
- `model_features.pkl` — model input feature names
- `requirements.txt` — Python dependencies

## Setup Instructions

### 1. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 2. Run the application

```bash
python -m streamlit run app.py
```

### 3. Open the dashboard

Visit `http://localhost:8501` in your browser.

## Model Evaluation

The models are evaluated using:

- **MAE:** Mean Absolute Error
- **RMSE:** Root Mean Squared Error
- **R²:** Coefficient of Determination

The dashboard displays evaluation metrics for the saved model.

## Dataset Disclaimer

This project uses synthetic practice data for educational purposes. Its predictions have not been validated against real-world demand data and should not be used for actual production decisions.

## Future Improvements

- Evaluate the model on real-world demand data
- Improve forecasting features and prevent target leakage
- Compare additional regression models
- Add model monitoring and deployment
