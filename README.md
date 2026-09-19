# Production Demand Prediction System

A Machine Learning-based demand forecasting system that predicts product demand using historical production, sales, inventory, pricing, promotion, holiday, and date-related features.

## 📌 Problem Statement

Accurate demand forecasting is important for efficient production planning. Overproduction can lead to excess inventory and storage costs, while underproduction can result in unmet customer demand and lost revenue. Manual demand estimation is also time-consuming and difficult to scale.

This project uses Machine Learning to predict product demand from historical business data and support better production planning.

## 🎯 Objective

The main objective is to build a regression-based Machine Learning system that can:

- Predict product demand
- Analyze factors affecting demand
- Reduce dependency on manual estimation
- Compare multiple regression algorithms
- Identify a suitable model based on evaluation metrics

## 📊 Dataset

The dataset contains **500 records** across **9 original columns**.

### Original Features

| Feature | Description |
|---|---|
| Date | Date of the record |
| Product_ID | Unique product identifier |
| Production | Quantity produced |
| Sales | Quantity sold |
| Inventory | Available inventory |
| Price | Product price |
| Promotion | Whether promotion was active (0/1) |
| Holiday | Whether the day was a holiday (0/1) |
| Demand | Target variable |

The dataset contains **5 unique products** and no missing values.

## ⚙️ Data Preprocessing

The following preprocessing steps were performed:

1. Loaded the dataset using Pandas.
2. Converted the `Date` column into datetime format.
3. Extracted time-based features:
   - Year
   - Month
   - Day
   - DayOfWeek
4. Checked the dataset for missing values.
5. Selected relevant features for Machine Learning.

## 🧩 Features Used for Modeling

The final model uses 10 input features:

- Production
- Sales
- Inventory
- Price
- Promotion
- Holiday
- Year
- Month
- Day
- DayOfWeek

### Target Variable

`Demand`

## 🔀 Train-Test Split

The dataset was divided using an **80:20 train-test split**.

- Training records: **400**
- Testing records: **100**

The model learns patterns from the training set and is evaluated on previously unseen test data.

## 🤖 Machine Learning Models

### 1. Linear Regression

Linear Regression was used as the baseline regression model.

The model was trained using the 400 training records and evaluated on the 100 test records.

### 2. Random Forest Regression

Random Forest Regressor was also trained to compare its performance with the Linear Regression baseline.

## 📈 Model Evaluation

The models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

### Results

| Model | MAE | RMSE | R² Score |
|---|---:|---:|---:|
| Linear Regression | 22.24 | 28.33 | 0.9455 |
| Random Forest | 26.82 | 33.91 | 0.9219 |

Based on the current test-set results, Linear Regression produced lower MAE and RMSE and a higher R² score than the Random Forest model.

### Linear Regression Performance

- **MAE:** 22.24
- **RMSE:** 28.33
- **R²:** 0.9455

The R² score indicates that the model explains approximately **94.5% of the variance in demand on the test set**.

## 📊 Actual vs Predicted Demand

The project also visualizes actual demand against predicted demand to understand how closely the model follows the observed demand patterns.

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- JupyterLab

## 📁 Project Structure

```text
Production-Demand-Prediction/
│
├── ML Project.ipynb
├── production_demand_dataset.csv
└── README.md
