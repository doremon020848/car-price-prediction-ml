# 🚗 Car Price Prediction (MSRP) using Machine Learning

An end-to-end Machine Learning project to predict Manufacturer's Suggested Retail Price (MSRP) using vehicle specifications and performance metrics.

---

## 📌 Project Overview
The goal of this project is to develop a predictive regression model for vehicle pricing based on technical attributes (e.g., horsepower, fuel economy, vehicle age). The project covers full data cleaning, exploratory data analysis (EDA), handling skewed distributions, feature engineering, and evaluating multiple ML algorithms.

---

## 🛠️ Data Preprocessing & Feature Engineering
- **Missing Values & Duplicates:** Dropped duplicate records and removed columns with high missing rates (`market_category`).
- **Feature Selection:** 
  - Dropped `model` to avoid high cardinality issues (>900 unique values).
  - Dropped `city_mpg` to resolve multicollinearity with `highway_mpg`.
  - Dropped `popularity` due to low predictive power.
- **Feature Engineering:** Derived `car_age` from production year to capture vehicle depreciation.
- **Outlier Removal:** Filtered domain-specific anomalies (e.g., MSRP < $2,000, MPG > 60) and clipped extreme performance values via IQR.
- **Data Transformation:**
  - Applied **Log Transformation (`np.log1p`)** on the right-skewed target variable (`msrp`).
  - Transformed skewed numeric feature (`car_age`) before scaling.
  - Implemented standard preprocessing pipeline using `ColumnTransformer` (`StandardScaler` for numeric, `OneHotEncoder` for categorical).

---

## 📊 Model Evaluation & Results
Evaluated on an 80/20 train-test split using real dollar ($) scale metrics:

| Model | MAE ($) | RMSE ($) | $R^2$ Score | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression** | $9,210.27 | $13,450.60 | 0.6563 | Baseline |
| **HistGradientBoosting** | $3,017.65 | $5,872.81 | 0.9345 | High-efficiency Model |
| **Random Forest** | **$2,709.98** | **$5,220.57** | **0.9482** | **Best Performing Model** |

---

## 💡 Key Insights
1. **Non-linear Dynamics:** Vehicle pricing exhibits strong non-linear interactions (e.g., brand tier interacting with horsepower), allowing Tree Ensemble models to drastically outperform Linear Regression.
2. **Impact of Log Transformation:** Taming the heavy right-skewed price distribution stabilized gradient updates and prevented high-end luxury outliers from skewing model weights, resulting in an MAE under $3,000 across price tiers.
3. **Pipeline Modularity:** Bundling transformers and estimators into Scikit-Learn `Pipeline` completely mitigated data leakage and produced an artifact ready for production deployment.

---

## 💻 Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib
