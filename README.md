# Car Price Prediction (MSRP) using Machine Learning

An end-to-end Machine Learning pipeline to predict vehicle MSRP using technical specifications, feature selection, and ensemble regression.

---

## Project Overview
Develop a robust predictive engine for vehicle pricing based on engine specs, fuel economy, body style, and vehicle age. The project addresses data leakage, extreme outliers, right-skewed price distributions, and optimizes model generalization using bagging and boosting ensemble techniques.

---

## Data Preprocessing & Feature Engineering

### 1. Data Cleaning
- **Raw Data Profile:** Ingested 11,914 vehicle entries across 16 technical attributes[cite: 1].
- **Deduplication:** Identified and dropped 715 duplicate rows[cite: 1].
- **Missing Value Handling:** Dropped `market_category` due to excessive missingness (3,742 nulls) and high cardinality; resolved 108 missing entries across engine and fuel specifications[cite: 1].
- **Dimensionality Reduction:** Dropped high-cardinality `model` (900+ categories), `city_mpg` (collinear with `highway_mpg`), and `popularity` to minimize noise[cite: 1].
- **Outlier Filtering (Domain & IQR):** Pruned physical boundaries (`msrp` between $2,000–$200,000, `highway_mpg` $\le$ 60, `engine_hp` $\le$ 750) alongside 1.5 $\times$ IQR clipping, leaving **10,409 verified records** for modeling[cite: 1].

### 2. Feature Transformation
- **Engineered Attribute:** Derived `car_age` from vehicle model year (`2024 - year`) to capture depreciation[cite: 1].
- **Target Normalization:** Applied $\log(1+y)$ to normalize right-skewed pricing distribution[cite: 1].
- **Modular Pipeline:** Constructed a Scikit-Learn `ColumnTransformer` with `StandardScaler` for numeric features and `OneHotEncoder(handle_unknown='ignore')` for categoricals, yielding 80 feature dimensions with zero data leakage[cite: 1].

---

## Model Evaluation & Benchmarks
All models were benchmarked on an 80/20 train-test split, with predictions converted back to actual dollar ($) metrics:

| Model | MAE ($) | RMSE ($) | $R^2$ Score | Key Note |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | $9,303.23 | $13,530.35 | 0.6522 | Baseline linear benchmark |
| **Random Forest (Tuned)** | $2,637.08 | $5,159.32 | 0.9494 | Tuned bagging ensemble |
| **HistGradientBoosting (Tuned)** | $2,661.75 | $5,008.33 | 0.9523 | Tuned gradient boosting |
| **Ensemble (RF + HGB)** | **$2,582.85** | **$4,947.44** | **0.9535** | **Champion Model (Voting Regressor)** |

---

## Key Takeaways
1. **Ensemble Performance:** The Voting Regressor combined Random Forest and HistGradientBoosting to achieve peak generalization, reaching **$R^2 = 0.9535$** and reducing MAE to **$2,582.85$** (71.9% error reduction over baseline).
2. **Data Leak-Proofing:** Combining transformation steps within Scikit-Learn pipelines guaranteed unbiased evaluation between training and test sets.
3. **Artifact Serialization:** The final champion estimator and its preprocessing workflow were serialized into `car_price_ensemble_model.joblib` for production inference.

---

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib
