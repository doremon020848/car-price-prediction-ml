# Car Price Prediction (MSRP) using Machine Learning

An end-to-end production-ready Machine Learning pipeline to predict manufacturer suggested retail prices (MSRP) of vehicles using specification analysis, domain feature interactions, and ensemble regression.

---

## Project Overview
The goal of this project is to build an accurate car valuation engine based on technical specifications, depreciation dynamics, and market positioning. The pipeline handles missing values, high cardinality, extreme physical outliers, right-skewed price distribution, and optimizes performance using tuned bagging and boosting tree ensembles.

---

## Data Preprocessing & Feature Engineering

### 1. Data Cleaning
- **Raw Data Profile:** Began with **11,914 records across 16 columns**.
- **Duplicates & Redundancy:** Removed 715 duplicate rows. Dropped `market_category` due to high missingness (3,742 missing entries) and high cardinality.
- **Missing Value Handling:** Pruned 108 missing entries across `engine_hp`, `engine_cylinders`, `number_of_doors`, and `engine_fuel_type`.
- **Feature Selection:** Dropped `model` (mitigating high cardinality of 900+ unique models), `city_mpg` (collinear with `highway_mpg`), and `popularity` to reduce noise.
- **Outlier Elimination (Domain & IQR):** Filtered domain boundaries (`msrp` between $2,000 and $200,000, `highway_mpg` $\le$ 60, `engine_hp` $\le$ 750) and applied 1.5 $\times$ IQR clipping on continuous attributes, retaining **10,409 clean training samples**.

### 2. Feature Engineering
- `car_age`: Derived vehicle age relative to the dataset baseline (`2024 - year`).
- `hp_per_cylinder`: Power-to-cylinder ratio (`engine_hp / engine_cylinders`) separating high-performance compact engines from commuter setups.
- `hp_age_ratio`: Interaction term (`engine_hp / (car_age + 1)`) capturing high-performance classic car value retention.
- `is_luxury`: Binary indicator flag identifying tier-1 luxury manufacturers (e.g., BMW, Mercedes-Benz, Porsche, Bentley, Ferrari).

### 3. Transformation Pipeline
- **Target Variable:** Applied Log Transformation ($\log(1+y)$) to stabilize right-skewed pricing distribution.
- **Feature Preprocessing (`ColumnTransformer`):**
  - **Skewed Numerics** (`car_age`, `hp_age_ratio`): `np.log1p` $\rightarrow$ `StandardScaler`.
  - **Standard Numerics** (`engine_hp`, `engine_cylinders`, `number_of_doors`, `highway_mpg`, `hp_per_cylinder`, `is_luxury`): `StandardScaler`.
  - **Categoricals** (`make`, `engine_fuel_type`, `transmission_type`, `driven_wheels`, `vehicle_size`, `vehicle_style`): `OneHotEncoder(handle_unknown='ignore')`.
  - Guarantees strict zero-data-leakage across 80/20 train-test validation.

---

## Model Evaluation & Benchmarks
All models were evaluated on the test set by inverting logarithmic predictions back to the actual dollar scale:

| Model | MAE ($) | RMSE ($) | $R^2$ Score | Key Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | $9,188.46 | $13,408.79 | 0.6584 | Underfits nonlinear market interactions |
| **Random Forest (Tuned)** | $2,640.37 | $5,151.27 | 0.9496 | High stability, reduced variance |
| **HistGradientBoosting (Tuned)** | $2,664.47 | $5,018.90 | 0.9521 | Fast bin-based gradient boosting |
| **Ensemble (RF + HGB)** | **$2,585.22** | **$4,943.71** | **0.9536** | **Champion Model (Voting Regressor)** |

---

## Key Takeaways
1. **Bagging + Boosting Synergy:** The Voting Regressor combined Random Forest and HistGradientBoosting to achieve peak generalization, reaching **$R^2 = 0.9536$** and driving MAE down to **$2,585.22** (a 71.86% error reduction over baseline).
2. **Domain Signal Advantage:** Incorporating engine output density (`hp_per_cylinder`) and luxury segmentation accounted for large price discrepancies that raw numeric features alone could not explain.
3. **Production Export:** Serialized the complete end-to-end preprocessing and ensemble estimator into `car_price_ensemble_model.joblib` for zero-leakage inference.

---

## Tech Stack
- **Language:** Python
- **Core Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Joblib
