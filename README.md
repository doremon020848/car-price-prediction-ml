# Car Price Prediction (MSRP) using Machine Learning

An end-to-end Machine Learning pipeline to predict vehicle MSRP using technical specifications, feature interactions, and ensemble methods.

---

## Project Overview
Develop a predictive valuation engine for vehicles based on technical specs, depreciation metrics, and brand tiers. The project covers data cleaning, handling skewed distributions, feature engineering, and ensembling tree-based algorithms.

---

## Feature Engineering & Preprocessing
- **Data Cleaning:** Removed duplicates, eliminated high-cardinality/redundant attributes (`model`, `city_mpg`, `popularity`), and pruned physical outliers via domain rules and IQR.
- **Engineered Features:**
  - `car_age`: Derived vehicle age from model year to model depreciation.
  - `hp_per_cylinder`: Power-to-cylinder ratio capturing engine efficiency.
  - `hp_age_ratio`: Interaction term capturing age-performance dynamics.
  - `is_luxury`: Binary flag for tier-1 premium/luxury automakers.
- **Transformation Pipeline:** Applied `np.log1p` to normalize right-skewed features (`car_age`, `hp_age_ratio`) and target pricing (`msrp`). Modularized using `ColumnTransformer` with `StandardScaler` and `OneHotEncoder` to guarantee zero data leakage.

---

## Model Benchmark & Results
Evaluated on an 80/20 train-test split measured on real dollar ($) scale:

| Model | MAE ($) | RMSE ($) | $R^2$ Score | Role |
| :--- | :--- | :--- | :--- | :--- |
| **Linear Regression (Baseline)** | $9,188.46 | $13,408.79 | 0.6584 | Baseline |
| **Random Forest (Tuned)** | $2,640.37 | $5,151.27 | 0.9496 | Tuned Bagging |
| **HistGradientBoosting (Tuned)** | $2,664.47 | $5,018.90 | 0.9521 | Tuned Boosting |
| **Ensemble (RF + HGB)** | **$2,585.22** | **$4,943.71** | **0.9536** | **Champion Model** |

---

## Key Takeaways
1. **Bagging + Boosting Synergy:** Ensembling Random Forest and HistGradientBoosting via `VotingRegressor` yielded the best generalization, reaching **$R^2 = 0.9536$** and reducing MAE to **$2,585.22**.
2. **Domain-Driven Features:** Engine efficiency (`hp_per_cylinder`) and luxury segmentation significantly boosted model capability over raw specs.
3. **Deployment Ready:** The end-to-end pipeline was serialized into a production artifact (`car_price_ensemble_model.joblib`) for web inference.

---

## Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn, Joblib
