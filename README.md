# 🏠 Airbnb Melbourne Price Prediction

**2nd Place** — Kaggle-style ML competition | Applied Predictive Analytics, Macquarie University

A full end-to-end machine learning pipeline to predict Airbnb listing prices across Melbourne using property, host, location, review, and amenity data. The final submission used a stacking ensemble of seven models, achieving a Kaggle MAE of **109.815** on the final leaderboard.

---

## 📌 Project Overview

Airbnb pricing is notoriously difficult to predict due to the many interacting factors — location, room type, host quality, amenities, availability, and review history. This project tackled a high-dimensional, real-world dataset with significant messiness and required a complete pipeline from raw data cleaning through to competition-ready predictions.

**Competition:** Kaggle private leaderboard (3,000 test observations, 50/50 public/final split)  
**Evaluation metric:** Mean Absolute Error (MAE) on listing price (AUD)

---

## 🗂️ Repository Structure

```
├── Group-Project_final.ipynb   # Full pipeline notebook (cleaning → modelling → submission)
├── train.csv                   # Training dataset (not included — sourced from Kaggle)
├── test.csv                    # Test dataset (not included — sourced from Kaggle)
├── stacked_submission.csv      # Final Kaggle submission (stacking ensemble)
├── bad_kaggle_score.png        # Early leaderboard screenshot
├── latest_kaggle_score.png     # Mid-competition leaderboard
├── Kaggle_stacked.png          # Stacking model submission result
└── Kaggle_final.png            # Final leaderboard result (post-competition)
```

---

## ⚙️ Technical Stack

| Library | Purpose |
|---|---|
| Python 3 | Core language |
| Pandas & NumPy | Data cleaning, transformation, feature engineering |
| Scikit-learn | Preprocessing, cross-validation, PCA, ensemble methods |
| LightGBM | Primary gradient boosting model |
| XGBoost | Secondary gradient boosting model |
| CatBoost | Tertiary gradient boosting model |
| Optuna | Automated hyperparameter tuning |
| Matplotlib & Seaborn | EDA and visualisation |

---

## 🔧 Pipeline Summary

### 1. Data Cleaning
- Stripped `$` and `,` from the `price` column and converted to float
- Cleaned `host_response_rate` and `host_acceptance_rate` from percentage strings to proportions
- Parsed and cleaned the `bathrooms` column (mixed text/numeric format) into two structured features
- Imputed missing values across all features in both train and test sets using cross-mapping and frequency-based strategies

### 2. Feature Engineering
Key engineered features included:

- **`bathroom_count`** and **`bathroom_type_derived`** — extracted from the raw text `bathrooms` field
- **`verif_email`, `verif_phone`, `verif_work_email`, `verif_count`** — binary indicators and count from `host_verifications`
- **Missing value indicators** — binary flags (`_missing`) for all columns with significant nulls
- **Host activity features** — years active, years since first/last review
- **Amenity features** — total amenity count, binary dummies for common/luxury amenities
- **Location clusters** — K-Means clustering on latitude/longitude to capture neighbourhood-level pricing patterns
- **Interaction features** — `accommodates_per_bedroom`, `accommodates_x_num_amenities`, `bedrooms_x_bathrooms`
- **Property type consolidation** — rare categories grouped to reduce noise

Features that were trialled and removed (introduced noise without meaningful MAE improvement): sentiment scores from descriptions, distance from CBD, luxury amenity flags, description word count, high/low neighbourhood splits.

### 3. Model Comparison

Seven models were trained and evaluated with K-Fold cross-validation (log-transformed target, MAE on original scale):

| Model | CV MAE (AUD) |
|---|---|
| LightGBM | ~115.6 |
| CatBoost | ~116.1 |
| XGBoost | ~117.3 |
| Extra Trees | ~123.8 |
| Random Forest | ~129.9 |
| MLP + PCA | ~159.1 |
| Linear Regression | baseline |

### 4. Hyperparameter Tuning
All models were tuned using **Optuna** with K-Fold cross-validation (5, 10, and 15 splits compared). Search spaces were initially broad and iteratively narrowed based on best trial results. MAE on the original price scale (via `np.expm1`) was used as the optimisation objective throughout.

### 5. Final Model — Stacking Ensemble
The final submission used a `StackingRegressor` combining all seven base models with a **RidgeCV** meta-learner:

```python
StackingRegressor(
    estimators=[
        ('lgb', LGBMRegressor(...)),
        ('xgb', XGBRegressor(...)),
        ('cb', CatBoostRegressor(...)),
        ('mlp', Pipeline([PCA, MLPRegressor])),
        ('et', ExtraTreesRegressor(...)),
        ('rf', RandomForestRegressor(...)),
        ('lr', LinearRegression()),
    ],
    final_estimator=RidgeCV(),
    cv=5,
    passthrough=False
)
```

Log-transformed target (`np.log1p`) was used for training, with `np.expm1` applied to predictions before submission.

---

## 📊 Results

| Stage | Kaggle MAE | Leaderboard Position |
|---|---|---|
| Initial submission (no tuning) | ~350 | 16th |
| After location clusters + amenity features | ~270 | 1st (briefly) |
| Final (stacking + interaction features) | 109.815 | **2nd** |

---

## 🚀 Getting Started

1. Clone the repo and install dependencies:
```bash
pip install pandas numpy scikit-learn lightgbm xgboost catboost optuna matplotlib seaborn
```

2. Place `train.csv` and `test.csv` in the project root (sourced from the original Kaggle competition).

3. Run the notebook top to bottom:
```bash
jupyter notebook Group-Project_final.ipynb
```

Predictions will be saved as `stacked_submission.csv` in the project root.

---

## 💡 Key Takeaways

- **Feature engineering mattered more than model choice.** Location clusters, amenity counts, and interaction terms drove the biggest MAE improvements — more so than switching between gradient boosting variants.
- **Gradient boosting dominated.** LightGBM, XGBoost, and CatBoost all performed similarly and significantly outperformed the neural network and simpler tree models.
- **Iterative validation prevents leakage.** Using cross-validated MAE on the original price scale (not log scale) at every tuning step ensured leaderboard scores matched CV estimates closely.
- **Real-world data is messy.** The `bathrooms`, `amenities`, `host_verifications`, and price columns all required non-trivial cleaning before they could be used.

---

## 📄 License

This project was completed as part of the BUSA8030 Applied Predictive Analytics unit at Macquarie University. It is shared for portfolio and educational purposes.
