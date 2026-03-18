# 🏠 House Price Prediction

## 📌 Objective
Build a machine learning model to predict residential house prices and compare multiple algorithms to find the best performer.

---

## 📊 Dataset
- **Source**: Ames Housing Dataset
- **Train set**: 1,460 samples | 81 features
- **Target**: `SalePrice`

---

## 🛠️ Tech Stack
- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn, XGBoost

---

## ⚙️ Data Preprocessing
- Dropped columns with 90%+ missing values (PoolQC, Alley, Fence, MiscFeature)
- Filled missing numerical values with median, categorical with mode
- Label encoded all categorical columns
- Log transformation on `SalePrice` to reduce skewness
- Engineered 9 new features — `TotalSF`, `HouseAge`, `TotalBaths`, `QualityXArea` etc.

---

## 🤖 Models & Results

| Model | MAE | RMSE | R² |
|---|---|---|---|
| Linear Regression | $17,957 | $28,896 | 0.884 |
| Random Forest | $17,158 | $29,253 | 0.887 |
| **XGBoost** ✅ | **$15,363** | **$26,127** | **0.900** |

---

## 📌 Key Insights
- `OverallQual`, `GrLivArea` and `TotalSF` are the strongest price predictors
- Log transformation significantly stabilized model predictions
- Feature engineering improved R² by ~2% over baseline

---

## 🚀 Future Improvements
- Hyperparameter tuning with GridSearchCV
- Try LightGBM and CatBoost
- Deploy using Flask or Streamlit

---

## 👩‍💻 Author
**Janvi Prajapati** 