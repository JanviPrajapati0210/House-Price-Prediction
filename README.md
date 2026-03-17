🏠 House Price Prediction
📌 Objective

- Build a machine learning model to predict house prices using structured housing data and compare multiple algorithms to find the best performer.

📊 Dataset

Dataset: train.csv (Ames Housing)
Contains features like property size, location, basement, garage, and overall quality
Target Variable: SalePrice

🛠️ Tech Stack

- Python, Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn, XGBoost

⚙️ Data Preprocessing

- Dropped columns with high missing values
- Filled missing values using:
- Median (numerical)
- Applied One-Hot Encoding
- Log transformation on SalePrice to reduce skewness

🤖 Models Used

1.Linear Regression
2.Random Forest Regressor
3.XGBoost Regressor

🎯 Evaluation Metrics

1.MAE
2.RMSE
3.R² Score
4.Evaluation was done on both:
       - Log-transformed values
       - Actual price values (USD)

🏆 Results

- Best Model: XGBoost Regressor
- Provided better accuracy and handled non-linear relationships effectively
- Tree-based models outperformed Linear Regression

📌 Key Insights

- Feature engineering and missing value handling improved performance
- Log transformation stabilized predictions
- Property quality and area are major price drivers

🚀 Future Improvements

- Hyperparameter tuning
- Advanced models (LightGBM, CatBoost)
- Model deployment using Flask/Streamlit

👩‍💻 Author
Janvi Prajapati