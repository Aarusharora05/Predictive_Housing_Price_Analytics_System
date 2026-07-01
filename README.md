# 🏠 Predictive Housing Price Analytics System

An end-to-end Machine Learning project that predicts residential house prices using the Ames Housing Dataset. The project includes data preprocessing, feature engineering, model comparison, and deployment through an interactive Streamlit web application.

---

## 🌐 Live Demo

**Streamlit App:**  
https://predictive-housing-price-analytics.streamlit.app/

**GitHub Repository:**  
https://github.com/Aarusharora05/Predictive_Housing_Price_Analytics_System

---

# 📌 Project Overview

The objective of this project is to accurately predict house prices based on various property characteristics using Machine Learning.

The project follows the complete Machine Learning workflow:

- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Data Preprocessing
- Model Building
- Model Evaluation
- Model Deployment

The final model is deployed using Streamlit, allowing users to modify important house features and receive real-time price predictions.

---

# 📂 Dataset

**Dataset:** Ames Housing Dataset

The dataset contains information about residential houses including:

- Lot Area
- Neighborhood
- Overall Quality
- Garage Capacity
- Basement Area
- Living Area
- Year Built
- Bathrooms
- Sale Price
- and many more...

Dataset Size:

- **1460 Rows**
- **80 Original Features**
- **5 Engineered Features**

---

# ⚙️ Feature Engineering

Additional features were created to improve prediction performance:

- House Age
- Remodel Age
- Total Bathrooms
- Total Porch Area
- Total Living Area

These engineered features significantly improved the predictive capability of the models.

---

# 🤖 Machine Learning Models Evaluated

The following regression models were trained and compared:

- Linear Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Random Forest Regressor
- Gradient Boosting Regressor

The best-performing model was selected based on R² Score and Cross Validation performance.

---

# 📊 Model Performance

| Metric | Value |
|---------|-------|
| Best Model | Gradient Boosting Regressor |
| R² Score | **0.9073** |
| RMSE | **26659.40** |
| MAE | **15859.79** |
| 10-Fold Cross Validation | **0.9025** |

The model demonstrates strong predictive performance and generalizes well on unseen data.

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

# 📁 Project Structure

```
Predictive_Housing_Price_Analytics_System
│
├── app
│   └── app.py
│
├── data
│   └── train.csv
│
├── images
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   └── residual_plot.png
│
├── models
│   ├── best_model.pkl
│   ├── feature_columns.pkl
│   ├── feature_importance.csv
│   └── model_comparison.csv
│
├── notebooks
│   └── 01_EDA.ipynb
│
├── train_model.py
├── requirements.txt
└── README.md
```

---

# 🚀 Running the Project Locally

Clone the repository:

```bash
git clone https://github.com/Aarusharora05/Predictive_Housing_Price_Analytics_System.git
```

Move into the project folder:

```bash
cd Predictive_Housing_Price_Analytics_System
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app/app.py
```

---

# 📈 Project Workflow

1. Data Cleaning
2. Missing Value Handling
3. Feature Engineering
4. Exploratory Data Analysis
5. Feature Encoding
6. Data Preprocessing Pipeline
7. Model Training
8. Model Comparison
9. Model Evaluation
10. Streamlit Deployment

---

# 🎯 Key Features

- End-to-End Machine Learning Workflow
- Feature Engineering
- Data Preprocessing Pipeline
- Multiple Model Comparison
- Cross Validation
- Model Persistence using Joblib
- Interactive Streamlit Dashboard
- Real-Time House Price Prediction

---

# 📌 Future Improvements

- Hyperparameter Optimization
- XGBoost Model
- Model Explainability using SHAP
- User Authentication
- Historical Price Trends
- Cloud Database Integration

---

# 👨‍💻 Author

**Aarush Arora**

B.Tech Computer Science (Data Science)

Maharaja Agrasen Institute of Technology (MAIT)

GitHub:
https://github.com/Aarusharora05

---

## ⭐ If you found this project useful, consider giving it a star!
