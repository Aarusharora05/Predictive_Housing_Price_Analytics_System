# ==========================================
# Predictive Housing Price Analytics System
# Model Training Script
# ==========================================

import time

start = time.time()

import warnings
warnings.filterwarnings("ignore")

# Data manipulation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Train Test Split
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

# Preprocessing
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# Regression Models
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet
)

from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)

# Evaluation Metrics
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

# Save Model
import joblib

print("Libraries imported successfully.")

# ==========================================
# Load Dataset
# ==========================================

housing = pd.read_csv("data/train.csv")

print("Dataset Loaded Successfully")

print(f"Shape : {housing.shape}")

# ==========================================
# Drop Columns with Excessive Missing Values
# ==========================================

columns_to_drop = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence"
]

housing.drop(columns=columns_to_drop, inplace=True)

print("Dropped Columns:")
print(columns_to_drop)

# ==========================================
# Feature Engineering
# ==========================================

# Create a copy of the dataset
housing_fe = housing.copy()

# Age of the house at the time of sale
housing_fe["HouseAge"] = housing_fe["YrSold"] - housing_fe["YearBuilt"]

# Age since last remodel
housing_fe["RemodelAge"] = housing_fe["YrSold"] - housing_fe["YearRemodAdd"]

# Total Bathrooms
housing_fe["TotalBathrooms"] = (
    housing_fe["FullBath"] +
    (0.5 * housing_fe["HalfBath"]) +
    housing_fe["BsmtFullBath"] +
    (0.5 * housing_fe["BsmtHalfBath"])
)

# Total Porch Area
housing_fe["TotalPorchSF"] = (
    housing_fe["OpenPorchSF"] +
    housing_fe["EnclosedPorch"] +
    housing_fe["3SsnPorch"] +
    housing_fe["ScreenPorch"]
)

# Total Living Area
housing_fe["TotalLivingArea"] = (
    housing_fe["GrLivArea"] +
    housing_fe["TotalBsmtSF"]
)

print("Feature Engineering Completed Successfully.")

# ==========================================
# Separate Features and Target
# ==========================================

X = housing_fe.drop("SalePrice", axis=1)

y = housing_fe["SalePrice"]

print("Features Shape :", X.shape)

print("Target Shape :", y.shape)

# ==========================================
# Identify Numerical and Categorical Columns
# ==========================================

numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X.select_dtypes(
    include=["object"]
).columns

print(f"Numerical Features : {len(numerical_features)}")

print(f"Categorical Features : {len(categorical_features)}")

# ==========================================
# Numerical Pipeline
# ==========================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

# ==========================================
# Categorical Pipeline
# ==========================================

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant",
                                  fill_value="Missing")),

        ("encoder",
         OneHotEncoder(handle_unknown="ignore"))
    ]
)

# ==========================================
# Combine Preprocessing Steps
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

print("Preprocessing Pipeline Created Successfully.")

# ==========================================
# Train-Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Samples :", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

# ==========================================
# Regression Models
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Lasso Regression": Lasso(alpha=0.0005, max_iter=10000),

    "ElasticNet": ElasticNet(
    alpha=0.0005,
    l1_ratio=0.3,
    max_iter=10000
),

    "Random Forest": RandomForestRegressor(
    n_estimators=500,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
    ,

    "Gradient Boosting": GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42
)
}

# ==========================================
# Model Evaluation
# ==========================================

results = []

best_model = None
best_pipeline = None
best_score = -999

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Train model
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, y_pred)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    mae = mean_absolute_error(y_test, y_pred)

    # Cross Validation
    cv_score = cross_val_score(
        pipeline,
        X,
        y,
        cv=10,
        scoring="r2"
    ).mean()

    results.append({

    "Model": model_name,

    "R2 Score": round(r2,4),

    "RMSE": round(rmse,2),

    "MAE": round(mae,2),

    "Cross Validation": round(cv_score,4)

})

    if r2 > best_score:
        best_score = r2
        best_model = model_name
        best_pipeline = pipeline

print("Model Evaluation Completed Successfully.")

# ==========================================
# Model Comparison
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="R2 Score",
    ascending=False
)

print()

print(results_df.to_string(index=False))

print()

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

print("Model comparison saved.")

print("=" * 60)

print("Best Model :", best_model)

print("Best R2 :", round(best_score,4))

print("=" * 60)

joblib.dump(
    best_pipeline,
    "models/best_model.pkl"
)

feature_columns = X.columns.tolist()

joblib.dump(
    feature_columns,
    "models/feature_columns.pkl"
)

print("Best model saved successfully.")

if best_model in ["Random Forest", "Gradient Boosting"]:

    trained_model = best_pipeline.named_steps["model"]

    feature_names = best_pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()

    importance = trained_model.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )
    
    importance_df.to_csv(
    "models/feature_importance.csv",
    index=False
)

    print("\nTop 20 Important Features\n")

    print(importance_df.head(20))

    plt.figure(figsize=(10,8))

    sns.barplot(
        data=importance_df.head(20),
        x="Importance",
        y="Feature"
    )

    plt.title("Top 20 Feature Importances")
    
    plt.savefig(
    "images/feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

    plt.show()
    
    y_pred = best_pipeline.predict(X_test)

plt.figure(figsize=(8,8))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linewidth=2
)

plt.xlabel("Actual Price")

plt.ylabel("Predicted Price")

plt.title("Actual vs Predicted")

plt.savefig(
    "images/actual_vs_predicted.png",
    dpi=300,
    bbox_inches="tight")

plt.show()


residuals = y_test - y_pred

plt.figure(figsize=(10,6))

sns.scatterplot(
    x=y_pred,
    y=residuals
)

plt.axhline(
    0,
    color="red",
    linestyle="--"
)

plt.xlabel("Predicted Price")

plt.ylabel("Residual")

plt.title("Residual Plot")

plt.savefig(
    "images/residual_plot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\n")
print("=" * 70)
print("PROJECT TRAINING COMPLETED")
print("=" * 70)

best_row = results_df.iloc[0]

print(f"Best Model      : {best_row['Model']}")
print(f"R2 Score        : {best_row['R2 Score']}")
print(f"RMSE            : {best_row['RMSE']}")
print(f"MAE             : {best_row['MAE']}")
print(f"Cross Validation: {best_row['Cross Validation']}")
print(f"Best R2 Score         : {best_score:.4f}")
print(f"Model Saved           : models/best_model.pkl")
print(f"Comparison Table      : models/model_comparison.csv")

print("=" * 70)

end = time.time()

print(f"\nExecution Time : {end-start:.2f} seconds")
