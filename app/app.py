import streamlit as st
import pandas as pd
import joblib
import os

# -----------------------------------
# Page Configuration
# -----------------------------------

st.set_page_config(
    page_title="Predictive Housing Price Analytics System",
    page_icon="🏠",
    layout="wide"
)

st.sidebar.title("🏠 About Project")

st.sidebar.info(
"""
### Model

Gradient Boosting Regressor

### Performance

R² Score : 0.9073

Cross Validation : 0.9025

### Dataset

• 1460 Houses

• 80 Original Features

• 5 Engineered Features

### Tech Stack

Python

Scikit-learn

Pandas

Streamlit
"""
)

st.sidebar.markdown("---")

st.sidebar.write(
    "Built using Python, Scikit-learn and Streamlit."
)

st.title("🏡 Predictive Housing Price Analytics System")

st.caption("Machine Learning Powered Residential Property Valuation")

st.info(
"""
Select a neighborhood and choose a similar property.

Modify its characteristics to estimate the updated market value using Machine Learning.
"""
)

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

DATA_PATH = os.path.join(BASE_DIR,"data","train.csv")

MODEL_PATH = os.path.join(BASE_DIR,"models","best_model.pkl")

housing = pd.read_csv(DATA_PATH)

model = joblib.load(MODEL_PATH)

st.header("🏘️ Select a Property")

# -----------------------------------
# Select Neighborhood
# -----------------------------------

neighborhood = st.selectbox(
    "Select Neighborhood",
    sorted(housing["Neighborhood"].unique())
)

filtered_houses = housing[
    housing["Neighborhood"] == neighborhood
].copy()

filtered_houses["Display"] = (

    filtered_houses["Neighborhood"]

    + " | "

    + filtered_houses["GrLivArea"].astype(str)

    + " sq.ft"

    + " | Built "

    + filtered_houses["YearBuilt"].astype(str)

)

selected_display = st.selectbox(

    "Available Properties",

    filtered_houses["Display"]

)

selected_house = filtered_houses[
    filtered_houses["Display"] == selected_display
].copy()

house = selected_house.iloc[0].copy()

st.subheader("Selected Property")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Neighborhood",
        house["Neighborhood"]
    )

with c2:
    st.metric(
        "Living Area",
        f"{house['GrLivArea']} sq.ft."
    )

with c3:
    st.metric(
        "Year Built",
        int(house["YearBuilt"])
    )

with c4:
    st.metric(
        "Original Price",
        f"${house['SalePrice']:,.0f}"
    )

st.markdown("---")

st.subheader("Modify Property Features")

col1,col2 = st.columns(2)

with col1:

    overall_quality = st.slider(
        "Overall Quality",
        1,
        10,
        int(house["OverallQual"])
    )

    living_area = st.number_input(
        "Ground Living Area",
        value=int(house["GrLivArea"])
    )

    garage_cars = st.slider(
        "Garage Cars",
        0,
        5,
        int(house["GarageCars"])
    )

    garage_area = st.number_input(
        "Garage Area",
        value=int(house["GarageArea"])
    )

with col2:

    basement = st.number_input(
        "Basement Area",
        value=int(house["TotalBsmtSF"])
    )

    year_built = st.slider(
        "Year Built",
        1870,
        2010,
        int(house["YearBuilt"])
    )

    full_bath = st.slider(
        "Full Bathrooms",
        0,
        5,
        int(house["FullBath"])
    )
    
    # -----------------------------------
# Update Selected House
# -----------------------------------

house["OverallQual"] = overall_quality
house["GrLivArea"] = living_area
house["GarageCars"] = garage_cars
house["GarageArea"] = garage_area
house["TotalBsmtSF"] = basement
house["YearBuilt"] = year_built
house["FullBath"] = full_bath

# -----------------------------------
# Feature Engineering
# -----------------------------------

house["HouseAge"] = house["YrSold"] - house["YearBuilt"]

house["RemodelAge"] = house["YrSold"] - house["YearRemodAdd"]

house["TotalBathrooms"] = (
    house["FullBath"]
    + (0.5 * house["HalfBath"])
    + house["BsmtFullBath"]
    + (0.5 * house["BsmtHalfBath"])
)

house["TotalPorchSF"] = (
    house["OpenPorchSF"]
    + house["EnclosedPorch"]
    + house["3SsnPorch"]
    + house["ScreenPorch"]
)

house["TotalLivingArea"] = (
    house["GrLivArea"]
    + house["TotalBsmtSF"]
)

st.markdown("---")

predict = st.button(
    "🏠 Estimate Property Value",
    use_container_width=True,
    type="primary"
)
if predict:

    prediction = model.predict(
        pd.DataFrame([house])
    )[0]

    actual_price = house["SalePrice"]
    
    difference = prediction - actual_price
    
    st.markdown("---")

    st.subheader("Prediction Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Predicted Price",
            f"${prediction:,.0f}"
        )

    with col2:
        st.metric(
            "Original Price",
            f"${actual_price:,.0f}"
        )

    with col3:
        st.metric(
            "Difference",
            f"${difference:,.0f}",
            delta=f"{difference/actual_price:.1%}"
        )

    if difference > 0:
        st.success(
            f"Estimated increase of ${difference:,.0f} compared to the original sale price."
        )
    elif difference < 0:
        st.warning(
            f"Estimated decrease of ${abs(difference):,.0f} compared to the original sale price."
        )
    else:
        st.info("Predicted price matches the original sale price.")
    
    st.markdown("---")

    st.subheader("Prediction Inputs")

    summary = pd.DataFrame({

        "Feature":[
            "Overall Quality",
            "Living Area",
            "Garage Cars",
            "Garage Area",
            "Basement Area",
            "Year Built",
            "Bathrooms"
        ],

        "Value":[
            overall_quality,
            living_area,
            garage_cars,
            garage_area,
            basement,
            year_built,
            full_bath
        ]

    })

    st.table(summary)
    
    st.markdown("---")

st.subheader("Top 10 Most Important Features")

importance = pd.read_csv(
    os.path.join(
        BASE_DIR,
        "models",
        "feature_importance.csv"
    )
)

st.bar_chart(
    importance.head(10).set_index("Feature")
)

st.markdown("---")

st.subheader("Model Evaluation")

img1, img2 = st.columns(2)

with img1:
    st.image(
        os.path.join(
            BASE_DIR,
            "images",
            "actual_vs_predicted.png"
        ),
        caption="Actual vs Predicted"
    )

with img2:
    st.image(
        os.path.join(
            BASE_DIR,
            "images",
            "residual_plot.png"
        ),
        caption="Residual Plot"
    )
    
    st.image(
    os.path.join(
        BASE_DIR,
        "images",
        "feature_importance.png"
    ),
    caption="Feature Importance"
)
    
st.markdown("---")

st.caption(
    "Developed using Python, Scikit-learn and Streamlit."
)   