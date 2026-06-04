import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression


# PAGE TITLE


st.set_page_config(
    page_title="House Price Prediction System",
    page_icon="🏠",
    layout="wide"
)

col1, col2 = st.columns([2,1])

with col1:
    st.markdown("""
    <h2 style='color:white;'>🏠 House Price Prediction System</h1>
    <h4 style='color:#94A3B8;'>
    Predict Real Estate Prices Using Machine Learning
    </h4>
    """, unsafe_allow_html=True)

    st.markdown("""
    <style>
    .metric-card{
        background-color:#111827;
        padding:15px;
        border-radius:12px;
        border:1px solid #374151;
        text-align:center;
    }
    </style>
    """, unsafe_allow_html=True)

# LOAD DATASET


housing = pd.read_csv("housing.csv")


# CREATE INCOME CATEGORY


housing['income_cat'] = pd.cut(
    housing['median_income'],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)


# STRATIFIED SPLIT

split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)

housing = strat_train_set.copy()

# FEATURES AND LABELS

housing_labels = housing["median_house_value"].copy()

housing = housing.drop("median_house_value", axis=1)


# PIPELINES


num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()

cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])

# PREPARE DATA

housing_prepared = full_pipeline.fit_transform(housing)

# TRAIN MODEL

model = LinearRegression()

model.fit(housing_prepared, housing_labels)


# USER INPUTS


st.sidebar.title("🏠 Enter House Details")

longitude = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=-122.23,
    step=0.01
)

latitude = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=37.88,
    step=0.01
)

house_age = st.sidebar.number_input(
    "House Age",
    value=41
)

total_rooms = st.sidebar.number_input(
    "Total Rooms",
    value=880
)

total_bedrooms = st.sidebar.number_input(
    "Total Bedrooms",
    value=129
)

population = st.sidebar.number_input(
    "Population",
    value=322
)

households = st.sidebar.number_input(
    "Households",
    value=126
)

median_income = st.sidebar.number_input(
    "Median Income",
    value=8.3252,
    format="%.4f"
)
ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ['NEAR BAY', 'INLAND', 'NEAR OCEAN', 'ISLAND', '<1H OCEAN']
)


# PREDICTION BUTTON


if st.button("Predict House Price"):

    sample_data = pd.DataFrame({
        'longitude': [longitude],
        'latitude': [latitude],
        'housing_median_age': [house_age],
        'total_rooms': [total_rooms],
        'total_bedrooms': [total_bedrooms],
        'population': [population],
        'households': [households],
        'median_income': [median_income],
        'ocean_proximity': [ocean_proximity]
    })

    sample_prepared = full_pipeline.transform(sample_data)

    prediction = model.predict(sample_prepared)

    usd_price = prediction[0]
    inr_price = usd_price * 83
    crore_price = inr_price / 10000000


    # FUTURE PRICE ESTIMATION

    growth_rate = 0.08

    future_price_usd = usd_price * (1 + growth_rate)
    future_price_inr = future_price_usd * 83
    future_price_crore = future_price_inr / 10000000

    st.subheader("📊 Current vs Future Price Comparison")

    current_col, future_col = st.columns(2)

    with current_col:

        st.markdown("## 🏠 Current Price")

        st.metric(
            "USD Price",
            f"${usd_price:,.2f}"
        )

        st.metric(
            "INR Price",
            f"₹ {inr_price:,.0f}"
        )

        st.metric(
            "Price in Crores",
            f"₹ {crore_price:.2f} Cr"
        )

    with future_col:

        st.markdown("## 🚀 Future Price (After 1 Year)")

        st.metric(
            "USD Price",
            f"${future_price_usd:,.2f}",
            "+8%"
        )

        st.metric(
            "INR Price",
            f"₹ {future_price_inr:,.0f}"
        )

        st.metric(
            "Price in Crores",
            f"₹ {future_price_crore:.2f} Cr"
        )

    growth_amount = future_price_inr - inr_price

    st.success(
        f"📈 Expected Appreciation After 1 Year: ₹ {growth_amount:,.0f}"
    )
    st.success("✅ Prediction Generated Successfully")

    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown(
            f"""
            <div class="metric-card">
            <h4>🌊 Ocean</h4>
            <h3>{ocean_proximity}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c2:
        st.markdown(
            f"""
            <div class="metric-card">
            <h4>📍 Longitude</h4>
            <h3>{longitude}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c3:
        st.markdown(
            f"""
            <div class="metric-card">
            <h4>📍 Latitude</h4>
            <h3>{latitude}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c4:
        st.markdown(
            f"""
            <div class="metric-card">
            <h4>💰 Income</h4>
            <h3>{median_income}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )

    with c5:
        st.markdown(
            f"""
            <div class="metric-card">
            <h4>🏠 Rooms</h4>
            <h3>{total_rooms}</h3>
            </div>
            """,
            unsafe_allow_html=True
        )