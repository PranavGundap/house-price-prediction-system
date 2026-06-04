import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import cross_val_score

# 1. LOAD DATASET


housing = pd.read_csv("housing.csv")


# 2. CREATE INCOME CATEGORY FOR STRATIFIED SAMPLING


housing['income_cat'] = pd.cut(
    housing['median_income'],
    bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
    labels=[1, 2, 3, 4, 5]
)

# 3. SPLIT TRAIN AND TEST DATA


split = StratifiedShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

for train_index, test_index in split.split(housing, housing['income_cat']):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)


# 4. WORK ON TRAINING DATA


housing = strat_train_set.copy()


# 5. SEPARATE FEATURES AND LABELS


housing_labels = housing["median_house_value"].copy()

housing = housing.drop("median_house_value", axis=1)


# 6. NUMERICAL AND CATEGORICAL COLUMNS


num_attribs = housing.drop("ocean_proximity", axis=1).columns.tolist()

cat_attribs = ["ocean_proximity"]


# 7. NUMERICAL PIPELINE


num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])


# 8. CATEGORICAL PIPELINE


cat_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])


# 9. FULL PIPELINE


full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', cat_pipeline, cat_attribs)
])


# 10. PREPARE DATA


housing_prepared = full_pipeline.fit_transform(housing)

print("\nData preprocessing completed successfully!")

# 11. LINEAR REGRESSION MODEL


lin_reg = LinearRegression()

lin_reg.fit(housing_prepared, housing_labels)

lin_scores = -cross_val_score(
    lin_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10
)

lin_rmse = np.sqrt(lin_scores)

print("\n * Linear Regression RMSE Scores *")
print(pd.Series(lin_rmse).describe())


# 12. DECISION TREE MODEL


dec_reg = DecisionTreeRegressor(random_state=42)

dec_reg.fit(housing_prepared, housing_labels)

dec_scores = -cross_val_score(
    dec_reg,
    housing_prepared,
    housing_labels,
    scoring="neg_mean_squared_error",
    cv=10
)

dec_rmse = np.sqrt(dec_scores)

print("\n* Decision Tree RMSE Scores *")
print(pd.Series(dec_rmse).describe())

# 13. SAMPLE PREDICTION

sample_data = pd.DataFrame({
    'longitude': [-122.23],
    'latitude': [37.88],
    'housing_median_age': [41],
    'total_rooms': [880],
    'total_bedrooms': [129],
    'population': [322],
    'households': [126],
    'median_income': [8.3252],
    'ocean_proximity': ['NEAR BAY']
})

# Transform sample data
sample_prepared = full_pipeline.transform(sample_data)

# Predict price
prediction = lin_reg.predict(sample_prepared)

# 14. FINAL OUTPUT


print("* HOUSE PRICE PREDICTION SYSTEM *")


print("\nSample House Details:")
print("Longitude:", -122.23)
print("Latitude:", 37.88)
print("House Age:", 41)
print("Total Rooms:", 880)
print("Median Income:", 8.3252)
print("Ocean Proximity:", "NEAR BAY")

print("\nPredicted House Price: $", round(prediction[0], 2))

# Convert USD to INR
usd_to_inr = 83

price_in_inr = prediction[0] * usd_to_inr

print("Predicted House Price in INR: ₹", round(price_in_inr, 2))

print("Project Execution Completed Successfully")
