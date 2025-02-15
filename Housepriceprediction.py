import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

base_path = "d:/Python/prodigy infotech intern/Task 1/"
train_file = base_path + "train.csv"
test_file = base_path + "test.csv"
submission_file = base_path + "sample_submission.csv"

df_train = pd.read_csv(train_file)
df_test = pd.read_csv(test_file)

print("Training Dataset Sample:\n", df_train.head())

target_column = 'SalePrice'
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'YearBuilt', 'LotArea', 'TotRmsAbvGrd', 'MasVnrArea', 'Fireplaces']

for col in features + [target_column]:
    if col not in df_train.columns:
        print(f"Error: Missing column {col} in the training dataset.")
        exit()

X_train = df_train[features]
y_train = df_train[target_column]

y_train = np.log1p(y_train)

imputer = SimpleImputer(strategy="median")
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=features)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)

gb_model = GradientBoostingRegressor()

param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0]
}

grid_search = GridSearchCV(gb_model, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
grid_search.fit(X_train_poly, y_train)

best_gb_model = grid_search.best_estimator_

y_train_pred = best_gb_model.predict(X_train_poly)

y_train_pred = np.expm1(y_train_pred)
y_train = np.expm1(y_train)

mse_train = mean_squared_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

print(f"Optimized Training Mean Squared Error with Gradient Boosting: {mse_train:.5f}")
print(f"Optimized Training R² Score with Gradient Boosting: {r2_train:.5f}")

if not set(features).issubset(df_test.columns):
    print("Error: Some required features are missing in the test dataset.")
    exit()

X_test = df_test[features]

X_test = pd.DataFrame(imputer.transform(X_test), columns=features)
X_test_scaled = scaler.transform(X_test)
X_test_poly = poly.transform(X_test_scaled)

test_predictions = best_gb_model.predict(X_test_poly)
test_predictions = np.expm1(test_predictions) 

if 'Id' not in df_test.columns:
    print("Error: 'Id' column is missing in test dataset.")
    exit()

submission = pd.DataFrame({'Id': df_test['Id'], 'SalePrice': test_predictions})

submission_output = base_path + "optimized_submission.csv"
submission.to_csv(submission_output, index=False)
print(f"Predictions saved successfully to {submission_output}")

plt.scatter(y_train, y_train_pred, color='blue', alpha=0.5)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Optimized: Actual vs Predicted House Prices")
plt.show()