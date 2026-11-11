import kagglehub
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# 2. Download Dataset

print("Downloading Housing Prices Dataset...")

path = kagglehub.dataset_download("yasserh/housing-prices-dataset")

print("Dataset downloaded successfully!")
print("Dataset path:", path)

# 3. Load Dataset

df = pd.read_csv(os.path.join(path, "Housing.csv"))

print("\nFirst Five Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

# 4. Data Preprocessing

print("\nMissing Values:")
print(df.isnull().sum())

# Handle missing numerical and categorical values
for col in df.columns:
    if df[col].isnull().any():
        if df[col].dtype == "object":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

# Convert categorical variables into numerical variables
df = pd.get_dummies(df, drop_first=True, dtype=int)

# Separate features and target
X = df.drop("price", axis=1)
y = df["price"]

# 5. Split Dataset

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# 6. Train Linear Regression Model

model = LinearRegression()

model.fit(X_train, y_train)

print("\nLinear Regression Model Trained Successfully!")

# 7. Predict House Prices

y_pred = model.predict(X_test)

results = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

print("\nActual vs Predicted Prices:")
print(results.head(10).round(2))

# 8. Evaluate Model Performance

r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("R2 Score:", round(r2, 4))
print("R2 Score Percentage:", round(r2 * 100, 2), "%")
print("Mean Squared Error:", round(mse, 2))
print("Root Mean Squared Error:", round(rmse, 2))
print("Mean Absolute Error:", round(mae, 2))

# 9. Plot Actual vs Predicted Values

plt.figure(figsize=(9, 6))

plt.scatter(y_test, y_pred, alpha=0.7)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    color="red",
    linestyle="--",
    linewidth=2
)

plt.xlabel("Actual House Prices")
plt.ylabel("Predicted House Prices")
plt.title("Actual vs Predicted House Prices")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 10. Display Final Results

print("\n========== PROJECT SUMMARY ==========")
print("Project: House Price Prediction Model")
print("Algorithm: Linear Regression")
print("Dataset: Kaggle Housing Prices")
print("Total Records:", len(df))
print("Training Records:", len(X_train))
print("Testing Records:", len(X_test))
print("R2 Score:", round(r2, 4))

print("\nProject Completed Successfully!")

