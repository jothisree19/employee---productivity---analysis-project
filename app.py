# ================================
# STEP 0: Check & Unzip Dataset
# ================================

# List files
!ls

# Unzip dataset (make sure zip file name matches exactly)
!unzip garments_worker_productivity.zip

# Check extracted files
!ls


# ================================
# STEP 1: Import Libraries
# ================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor

import joblib


# ================================
# STEP 2: Load Dataset
# ================================

df = pd.read_csv("garments_worker_productivity.csv")

print("Dataset Shape:", df.shape)
df.head()


# ================================
# STEP 3: Basic Data Cleaning
# ================================

# Drop date column (not useful for prediction)
df.drop(columns=['date'], inplace=True)

# Fill missing values
df.fillna(df.mean(numeric_only=True), inplace=True)

df.isnull().sum()


# ================================
# STEP 4: Encode Categorical Columns
# ================================

label_encoder = LabelEncoder()

categorical_cols = ['quarter', 'department', 'day', 'team']

for col in categorical_cols:
    df[col] = label_encoder.fit_transform(df[col])

df.head()


# ================================
# STEP 5: Split Features & Target
# ================================

X = df.drop(columns=['actual_productivity'])
y = df['actual_productivity']

print("Features Shape:", X.shape)
print("Target Shape:", y.shape)


# ================================
# STEP 6: Train-Test Split
# ================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ================================
# STEP 7: Train Model
# ================================

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("Model training completed ✅")


# ================================
# STEP 8: Predictions
# ================================

y_pred = model.predict(X_test)


# ================================
# STEP 9: Model Evaluation
# ================================

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("R2 Score:", r2)


# ================================
# STEP 10: Save Model
# ================================

joblib.dump(model, "employee_productivity_model.pkl")

print("Model saved as employee_productivity_model.pkl ✅")


# ================================
# STEP 11: Sample Prediction
# ================================

sample_input = X.iloc[0:1]
sample_prediction = model.predict(sample_input)

print("Sample Prediction:", sample_prediction)