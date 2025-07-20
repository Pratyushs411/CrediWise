# train_model.py

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("data/train.csv")

# Drop Loan_ID and handle target column
df.drop("Loan_ID", axis=1, inplace=True)
df.dropna(subset=["Loan_Status"], inplace=True)
df["Loan_Status"] = df["Loan_Status"].map({"Y": 1, "N": 0})

# Separate features and target
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# One-hot encode categorical features
X = pd.get_dummies(X)

# Save the column structure for future use
joblib.dump(X.columns.tolist(), "model/feature_columns.pkl")

# Split data BEFORE fitting the imputer
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# Impute missing values using median
imputer = SimpleImputer(strategy="median")
X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_imputed, y_train)

# Save model and imputer
joblib.dump(model, "model/loan_model.pkl")
joblib.dump(imputer, "model/imputer.pkl")

print("✅ Model training complete. Files saved to 'model/'")
