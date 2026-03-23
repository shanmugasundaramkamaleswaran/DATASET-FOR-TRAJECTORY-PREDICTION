import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report, r2_score
import os

# --- Load Dataset ---
dataset_path = r'c:\Users\ELCOT\p\dataset.csv'

# Check if dataset exists and is not empty
if not os.path.exists(dataset_path) or os.path.getsize(dataset_path) == 0:
    print(f"Dataset file '{dataset_path}' is empty or not found.")
    print("Creating a sample dataset for demonstration...")
    
    # Sample data for Indoor Air Pollution
    data = {
        'CO': [0.1, 0.4, 0.2, 0.8, 0.5, 0.1, 1.2, 0.3, 0.9, 0.2],
        'CO2': [400, 450, 420, 600, 500, 410, 800, 430, 700, 415],
        'PM2.5': [5, 12, 8, 35, 20, 6, 50, 9, 30, 7],
        'Temperature': [22, 23, 22, 25, 24, 22, 26, 23, 25, 22],
        'Humidity': [45, 48, 46, 60, 55, 45, 65, 47, 58, 46],
        'AirQualityIdx': [10, 25, 15, 60, 40, 12, 90, 18, 55, 14], # Continuous
        'Status': [0, 0, 0, 1, 1, 0, 1, 0, 1, 0] # 0: Good, 1: Poor (Categorical)
    }
    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)
    print(f"Sample dataset saved to {dataset_path}.")
else:
    df = pd.read_csv(dataset_path)
    print("Dataset loaded successfully.")

# --- Data Exploration ---
print("\n--- Dataset Preview ---")
print(df.head())

# --- Feature Selection ---
# Features for both: CO, CO2, PM2.5, Temperature, Humidity
X = df[['CO', 'CO2', 'PM2.5', 'Temperature', 'Humidity']]

# Targets
y_reg = df['AirQualityIdx'] if 'AirQualityIdx' in df.columns else df.iloc[:, -2] # Regression
y_clf = df['Status'] if 'Status' in df.columns else df.iloc[:, -1] # Classification

# Splitting
X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_clf_train, y_clf_test = train_test_split(X, y_clf, test_size=0.2, random_state=42)

# --- 1. Linear Regression ---
print("\n--- 1. Linear Regression Output ---")
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_reg_train)
y_reg_pred = lin_reg.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_reg_test, y_reg_pred):.4f}")
print(f"R2 Score: {r2_score(y_reg_test, y_reg_pred):.4f}")
print(f"Sample Predictions: {y_reg_pred[:3]}")

# --- 2. Logistic Regression ---
print("\n--- 2. Logistic Regression Output ---")
# If y_clf has only one class, this will error, but we expect it to be binary
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_clf_train)
y_clf_pred_log = log_reg.predict(X_test)
print(f"Accuracy: {accuracy_score(y_clf_test, y_clf_pred_log):.4f}")
print("Classification Report:")
print(classification_report(y_clf_test, y_clf_pred_log, zero_division=0))

# --- 3. Decision Tree ---
print("\n--- 3. Decision Tree Output ---")
dt_clf = DecisionTreeClassifier(random_state=42)
dt_clf.fit(X_train, y_clf_train)
y_clf_pred_dt = dt_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_clf_test, y_clf_pred_dt):.4f}")
print("Classification Report:")
print(classification_report(y_clf_test, y_clf_pred_dt, zero_division=0))

# --- 4. Random Forest Output ---
print("\n--- 4. Random Forest Output ---")
rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rf_clf.fit(X_train, y_clf_train)
y_clf_pred_rf = rf_clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_clf_test, y_clf_pred_rf):.4f}")
print("Classification Report:")
print(classification_report(y_clf_test, y_clf_pred_rf, zero_division=0))

# Summary
print("\n--- Summary ---")
print("All four models have been trained and evaluated.")
