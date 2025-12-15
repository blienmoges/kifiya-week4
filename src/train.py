# src/train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import mlflow
import mlflow.sklearn
import category_encoders as ce

# --------------------------
# 1. Load Data
# --------------------------
data_path = './data/final_training_data.csv'
df = pd.read_csv(data_path)

# --------------------------
# 2. Target & Features
# --------------------------
target = 'is_high_risk'  # Make sure Task-4 has been completed

if target not in df.columns:
    raise ValueError(f"{target} column is missing. Run Task-4 first.")

X = df.drop(columns=[target])
y = df[target]

# --------------------------
# 3. Drop high-cardinality IDs
# --------------------------
drop_cols = ['TransactionId', 'BatchId', 'SubscriptionId', 'AccountId', 'ProductId']
X = X.drop(columns=[c for c in drop_cols if c in X.columns])

# --------------------------
# 4. Datetime processing
# --------------------------
if 'TransactionStartTime' in X.columns:
    X['TransactionStartTime'] = pd.to_datetime(X['TransactionStartTime'])
    X['hour'] = X['TransactionStartTime'].dt.hour
    X['day'] = X['TransactionStartTime'].dt.day
    X['month'] = X['TransactionStartTime'].dt.month
    X['weekday'] = X['TransactionStartTime'].dt.weekday
    X = X.drop(columns=['TransactionStartTime'])

# --------------------------
# 5. Encode categorical features
# --------------------------
low_card_cols = ['CurrencyCode', 'CountryCode', 'ChannelId', 'PricingStrategy']
high_card_cols = ['CustomerId', 'ProviderId', 'ProductCategory']

# Target encoding for high-cardinality
target_encoder = ce.TargetEncoder(cols=[c for c in high_card_cols if c in X.columns])
X[high_card_cols] = target_encoder.fit_transform(X[high_card_cols], y)

# One-hot encoding for low-cardinality
X = pd.get_dummies(X, columns=[c for c in low_card_cols if c in X.columns], drop_first=True)

# --------------------------
# 6. Train/Test Split
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --------------------------
# 7. Scale numeric features
# --------------------------
numeric_cols = X_train.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])

# --------------------------
# 8. MLflow Experiment
# --------------------------
mlflow.set_experiment("Credit_Risk_Model_Training")

with mlflow.start_run(run_name="LogisticRegression_RandomForest"):

    # ----------------------
    # Logistic Regression
    # ----------------------
    print("Training LogisticRegression...")
    lr = LogisticRegression(max_iter=1000, solver='lbfgs')
    lr_param_grid = {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l2']
    }
    lr_grid = GridSearchCV(lr, lr_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    lr_grid.fit(X_train, y_train)

    best_lr = lr_grid.best_estimator_
    y_pred_lr = best_lr.predict(X_test)
    y_proba_lr = best_lr.predict_proba(X_test)[:, 1]

    # Log metrics
    mlflow.log_param("LR_best_C", lr_grid.best_params_['C'])
    mlflow.log_metric("LR_accuracy", accuracy_score(y_test, y_pred_lr))
    mlflow.log_metric("LR_precision", precision_score(y_test, y_pred_lr))
    mlflow.log_metric("LR_recall", recall_score(y_test, y_pred_lr))
    mlflow.log_metric("LR_f1", f1_score(y_test, y_pred_lr))
    mlflow.log_metric("LR_roc_auc", roc_auc_score(y_test, y_proba_lr))
    mlflow.sklearn.log_model(best_lr, "logistic_regression_model")

    # ----------------------
    # Random Forest
    # ----------------------
    print("Training RandomForestClassifier...")
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    rf_param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5]
    }
    rf_grid = GridSearchCV(rf, rf_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
    rf_grid.fit(X_train, y_train)

    best_rf = rf_grid.best_estimator_
    y_pred_rf = best_rf.predict(X_test)
    y_proba_rf = best_rf.predict_proba(X_test)[:, 1]

    # Log metrics
    mlflow.log_param("RF_best_params", rf_grid.best_params_)
    mlflow.log_metric("RF_accuracy", accuracy_score(y_test, y_pred_rf))
    mlflow.log_metric("RF_precision", precision_score(y_test, y_pred_rf))
    mlflow.log_metric("RF_recall", recall_score(y_test, y_pred_rf))
    mlflow.log_metric("RF_f1", f1_score(y_test, y_pred_rf))
    mlflow.log_metric("RF_roc_auc", roc_auc_score(y_test, y_proba_rf))
    mlflow.sklearn.log_model(best_rf, "random_forest_model")

    print("Training completed. Check MLflow UI for experiment logs.")
