import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, matthews_corrcoef
)

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("bank.csv")

# Target column
df['deposit'] = df['deposit'].map({'yes': 1, 'no': 0})

X = df.drop('deposit', axis=1)
y = df['deposit']

# Column types
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=True, handle_unknown='ignore'), categorical_cols)
    ]
)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Fit-transform train, transform test
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

# -----------------------------
# Models
# -----------------------------
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=5),
    "Naive_Bayes": GaussianNB(),  # will convert to dense
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

# -----------------------------
# Train, evaluate, save
# -----------------------------
os.makedirs("model", exist_ok=True)
results = []

for name, model in models.items():
    # Special handling for GaussianNB: needs dense
    if name == "Naive_Bayes":
        X_train_dense = X_train_p.toarray() if hasattr(X_train_p, "toarray") else X_train_p
        X_test_dense = X_test_p.toarray() if hasattr(X_test_p, "toarray") else X_test_p
        model.fit(X_train_dense, y_train)
        y_pred = model.predict(X_test_dense)
        y_prob = model.predict_proba(X_test_dense)[:, 1]
    else:
        model.fit(X_train_p, y_train)
        y_pred = model.predict(X_test_p)
        y_prob = model.predict_proba(X_test_p)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_prob),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    })

    joblib.dump(model, f"model/{name}.pkl")

# -----------------------------
# Results
# -----------------------------
results_df = pd.DataFrame(results)
print(results_df)

# Save preprocessor for Streamlit app
joblib.dump(preprocessor, "model/preprocessor.pkl")
