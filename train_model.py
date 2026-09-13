import pandas as pd
import numpy as np
import joblib

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# 1. LOAD DATA
# =========================================================

df = pd.read_csv("loan_approval_data.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)


# =========================================================
# 2. HANDLE MISSING VALUES
# =========================================================

categorical_cols = df.select_dtypes(include=["object"]).columns
numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns


# Numerical columns -> Mean
num_imp = SimpleImputer(strategy="mean")
df[numerical_cols] = num_imp.fit_transform(df[numerical_cols])


# Categorical columns -> Most frequent
cat_imp = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imp.fit_transform(df[categorical_cols])


# =========================================================
# 3. REMOVE APPLICANT ID
# =========================================================

df = df.drop("Applicant_ID", axis=1)


# =========================================================
# 4. LABEL ENCODING
# =========================================================

education_encoder = LabelEncoder()
target_encoder = LabelEncoder()

df["Education_Level"] = education_encoder.fit_transform(
    df["Education_Level"]
)

df["Loan_Approved"] = target_encoder.fit_transform(
    df["Loan_Approved"]
)


# =========================================================
# 5. ONE HOT ENCODING
# =========================================================

cols = [
    "Employment_Status",
    "Marital_Status",
    "Loan_Purpose",
    "Property_Area",
    "Gender",
    "Employer_Category"
]

ohe = OneHotEncoder(
    drop="first",
    sparse_output=False,
    handle_unknown="ignore"
)

encoded = ohe.fit_transform(df[cols])

encoded_df = pd.DataFrame(
    encoded,
    columns=ohe.get_feature_names_out(cols),
    index=df.index
)


# =========================================================
# 6. COMBINE ENCODED DATA
# =========================================================

df = pd.concat(
    [
        df.drop(columns=cols),
        encoded_df
    ],
    axis=1
)


# =========================================================
# 7. FEATURE ENGINEERING
# =========================================================

df["DTI_Ratio_sq"] = df["DTI_Ratio"] ** 2

df["Credit_Score_sq"] = df["Credit_Score"] ** 2


# =========================================================
# 8. CREATE X AND Y
# =========================================================

X = df.drop(
    columns=[
        "Loan_Approved",
        "Credit_Score",
        "DTI_Ratio"
    ]
)

Y = df["Loan_Approved"]


# Save exact feature order
feature_columns = X.columns.tolist()


print("\nFinal features:")
print(feature_columns)

print("\nNumber of features:", len(feature_columns))


# =========================================================
# 9. TRAIN TEST SPLIT
# =========================================================

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)


# =========================================================
# 10. FEATURE SCALING
# =========================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


# =========================================================
# 11. TRAIN LOGISTIC REGRESSION
# =========================================================

log_model = LogisticRegression()

log_model.fit(
    X_train_scaled,
    Y_train
)


# =========================================================
# 12. PREDICTION
# =========================================================

Y_pred = log_model.predict(X_test_scaled)


# =========================================================
# 13. MODEL EVALUATION
# =========================================================

accuracy = accuracy_score(Y_test, Y_pred)
precision = precision_score(Y_test, Y_pred)
recall = recall_score(Y_test, Y_pred)
f1 = f1_score(Y_test, Y_pred)

cm = confusion_matrix(Y_test, Y_pred)


print("\n========================================")
print("       CREDITWISE MODEL RESULTS")
print("========================================")

print(f"Accuracy  : {accuracy * 100:.2f}%")
print(f"Precision : {precision * 100:.2f}%")
print(f"Recall    : {recall * 100:.2f}%")
print(f"F1 Score  : {f1 * 100:.2f}%")

print("\nConfusion Matrix:")
print(cm)


# =========================================================
# 14. SAVE EVERYTHING REQUIRED FOR DEPLOYMENT
# =========================================================

model_bundle = {

    "model": log_model,

    "scaler": scaler,

    "ohe": ohe,

    "education_encoder": education_encoder,

    "target_encoder": target_encoder,

    "feature_columns": feature_columns
}


joblib.dump(
    model_bundle,
    "creditwise_model.pkl"
)


print("\n========================================")
print("Model saved successfully!")
print("File: creditwise_model.pkl")
print("========================================")