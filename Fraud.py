import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report

# -------------------------------
# 1. Get project folder dynamically
# -------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "creditcard.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------------
# 2. Load dataset
# -------------------------------
data = pd.read_csv(DATA_PATH)
X = data.drop("Class", axis=1)
y = data["Class"]

# -------------------------------
# 3. Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# 4. Scale features
# -------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))

# -------------------------------
# 5. Define models
# -------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        n_jobs=-1,      # use all cores
        verbose=2,      # show progress per tree
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100,
        verbose=2,      # show progress per stage
        random_state=42
    )
}

# -------------------------------
# 6. Train, Evaluate, Save
# -------------------------------
for name, model in models.items():
    print(f"\nTraining {name} ...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{name.replace(' ', '_').lower()}.pkl")
    joblib.dump(model, model_path)
    print(f"✅ Model saved to {model_path}")
