import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("dataset.csv")

# Encode categorical columns
label_encoders = {}
categorical_cols = [
    "material_type",
    "usage_frequency",
    "humidity_exposure",
    "load_stress_level",
    "cracks_visible"
]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Encode target
target_encoder = LabelEncoder()
df["maintenance_status"] = target_encoder.fit_transform(df["maintenance_status"])

# Features and target
X = df.drop("maintenance_status", axis=1)
y = df["maintenance_status"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and encoders
joblib.dump(model, "maintenance_model.pkl")
joblib.dump(label_encoders, "encoders.pkl")
joblib.dump(target_encoder, "target_encoder.pkl")

print("Model and encoders saved successfully.")
