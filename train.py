
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "dataset.csv"
MODEL_PATH = BASE_DIR / "symptom_model.joblib"

# 1. Load the Kaggle dataset
df = pd.read_csv(DATASET_PATH)

symptom_columns = [
    col for col in df.columns
    if col.startswith("Symptom_")
]

if "Disease" not in df.columns or not symptom_columns:
    raise ValueError("Check the dataset's column names.")

# 2. Clean the target labels
df = df.dropna(subset=["Disease"]).copy()
df["Disease"] = df["Disease"].astype(str).str.strip()

# 3. Convert each row into a list of symptoms
def extract_symptoms(row):
    symptoms = []

    for col in symptom_columns:
        value = row[col]

        if pd.notna(value):
            value = str(value).strip().lower()

            if value and value != "nan":
                symptoms.append(value)

    return sorted(set(symptoms))


df["symptoms_list"] = df.apply(extract_symptoms, axis=1)
df = df[df["symptoms_list"].map(len) > 0]

# 4. Convert symptoms into binary features
encoder = MultiLabelBinarizer()
X = encoder.fit_transform(df["symptoms_list"])
y = df["Disease"]

print("Dataset rows:", len(df))
print("Disease labels:", y.nunique())
print("Symptom features:", len(encoder.classes_))
print("Available symptoms:", list(encoder.classes_))

# 5. Split into training and testing data
stratify_labels = y if y.value_counts().min() >= 2 else None

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=stratify_labels,
)

# 6. Train the classifier
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

# 7. Evaluate performance
predictions = model.predict(X_test)

print("\nTest accuracy:", accuracy_score(y_test, predictions))
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0,
))

# 8. Save the model and symptom encoder together
joblib.dump(
    {"model": model, "encoder": encoder},
    MODEL_PATH,
)

print("\nModel saved to:", MODEL_PATH)
