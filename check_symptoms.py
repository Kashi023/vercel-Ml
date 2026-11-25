
import pandas as pd

df = pd.read_csv("dataset.csv")

symptom_columns = [
    col for col in df.columns
    if col.startswith("Symptom_")
]

all_symptoms = set()

for column in symptom_columns:
    values = df[column].dropna().astype(str).str.strip().str.lower()
    all_symptoms.update(value for value in values if value)

print("Total rows:", len(df))
print("Total disease labels:", df["Disease"].nunique())
print("\nAll symptoms found in the dataset:")

for symptom in sorted(all_symptoms):
    print(symptom)

print("\nDisease counts:")
print(df["Disease"].value_counts())
