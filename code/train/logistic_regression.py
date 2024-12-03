import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

# File path and dataset loading
file_path = '/Users/bushrabendou/Desktop/IndependentStudy/CLEF25/data/CLEF25_dataset.json'
df = pd.read_json(file_path, lines=True)

# Extract features and labels
X = df['claim']
y = df['maj_label']

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# First split: 80% training+validation and 20% testing
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X_vectorized, y, test_size=0.2, stratify=y, random_state=42
)

# Second split: 70% training and 10% validation (from the 80% training+validation split)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.125, stratify=y_train_val, random_state=42
)
# Note: 0.125 of 80% = 10% of the original data

# Train the classifier
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X_train, y_train)

# Save the trained model and the vectorizer to a specified path
model_path = "models/logistic_regression_model.pkl"
vectorizer_path = "models/tfidf_vectorizer.pkl"
joblib.dump(classifier, model_path)
joblib.dump(vectorizer, vectorizer_path)

# Evaluate on the validation set
y_val_pred = classifier.predict(X_val)
val_report = classification_report(y_val, y_val_pred, target_names=classifier.classes_)

# Evaluate on the test set
y_test_pred = classifier.predict(X_test)
test_report = classification_report(y_test, y_test_pred, target_names=classifier.classes_)

# Paths for organized reports
val_report_path = "results/validation/logistic_regression_report.txt"
test_report_path = "results/test/logistic_regression_report.txt"

# Ensure directories exist
os.makedirs(os.path.dirname(val_report_path), exist_ok=True)
os.makedirs(os.path.dirname(test_report_path), exist_ok=True)

# Save the reports to files
with open(val_report_path, "w") as f:
    f.write(val_report)

with open(test_report_path, "w") as f:
    f.write(test_report)

# Display results
print("Validation Report:")
print(val_report)
print("Test Report:")
print(test_report)

print(f"Model saved to: {model_path}")
print(f"Vectorizer saved to: {vectorizer_path}")
print(f"Validation report saved to: {val_report_path}")
print(f"Test report saved to: {test_report_path}")
