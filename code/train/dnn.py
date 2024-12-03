import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import numpy as np

# Load the dataset
file_path = 'data/export_105552_project-105552-at-2024-11-27-19-44-9355ba34_labels.json'
df = pd.read_json(file_path, lines=True)

# Extract features and labels
X = df['claim']
y = df['maj_label']

# Encode the labels numerically
label_mapping = {label: idx for idx, label in enumerate(y.unique())}
y_encoded = y.map(label_mapping)

# Vectorize the text data using TF-IDF
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Convert the sparse matrix to a dense format for compatibility with Keras
X_dense = X_vectorized.toarray()

# Split the dataset into training and testing sets with a balanced distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_dense, y_encoded, test_size=500, stratify=y_encoded, random_state=42
)

# Build a simple DNN
model = Sequential([
    Dense(128, activation='relu', input_dim=X_train.shape[1]),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dense(len(label_mapping), activation='softmax')  # Number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(np.array(X_train), np.array(y_train), epochs=10, batch_size=32, validation_data=(np.array(X_test), np.array(y_test)))

# Save the model and vectorizer
model_path = "models/dnn_claim_classifier_model.h5"
vectorizer_path = "models/tfidf_vectorizer.pkl"
model.save(model_path)

import joblib
joblib.dump(vectorizer, vectorizer_path)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(np.array(X_test), np.array(y_test))
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")

# Save a report
report_path = "results/dnn_classification_report.txt"
with open(report_path, "w") as f:
    f.write(f"Test Loss: {loss}\n")
    f.write(f"Test Accuracy: {accuracy}\n")

# Display results
print(f"Model saved to: {model_path}")
print(f"Vectorizer saved to: {vectorizer_path}")
print(f"Classification report saved to: {report_path}")
