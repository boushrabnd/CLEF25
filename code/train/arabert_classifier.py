import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import torch
from sklearn.metrics import classification_report
from arabert.preprocess import ArabertPreprocessor

# Define the model name for AraBERT and initialize the preprocessor
model_name = "aubmindlab/bert-base-arabertv02"
arabert_prep = ArabertPreprocessor(model_name=model_name)

# File path and dataset loading
file_path = '/Users/bushrabendou/Desktop/IndependentStudy/CLEF25/data/CLEF25_dataset.json'
df = pd.read_json(file_path, lines=True)

# Preprocess claims using AraBERT preprocessor
df['claim'] = df['claim'].apply(arabert_prep.preprocess)

# Extract features and labels
X = df['claim']
y = df['maj_label']

# Encode the labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)  # Convert labels to numeric values

# Load AraBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_encoder.classes_))

# Tokenize the data
tokens = tokenizer(
    list(X), padding=True, truncation=True, return_tensors="pt", max_length=512
)
input_ids, attention_mask = tokens['input_ids'], tokens['attention_mask']

# First split: 80% training+validation and 20% testing
X_train_val_ids, X_test_ids, y_train_val, y_test = train_test_split(
    input_ids, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
X_train_val_mask, X_test_mask = train_test_split(
    attention_mask, test_size=0.2, stratify=y_encoded, random_state=42
)

# Second split: 70% training and 10% validation (from the 80% training+validation split)
X_train_ids, X_val_ids, y_train, y_val = train_test_split(
    X_train_val_ids, y_train_val, test_size=0.125, stratify=y_train_val, random_state=42
)
X_train_mask, X_val_mask = train_test_split(
    X_train_val_mask, test_size=0.125, stratify=y_train_val, random_state=42
)

# Prepare DataLoader for training, validation, and testing
train_data = TensorDataset(X_train_ids, X_train_mask, torch.tensor(y_train))
val_data = TensorDataset(X_val_ids, X_val_mask, torch.tensor(y_val))
test_data = TensorDataset(X_test_ids, X_test_mask, torch.tensor(y_test))

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)
test_loader = DataLoader(test_data, batch_size=8)

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
loss_fn = torch.nn.CrossEntropyLoss()

# Train the model
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        input_ids, attention_mask, labels = [x.to(device) for x in batch]

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader)}")

# Evaluate the model
def evaluate(loader):
    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for batch in loader:
            input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
    return y_true, y_pred

# Validation evaluation
y_val_true, y_val_pred = evaluate(val_loader)
val_report = classification_report(
    label_encoder.inverse_transform(y_val_true),
    label_encoder.inverse_transform(y_val_pred),
    target_names=label_encoder.classes_,
)

# Test evaluation
y_test_true, y_test_pred = evaluate(test_loader)
test_report = classification_report(
    label_encoder.inverse_transform(y_test_true),
    label_encoder.inverse_transform(y_test_pred),
    target_names=label_encoder.classes_,
)

# Paths for organized reports
val_report_path = "results/validation/arabert_report.txt"
test_report_path = "results/test/arabert_report.txt"

# Ensure directories exist
os.makedirs(os.path.dirname(val_report_path), exist_ok=True)
os.makedirs(os.path.dirname(test_report_path), exist_ok=True)

# Save the reports to files
with open(val_report_path, "w") as f:
    f.write(val_report)

with open(test_report_path, "w") as f:
    f.write(test_report)

# Save the trained model
model_path = "models/arabert_claim_classifier_model.pt"
torch.save(model.state_dict(), model_path)

# Display results
print("Validation Report:")
print(val_report)
print("Test Report:")
print(test_report)

print(f"Model saved to: {model_path}")
print(f"Validation report saved to: {val_report_path}")
print(f"Test report saved to: {test_report_path}")
