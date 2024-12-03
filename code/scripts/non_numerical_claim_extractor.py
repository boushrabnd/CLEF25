import json
import pandas as pd

# Load JSON data from the file
with open('/Users/bushrabendou/Desktop/IndependentStudy/CLEF25/data/export_105552_no_zero_agreement_labels.json', 'r', encoding='utf-8') as f:
    json_data = [json.loads(line) for line in f]

# Extract existing claims
existing_claims = {entry["claim"] for entry in json_data}

# Load CSV data
df = pd.read_csv('/Users/bushrabendou/Desktop/IndependentStudy/CLEF25/data/AraFacts.csv')

# Extract new claims from the CSV
new_claims = df['claim'].dropna().tolist()

# Separate claims into unique and already existing
unique_claims = []
ignored_count = 0

for claim in new_claims:
    if claim in existing_claims:
        ignored_count += 1
    else:
        unique_claims.append(claim)

# Format the unique claims into the required JSON structure
new_entries = [{"id": i + 1, "claim": claim, "labels": [], "maj_label": "", "full_agmnt": ""} 
               for i, claim in enumerate(unique_claims)]

# Save the new entries to a JSON file
output_path = '/Users/bushrabendou/Desktop/IndependentStudy/CLEF25/data/non_numerical_claims.json'
with open(output_path, 'w', encoding='utf-8') as f:
    for entry in new_entries:
        json.dump(entry, f, ensure_ascii=False)
        f.write('\n')

# Output results
print(f"New claims have been saved to {output_path}")
print(f"Number of ignored (existing) claims: {ignored_count}")
