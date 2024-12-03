import json

# File paths
file1 = '/Users/bushrabendou/Desktop/IndependentStudy/CLEF25/data/sampled_non_numerical_claims_AraFacts.json'  
file2 = '/Users/bushrabendou/Desktop/IndependentStudy/CLEF25/data/annotated_claims.json'  
output_file = '/Users/bushrabendou/Desktop/IndependentStudy/CLEF25/data/CLEF25_dataset.json'

# Function to filter out claims labeled as "other"
def filter_out_other(data):
    return [entry for entry in data if entry.get('maj_label') != 'other']

# Load and filter data from the first file
with open(file1, 'r', encoding='utf-8') as f1:
    data1 = [json.loads(line) for line in f1]
    filtered_data1 = filter_out_other(data1)

# Load and filter data from the second file
with open(file2, 'r', encoding='utf-8') as f2:
    data2 = [json.loads(line) for line in f2]
    filtered_data2 = filter_out_other(data2)

# Merge the filtered datasets
merged_data = filtered_data1 + filtered_data2

# Save the merged dataset to a new file
with open(output_file, 'w', encoding='utf-8') as out:
    for entry in merged_data:
        out.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Merged and filtered dataset saved to {output_file}")

