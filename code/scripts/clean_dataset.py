import json
from collections import Counter

def process_json(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = infile.readlines()

    processed_data = []
    factuality_categories = Counter()  # Counter to track categorized frequencies

    # Define mapping for categorization
    label_mapping = {
        "True": "true",
        "False": "false",
        "Partly-false": "conflicting",
        "Half True/False": "conflicting",
        "Other": "conflicting"
    }

    for idx, line in enumerate(data):
        try:
            item = json.loads(line)
            if item["maj_label"] == "numerical":  # Process only numerical claims
                # Categorize the label
                category = label_mapping.get(item["factuality_label"], "conflicting")
                factuality_categories[category] += 1
                
                new_entry = {
                    "id": idx + 1,  # Generate new id starting from 1
                    "claim": item["claim"],
                    "factuality_label": category
                }
                processed_data.append(new_entry)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON line: {line}")

    # Print the categorized frequencies
    print("Categorized Factuality Frequencies (Numerical Claims):", factuality_categories)

    # Write the new JSON objects to the output file
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for entry in processed_data:
            outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

# Example usage
# Replace 'input.json' with the path to your input file, and 'output.json' with the desired output path
process_json('/Users/bushrabendou/Desktop/IndependentStudy/export_105552_project-105552-at-2024-11-28-08-56-3aecdd3d_labels2.json', '/Users/bushrabendou/Desktop/IndependentStudy/CLEF25/data/CLEF25_arabic_numerical_factuality_dataset.json')
