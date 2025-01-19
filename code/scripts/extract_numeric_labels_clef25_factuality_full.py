import json
from collections import Counter


def extract_ids_and_labels(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    results = []

    for item in data:
        # Extract the "id" value
        id_value = item.get("id")

        # Initialize a dictionary to store extracted information for each item
        extracted_data = {"id": id_value}

        # Extract and process the "data" fields (all fields from 'data')
        data_fields = item.get("data", {})
        extracted_data.update(data_fields)

        # Add the factuality_label field
        original_label = data_fields.get("label", "")
        if original_label == "True":
            extracted_data["factuality_label"] = "True"
        elif original_label == "False":
            extracted_data["factuality_label"] = "False"
        else:
            extracted_data["factuality_label"] = "Conflicting"

        # Extract and process "annotations" if it exists
        annotations = item["annotations"]
        if annotations:
            labels = []
            for annot in annotations:
                label = annot["result"][0]["value"]["choices"][0]
                if label == "عددية":
                    label = "numerical"
                elif label == "غير عددية":
                    label = "non_numerical"
                else:
                    label = "other"

                labels.append(label.strip())

            # extracted_data["labels"] = labels

            # Determine the majority label for 'numerical_label'
            numerical_label = Counter(labels).most_common(1)[0][0]
            if "numerical" in labels and "other" in labels and "non_numerical" in labels:
                extracted_data["numerical_label"] = "no_numerical_label"
            else:
                extracted_data["numerical_label"] = numerical_label

        # Append the extracted data to the results list
        results.append(extracted_data)

    # Write the results to the output JSON file
    with open(output_file, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False))
            f.write("\n")


# Usage example
input_fname = "/Users/bushrabendou/Desktop/IndependentStudy/export_105552_project-105552-at-2024-11-28-08-56-3aecdd3d.json"
output_fname = "/Users/bushrabendou/Desktop/IndependentStudy/CLEF25_numerical_claim_labels.json"

extract_ids_and_labels(input_fname, output_fname)
