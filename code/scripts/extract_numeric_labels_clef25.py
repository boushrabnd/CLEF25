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

        # Extract and process "annotations" if it exists
        annotations = item["annotations"]
        claim = item["data"]['claim']
        extracted_data["claim"] = claim
        if annotations:
            labels = []
            # Retrieve up to three labels from annotations with the specified path
            for annot in annotations:
                label = annot["result"][0]["value"]["choices"][0]
                if label == "عددية":
                    label = "numerical"
                elif label == "غير عددية":
                    label = "non_numerical"
                else:
                    label = "other"

                labels.append(label.strip())
                # Append the extracted data to the results list
            print(labels)
            extracted_data["labels"] = labels
            results.append(extracted_data)

    # Write the results to the output JSON file
    with open(output_file, 'w') as f:
        for r in results:
            maj = Counter(r["labels"]).most_common(1)[0][0]
            if "numerical" in r["labels"] and "other" in r["labels"] and "non_numerical" in r["labels"]:
                r["maj_label"] = "no_maj_label"
            else:
                r["maj_label"] = maj
            r['full_agmnt'] = "yes" if (all(x == r["labels"][0] for x in r["labels"])) else "no"
            f.write(json.dumps(r,ensure_ascii=False))
            f.write("\n")


# Usage example
input_fname = "/Users/bushrabendou/Desktop/IndependentStudy/export_105552_project-105552-at-2024-11-28-08-56-3aecdd3d.json"
output_fname = "/Users/bushrabendou/Desktop/IndependentStudy/export_105552_project-105552-at-2024-11-28-08-56-3aecdd3d_labels.json"

extract_ids_and_labels(input_fname, output_fname)
