import json

input_dataset = "data/full_zaman_consolidation_formatted_cleaned_final_binary.jsonl"
avg_length_file = "data/average_claim_length.txt"
output_filtered_paragraphs = "data/filtered_paragraphs_by_length.jsonl"

def load_average_length(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return float(f.read().strip())

def filter_paragraphs_by_length(input_file, output_file, max_length):
    filtered_paragraphs = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            paragraph_length = len(data["paragraph"].split())  

            if paragraph_length <= max_length:  
                filtered_paragraphs.append({
                    "paragraph_id": data["paragraph_id"],
                    "text": data["paragraph"]
                })

    with open(output_file, "w", encoding="utf-8") as f:
        for paragraph in filtered_paragraphs:
            f.write(json.dumps(paragraph, ensure_ascii=False) + "\n")

    print(f"Filtered {len(filtered_paragraphs)} paragraphs and saved to {output_file}")

# Run 
average_length = load_average_length(avg_length_file)
filter_paragraphs_by_length(input_dataset, output_filtered_paragraphs, average_length)
