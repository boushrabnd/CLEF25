import json
import numpy as np

def extract_false_claims(input_file, output_file, avg_length_file):
    false_claims = {}

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("label") == "false":  
                false_claims[data["paragraph_id"]] = data["paragraph"]  

    # Save extracted false claims
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(false_claims, f, ensure_ascii=False, indent=4)

    # Calculate average claim length (word count)
    claim_lengths = [len(claim.split()) for claim in false_claims.values()]
    average_length = np.mean(claim_lengths) if claim_lengths else 0

    # Save average length to a file
    with open(avg_length_file, "w", encoding="utf-8") as f:
        f.write(str(average_length))

    print(f"Extracted {len(false_claims)} false claims and saved to {output_file}")
    print(f"Average claim length: {average_length:.2f} words (saved to {avg_length_file})")

input_dataset = "data/full_zaman_consolidation_formatted_cleaned_final_binary.jsonl"
output_false_claims = "data/false_claims.json"
output_avg_length = "data/average_claim_length.txt"

extract_false_claims(input_dataset, output_false_claims, output_avg_length)
