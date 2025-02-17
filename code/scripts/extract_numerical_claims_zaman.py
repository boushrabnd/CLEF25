import json
import re
from num2words import num2words

input_paragraphs_file = "data/filtered_paragraphs_by_length.jsonl"
output_filtered_numerical = "data/filtered_paragraphs_with_numerical_claims.jsonl"

arabic_number_words = {num2words(i, lang='ar') for i in range(100)}

numeric_pattern = re.compile(r'\b[\d٠١٢٣٤٥٦٧٨٩]+\b')  # Matches Arabic & English digits
textual_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, arabic_number_words)) + r')\b')  # Matches textual numbers

def contains_numbers_or_words(text):
    return bool(numeric_pattern.search(text) or textual_pattern.search(text))

filtered_paragraphs = []

with open(input_paragraphs_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        if contains_numbers_or_words(data["text"]):
            filtered_paragraphs.append(data)

with open(output_filtered_numerical, "w", encoding="utf-8") as f:
    for paragraph in filtered_paragraphs:
        f.write(json.dumps(paragraph, ensure_ascii=False) + "\n")

print(f"Filtered {len(filtered_paragraphs)} paragraphs with numerical claims and saved to {output_filtered_numerical}")
