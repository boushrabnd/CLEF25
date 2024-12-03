import json
import random

input_file = '/Users/bushrabendou/Desktop/IndependentStudy/CLEF25/data/non_numerical_claims_Arafacts.json'
output_file = '/Users/bushrabendou/Desktop/IndependentStudy/CLEF25/data/sampled_non_numerical_claims.json'

def contains_number(claim):
    return any(char.isdigit() for char in claim)

with open(input_file, 'r', encoding='utf-8') as f:
    data = [json.loads(line) for line in f]

# Filter claims that do not contain numbers
non_numerical_claims = [item for item in data if not contains_number(item['claim'])]

random.seed(42)
sampled_claims = random.sample(non_numerical_claims, min(len(non_numerical_claims), 2500))

# Update maj_label to "non-numerical" for all sampled claims
for claim in sampled_claims:
    claim['maj_label'] = 'non_numerical'

# Save the sampled claims to a new JSON file
with open(output_file, 'w', encoding='utf-8') as f:
    for claim in sampled_claims:
        f.write(json.dumps(claim, ensure_ascii=False) + '\n')

print(f"Sampled claims saved to {output_file}")
