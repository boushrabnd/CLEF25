import pandas as pd
import re
import json

locations_file = f'data/all_country_and_city_names.txt' 
output_path = 'data/extracted_locations.txt'
file_path =  f'data/full_zaman_consolidation_formatted_cleaned_final_binary.jsonl' 

def load_locations_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        locations = {line.strip() for line in file if line.strip()}
    return locations

arabic_locations = load_locations_from_file(locations_file)

data = data = [json.loads(line) for line in open(file_path, "r", encoding="utf-8")]
df = pd.DataFrame(data)

def extract_locations_from_claim(text, locations):
    matches = [location for location in locations if re.search(rf'\b{re.escape(location)}\b', text)]
    return matches

df['extracted_locations'] = df['paragraph'].apply(lambda x: extract_locations_from_claim(str(x), arabic_locations))

unique_locations = set(loc for locations in df['extracted_locations'] for loc in locations)


with open(output_path, 'w', encoding='utf-8') as file:
    for location in sorted(unique_locations):
        file.write(location + '\n')

print(f"Unique locations extracted and saved to {output_path}")
