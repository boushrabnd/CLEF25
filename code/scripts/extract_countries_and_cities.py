import pandas as pd
import re

locations_file = 'data/arabic_names.txt' 

def load_locations_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        locations = {line.strip() for line in file if line.strip()}
    return locations

arabic_locations = load_locations_from_file(locations_file)

file_path = 'data/AraFacts 2.csv' 
data = pd.read_csv(file_path)

def extract_locations_from_claim(text, locations):
    matches = [location for location in locations if re.search(rf'\b{re.escape(location)}\b', text)]
    print(matches)
    return matches

data['extracted_locations'] = data['claim'].apply(lambda x: extract_locations_from_claim(str(x), arabic_locations))

unique_locations = set(loc for locations in data['extracted_locations'] for loc in locations)

output_path = 'data/extracted_locations.txt'
with open(output_path, 'w', encoding='utf-8') as file:
    for location in sorted(unique_locations):
        file.write(location + '\n')

print(f"Unique locations extracted and saved to {output_path}")
