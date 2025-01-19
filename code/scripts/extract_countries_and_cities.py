import pandas as pd
import re

# Path to the text file containing all countries and cities
locations_file = 'data/arabic_names.txt'  # Replace with the path to your file

# Load the countries and cities into a set
def load_locations_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        locations = {line.strip() for line in file if line.strip()}
    return locations

# Load the locations
arabic_locations = load_locations_from_file(locations_file)

# Load the dataset
file_path = 'data/AraFacts 2.csv'  # Replace with your dataset file path
data = pd.read_csv(file_path)

# Function to extract locations from claims
def extract_locations_from_claim(text, locations):
    matches = [location for location in locations if re.search(rf'\b{re.escape(location)}\b', text)]
    print(matches)
    return matches

# Extract locations and save the unique ones
data['extracted_locations'] = data['claim'].apply(lambda x: extract_locations_from_claim(str(x), arabic_locations))

# Create a set of all unique locations
unique_locations = set(loc for locations in data['extracted_locations'] for loc in locations)

# Save the unique locations to a text file
output_path = 'extracted_locations.txt'
with open(output_path, 'w', encoding='utf-8') as file:
    for location in sorted(unique_locations):
        file.write(location + '\n')

print(f"Unique locations extracted and saved to {output_path}")
