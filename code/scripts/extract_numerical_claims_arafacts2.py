import pandas as pd
import re
from num2words import num2words

# Load the dataset
file_path = 'CLEF25/data/AraFacts 2.csv'  
data = pd.read_csv(file_path)

arabic_number_words = {num2words(i, lang='ar') for i in range(100)}

numeric_pattern = re.compile(r'\b[\d٠١٢٣٤٥٦٧٨٩]+\b')  # Matches Arabic and English digits
textual_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, arabic_number_words)) + r')\b')

def contains_numbers_or_words(claim):
    if numeric_pattern.search(claim) or textual_pattern.search(claim):
        return True
    return False

filtered_data = data[data['claim'].apply(lambda x: contains_numbers_or_words(str(x)))]

output_path = 'CLEF25/data/numerical_claims_arafacts2.csv'
filtered_data.to_csv(output_path, index=False)

print(f"Filtered claims saved to {output_path}")
