import json
import re

# Paths
input_paragraphs_file = "data/full_zaman_consolidation_formatted_cleaned_final_binary.jsonl"  # Original dataset
output_filtered_trustworthy = "data/6.zaman_paragraphs_trustworthy_domain.jsonl"  # Filtered paragraphs
output_trusted_domains = "data/6.trusted_domains.txt"  # Unique trusted domains

# Trusted & suspicious rules
trusted_suffixes = {".gov", ".edu", ".org"}  
suspicious_suffixes = {".blog", ".info", ".xyz"}  

trusted_keywords = {"news", "gov", "university", "official", "research", "report"}  
suspicious_keywords = {"blog", "clickbait", "freeweb", "ads", "random"}  

def is_trusted_domain(domain):
    if not domain:
        return False  

    if any(domain.endswith(suffix) for suffix in trusted_suffixes):
        return True  

    if any(domain.endswith(suffix) for suffix in suspicious_suffixes):
        return False  

    if any(keyword in domain for keyword in trusted_keywords):
        return True 

    if any(keyword in domain for keyword in suspicious_keywords):
        return False 

    return False 

def is_trusted_url(url):
    if not url:
        return False  

    if not url.startswith("https://"):
        return False 

    if any(keyword in url for keyword in trusted_keywords):
        return True  
    if any(keyword in url for keyword in suspicious_keywords):
        return False  

    return False  

trusted_domains = set()
filtered_paragraphs = []

with open(input_paragraphs_file, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)
        
        domain = data.get("page_domain", "")
        url = data.get("page_url", "")

        if is_trusted_domain(domain) or is_trusted_url(url):
            trusted_domains.add(domain)  

            filtered_paragraphs.append({
                "paragraph_id": data["paragraph_id"],
                "text": data["paragraph"]
            })

with open(output_filtered_trustworthy, "w", encoding="utf-8") as f:
    for paragraph in filtered_paragraphs:
        f.write(json.dumps(paragraph, ensure_ascii=False) + "\n")

with open(output_trusted_domains, "w", encoding="utf-8") as f:
    for domain in sorted(trusted_domains):  
        f.write(domain + "\n")

print(f"Saved {len(filtered_paragraphs)} trusted paragraphs to {output_filtered_trustworthy}")
print(f"Saved {len(trusted_domains)} unique trusted domains to {output_trusted_domains}")
