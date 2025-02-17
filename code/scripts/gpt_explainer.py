import json
import os
import optparse
import requests
from dotenv import load_dotenv

# Load API credentials
load_dotenv("/Users/bushrabendou/Desktop/IndependentStudy/CLEF25/code/scripts/openai.env")

def load_topics(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]

# Function to send API request
def input_chat(input_prompt, api_url, model, headers):
    json_data = {
        "model": model,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": input_prompt}]}
        ],
    }

    response = requests.post(api_url, headers=headers, json=json_data)
    print("RESPONSE:", response)  # Debugging
    try:
        response_json = response.json()
        print("RAW RESPONSE:", response_json)  # Debugging
        return response_json
    except json.JSONDecodeError:
        print("❌ ERROR: Invalid JSON response from OpenAI")
        print("Raw Response Text:", response.text)
        return None

# Classify claims using GPT
def classify_claims(input_file, output_file, error_file, api_url, model, headers, topics, num_rows):
    print("📂 Processing file:", input_file)

    # Load existing results (if script crashes, it resumes from where it left off)
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            classified_claims = json.load(f)
    else:
        classified_claims = {}

    if os.path.exists(error_file):
        with open(error_file, "r", encoding="utf-8") as f:
            errors = json.load(f)
    else:
        errors = {}

    # ✅ Read JSONL file line by line
    with open(input_file, "r", encoding="utf-8") as f:
        claims_data = [json.loads(line) for line in f]

    # ✅ Filter only claims where `factuality_label == "False"`
    filtered_claims = {
        str(claim["id"]): claim["claim"]
        for claim in claims_data if claim.get("factuality_label") == "False"
    }

    for i, (claim_id, claim_text) in enumerate(filtered_claims.items()):
        if i >= num_rows:
            break
        if i % 50 == 0 and i != 0:
            print(f"✅ Processed {i} claims...")

        # Skip already processed claims
        if claim_id in classified_claims or claim_id in errors:
            continue

        # Arabic prompt
        input_prompt = f'هذا ادعاء: "{claim_text}". قم بتعيينه إلى أحد المواضيع التالية: {", ".join(topics)}. فقط قم بإرجاع اسم الموضوع المناسب دون أي تفسير إضافي.'

        try:
            response = input_chat(input_prompt, api_url, model, headers)
            if not response or "choices" not in response:
                raise ValueError("❌ Invalid API response")

            topic = response["choices"][0]["message"]["content"].strip()
            classified_claims[claim_id] = {"claim": claim_text, "topic": topic}

        except Exception as e:
            errors[claim_id] = {"claim": claim_text, "error": str(e)}
            print(f"❌ ERROR processing claim {claim_id}: {e}")
            print(f"🚨 Problematic claim: {claim_text}")

        # ✅ Save progress after each claim
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(classified_claims, f, ensure_ascii=False, indent=4)

        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(errors, f, ensure_ascii=False, indent=4)

    print(f"✅ Classification completed. Results saved to {output_file}. Errors saved to {error_file}.")

# Main function
if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-i', '--input_file', action="store", dest="input_file", default=None, type="string", help="Input file containing claims dataset")
    parser.add_option('-o', '--output_file', action="store", dest="out_fname", default=None, type="string", help="Output file for classified claims")
    parser.add_option('-e', '--err_output_file', action="store", dest="err_out_fname", default=None, type="string", help="Error output file")
    parser.add_option('-t', '--topics_file', action="store", dest="topics_file", default=None, type="string", help="File containing topics list")
    parser.add_option('-l', '--lines', action="store", dest="num_rows", default=None, type="int", help="Number of claims to process")

    options, args = parser.parse_args()

    # Load API credentials
    model = os.getenv("OPENAI_MODEL", "gpt-4")
    openai_api_base = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Check if API key is loaded
    if not openai_api_key:
        print("❌ ERROR: API key is missing! Make sure it is set in your .env file.")
        exit(1)

    print("🛠 Using OpenAI Model:", model)
    print("🌐 API Base:", openai_api_base)
    print("🔑 Loaded API Key:", openai_api_key[:5] + "..." + openai_api_key[-5:])  # Debugging

    # Correct API URL
    api_url = f"{openai_api_base}/chat/completions"
    headers = {"Authorization": f"Bearer {openai_api_key}"}  # ✅ Correct format

    topics_file = options.topics_file or "/Users/bushrabendou/Desktop/IndependentStudy/CLEF25/data/topics_list.txt"
    topics = load_topics(topics_file)

    # Run classification
    classify_claims(options.input_file, options.out_fname, options.err_out_fname, api_url, model, headers, topics, options.num_rows)
