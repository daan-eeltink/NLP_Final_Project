import json
from pathlib import Path
import sentencepiece as spm


# Function that parses JSONL file and returns list of text pairs
def load_pairs_from_jsonl(jsonl_path: str):
    """
    Returns list of tuples: (source_text, target_text)
    """

    # Initialze return object
    text_pairs = []

    # Open the train file and iterate over JSON objects
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, start=1):

            # Skip empty lines
            if not line.strip():
                continue

            # Read the JSON object and extract text pair
            try:
                obj = json.loads(line)

                # Ensure source and target keys exist
                if 'source' in obj and 'target' in obj:
                    text_pairs.append((obj['source'], obj['target']))
                
                else:
                    print(f"Missing 'source' or 'target' at line {i}")
            
            # Handle exceptions
            except Exception as e:
                print(f"Error at line {i}: {e}")

    return text_pairs



path = r"C:\Users\daane\Desktop\GitHub Repos\NLP_Final_Project\dataset\train\de\train.jsonl"
print(load_pairs_from_jsonl(path))