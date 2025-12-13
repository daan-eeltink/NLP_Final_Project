# GOOGLE_API_KEY = "AIzaSyAGS3c7UV93cQsMSzAg_O7py22_Nd2tZ5I"
# prompt = f"""
#     You are a professional English-to-German translator specializing in proper nouns and terminology.
    
#     Task: Translate the following 'Entity' from English to German.
    
#     Context Rules:
#     1. Use the 'Category' to decide if the name should be translated or kept in English.
#     2. If it is a Movie/Book/Such title that keeps its English name in Germany, keep it in English. If there is a known German title, use that.
#     3. If it has a specific German localized name (e.g., 'Munich' -> 'München'), use that.
#     4. Output ONLY the translated term. No explanations.

#     Category: {category}
#     Entity: {entity}
    
#     German Translation:
#     """

import pandas as pd
import google.generativeai as genai
import ast
import time
import os
import json
from tqdm import tqdm

# ==========================================
# 1. SETUP & CONFIGURATION
# ==========================================
GOOGLE_API_KEY = ""
genai.configure(api_key=GOOGLE_API_KEY)

MODEL_VERSION = 'gemini-2.5-pro' # Your working version

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

model = genai.GenerativeModel(MODEL_VERSION, 
                              safety_settings=safety_settings,
                              generation_config={"temperature": 0.3, "response_mime_type": "application/json"})

script_dir = os.path.dirname(os.path.abspath(__file__))
INPUT_FILE = os.path.join(script_dir, 'de_DE_entity_mapping.csv')
OUTPUT_FILE = os.path.join(script_dir, 'translated_entity.csv')

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def parse_category_context(cat_str):
    try:
        cats = ast.literal_eval(cat_str)
        if isinstance(cats, list):
            return ", ".join(cats)
        return str(cats)
    except:
        return str(cat_str)

def translate_batch(batch_df):
    """
    Sends the entire dataset in one go.
    """
    items_to_translate = []
    for idx, row in batch_df.iterrows():
        items_to_translate.append({
            "id": idx,
            "entity": row['entity_src'],
            "category": row['context_str']
        })

    prompt = f"""
    Role: Professional German Localization Linguist.
    Task: Translate the following list of entities into German.
    
    Context Rules:
    1. Use the 'Category' to decide if the name should be translated or kept in English.
    2. If it is a Movie/Book/Such title that keeps its English name in Germany, keep it in English. If there is a known German title, use that.
    3. If it has a specific German localized name (e.g., 'Munich' -> 'München'), use that.
    4. Output ONLY the translated term. No explanations.
    
    INPUT DATA:
    {json.dumps(items_to_translate)}

    OUTPUT FORMAT:
    Return a JSON list of objects. Each object must have:
    - "id": (The integer ID from input)
    - "german_translation": (The translated string)
    """
    
    print("Sending request to Gemini... (This might take 30-60 seconds to generate)")
    
    retries = 3
    for attempt in range(retries):
        try:
            response = model.generate_content(prompt)
            translations = json.loads(response.text)
            return {item['id']: item['german_translation'] for item in translations}

        except Exception as e:
            print(f"\n[Error - Attempt {attempt+1}] {e}")
            print("Retrying in 60 seconds...")
            time.sleep(60)
            
    print("Batch failed. Keeping originals.")
    return {row.name: row['entity_src'] for _, row in batch_df.iterrows()}

# ==========================================
# 3. MAIN FLOW
# ==========================================
def run_one_shot_translation():
    print(f"--- Starting ONE-SHOT Translation ({MODEL_VERSION}) ---")
    
    if not os.path.exists(INPUT_FILE):
        print(f"ERROR: Could not find {INPUT_FILE}")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_FILE)
    df['context_str'] = df['entity_category'].apply(parse_category_context)
    
    # 2. Optimization
    unique_pairs = df[['entity_src', 'context_str']].drop_duplicates()
    unique_pairs = unique_pairs.reset_index(drop=True)
    
    print(f"Total unique entities: {len(unique_pairs)}")
    
    # 3. PROCESS EVERYTHING IN ONE BATCH
    # Since you have ~371 items, a batch size of 500 covers 100% of your data.
    BATCH_SIZE = 500 
    results_map = {}
    
    for i in tqdm(range(0, len(unique_pairs), BATCH_SIZE)):
        batch = unique_pairs.iloc[i : i + BATCH_SIZE]
        
        # Call API (Only happens once)
        batch_translations = translate_batch(batch)
        results_map.update(batch_translations)

    # 4. Map Results
    unique_pairs['entity_translated'] = unique_pairs.index.map(results_map)
    
    # 5. Merge & Save
    df_final = pd.merge(df, unique_pairs, on=['entity_src', 'context_str'], how='left')
    df_final.drop(columns=['context_str'], inplace=True)
    
    df_final.to_csv(OUTPUT_FILE, index=False)
    print("-" * 30)
    print(f"DONE! Saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    run_one_shot_translation()