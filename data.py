import os
import json
import sentencepiece as spm



# Load JSONL file - extracts text pairs or full objects
def load_jsonl(jsonl_path: str, only_text_pairs = False):
    
    # Initialze return object
    data = []

    # Open the eval file and iterate over JSON objects
    with open(jsonl_path, "r", encoding="utf-8") as file:
        for i, line in enumerate(file, start=1):

            # Skip empty lines
            line = line.strip()
            if not line:
                continue

            # Read the JSON object and extract eval data
            try:
                obj = json.loads(line)

                # If text pairs are requested, extract them
                if only_text_pairs:

                    # Ensure source and target keys exist
                    if 'source' in obj and 'target' in obj:
                        data.append((obj['source'], obj['target']))
                
                    else:
                        print(f"Missing 'source' or 'target' at line {i}")

                # Otherwise, append the full JSON object
                else:
                    data.append(obj)
            
            # Handle exceptions
            except Exception as e:
                print(f"Error at line {i}: {e}")
    
    return data



# Train SentencePiece model on given set of text pairs
def train_sentencepiece_model(text_list, model_prefix, vocab_size=1000):

    # Write all texts to a file (required input for SentencePiece)
    with open(f"{model_prefix}.txt", "w", encoding="utf-8") as file:
        for text in text_list:
            file.write(text.lower().strip() + "\n")
    
    # Learn the optimal tokens using SentencePiece model
    spm.SentencePieceTrainer.train(
        input=f"{model_prefix}.txt",
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )

    # Remove the temporary text file
    os.remove(f"{model_prefix}.txt")



# Load a SentencePiece model
def load_sp_model(model_prefix):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp



# Use SentencePiece model to tokenize text pairs
def tokenize_text_pairs(text_pairs, sp_src, sp_tgt):
    
    # Initialize return object
    token_pairs = []

    # Get BOS/EOS token IDs
    bos = sp_tgt.bos_id()
    eos = sp_tgt.eos_id()

    # Iterate over text_pairs
    for src, tgt in text_pairs:

        # Use SentencePiece models to tokenize source or target text
        src_toks = [bos] + sp_src.encode(src.lower(), out_type=int) + [eos]
        tgt_toks = [bos] + sp_tgt.encode(tgt.lower(), out_type=int) + [eos]

        # Append token pair to return object
        token_pairs.append((src_toks, tgt_toks))

    return token_pairs
