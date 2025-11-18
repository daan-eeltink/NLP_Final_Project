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

# Train SentencePiece model on given set of text pairs
def train_sentencepiece_model(texts, model_prefix, vocab_size=1000):
    """
    Train a SentencePiece model on given list of texts
    """
    
    # Write all text pairs to a file (required input for SentencePiece)
    tmp_file = Path(model_prefix + "_tmp.txt")
    with open(tmp_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.lower().strip() + "\n")
    
    # Learn the optimal tokens using SentencePiece model
    spm.SentencePieceTrainer.train(
        input=str(tmp_file),
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type="unigram",
        pad_id=0, unk_id=1, bos_id=2, eos_id=3,
    )

    # Clean up the temporary file
    tmp_file.unlink()

# Load a SentencePiece model
def load_sp_model(model_prefix):
    sp = spm.SentencePieceProcessor()
    sp.load(f"{model_prefix}.model")
    return sp

# Use SentencePiece model to tokenize text pairs
def tokenize_text_pairs(text_pairs, sp_src, sp_tgt, max_src_len=128, max_tgt_len=128):
    
    # Initialize return objects
    src_ids, tgt_ids = [], []

    # Iterate over each text pair
    for src, tgt in text_pairs:

        # Use correct SentencePiece model to encode source or target text
        src_toks = [2] + sp_src.encode(src.lower(), out_type=int)[:max_src_len-2] + [3]
        tgt_toks = [2] + sp_tgt.encode(tgt.lower(), out_type=int)[:max_tgt_len-2] + [3]
        src_ids.append(src_toks)
        tgt_ids.append(tgt_toks)

    return src_ids, tgt_ids