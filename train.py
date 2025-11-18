### Training Script for NLP Final Project

from data import load_pairs_from_jsonl, load_sp_model, tokenize_text_pairs, train_sentencepiece_model

path = r"C:\Users\daane\Desktop\GitHub Repos\NLP_Final_Project\dataset\train\de\train.jsonl"

text_pairs = load_pairs_from_jsonl(path)
src_text = [s for s,t in text_pairs]
tgt_text = [t for s,t in text_pairs]

train_sentencepiece_model(src_text, "sp_en")
train_sentencepiece_model(tgt_text, "sp_de")

token_pairs = tokenize_text_pairs(text_pairs, load_sp_model("sp_en"), load_sp_model("sp_de"))

print(token_pairs)