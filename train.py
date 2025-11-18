from data import load_train_jsonl, load_sp_model, tokenize_text_pairs, train_sentencepiece_model
from model import Seq2Seq

path = r"C:\Users\daane\Desktop\GitHub Repos\NLP_Final_Project\dataset\train\de\train.jsonl"

text_pairs = load_train_jsonl(path)
src_text = [s for s, _ in text_pairs]
tgt_text = [t for _ ,t in text_pairs]

train_sentencepiece_model(src_text, "sp_en")
train_sentencepiece_model(tgt_text, "sp_de")

sp_src = load_sp_model("sp_en")
sp_tgt = load_sp_model("sp_de")

token_pairs = tokenize_text_pairs(text_pairs, sp_src, sp_tgt)

model = Seq2Seq(
    src_vocab_size=sp_src.get_piece_size(),
    tgt_vocab_size=sp_tgt.get_piece_size(),
    embedding_dim=256,
    hidden_size=512,
    src_tokenizer=sp_src,
    tgt_tokenizer=sp_tgt,
    bos_id=sp_tgt.bos_id(),
    eos_id=sp_tgt.eos_id(),
    max_decode_len=100
)

model.train(
    token_pairs,
    batch_size=32,
    epochs=20,
    pad_id=sp_tgt.pad_id(),
    val_path=r"C:\Users\daane\Desktop\GitHub Repos\NLP_Final_Project\dataset\validation\de_DE.jsonl"
)

print(model.translate("Hello, how are you?"))
