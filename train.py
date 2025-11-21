import tensorflow as tf
from model import Seq2Seq
from val import run_validation
from data import (
    load_jsonl,
    load_sp_model,
    tokenize_text_pairs,
    train_sentencepiece_model
)



def train_model(
    train_path,
    val_path,
    epochs = 20,
    sp_src_prefix = "sp_en",
    sp_tgt_prefix = "sp_de",
    model_export_dir = "checkpoints/seq2seq",
):

    print("Loading training data")
    text_pairs = load_jsonl(train_path, only_text_pairs=True)
    src_text = [src for src, _ in text_pairs]
    tgt_text = [tgt for _, tgt in text_pairs]

    print("Training SentencePiece models")
    train_sentencepiece_model(src_text, sp_src_prefix)
    train_sentencepiece_model(tgt_text, sp_tgt_prefix)

    print("Loading SentencePiece models")
    sp_src = load_sp_model(sp_src_prefix)
    sp_tgt = load_sp_model(sp_tgt_prefix)

    print("Tokenizing training data")
    token_pairs = tokenize_text_pairs(text_pairs, sp_src, sp_tgt)

    print("Initializing Seq2Seq model")
    model = Seq2Seq(
        src_tokenizer=sp_src,
        tgt_tokenizer=sp_tgt,
    )

    print("Training Seq2Seq model")
    model.train(
        token_pairs,
        epochs=epochs,
        val_path=val_path,
    )

    print("Saving trained model")
    model.export(model_export_dir, prefix="seq2seq")
    print(f"\tModel saved to: {model_export_dir}")

    print("Running evaluation")
    run_validation(val_path, model)

    print("Training complete!")
    print("Example translation:")
    print(model.translate("Hello, how are you?"))

    return model

def load_and_evaluate(
    val_path,
    sp_src_prefix = "sp_en",
    sp_tgt_prefix = "sp_de",
    model_export_dir = "checkpoints/seq2seq",
):

    print("Loading SentencePiece tokenizers")
    sp_src = load_sp_model(sp_src_prefix)
    sp_tgt = load_sp_model(sp_tgt_prefix)

    print("Initializing Seq2Seq model")
    model = Seq2Seq(
        src_tokenizer=sp_src,
        tgt_tokenizer=sp_tgt,
    )

    # Dummy inputs to initialize weights
    dummy_src = tf.constant([[1, 2, 3]], dtype=tf.int32)
    dummy_tgt = tf.constant([[1, 2, 3]], dtype=tf.int32)

    # Call encoder & decoder once to build internal weights
    _, h, c = model.encoder(dummy_src)
    _ = model.decoder(dummy_tgt, (h, c))

    print("Loading saved model weights")
    model.load(model_export_dir, prefix="seq2seq")

    print("Running validation")
    run_validation(val_path, model)



def main():
    train_path = r"C:\Users\daane\Desktop\GitHub Repos\NLP_Final_Project\dataset\train\de\train.jsonl"
    val_path = r"C:\Users\daane\Desktop\GitHub Repos\NLP_Final_Project\dataset\validation\de_DE.jsonl"
    model_export_dir = r"C:\Users\daane\Desktop\GitHub Repos\NLP_Final_Project\checkpoints\seq2seq"

    # train_model(
    #     train_path=train_path,
    #     val_path=val_path,
    #     epochs=200,
    #     sp_src_prefix="sp_en",
    #     sp_tgt_prefix="sp_de",
    #     model_export_dir=model_export_dir
    # )

    load_and_evaluate(
        val_path=val_path,
        sp_src_prefix="sp_en",
        sp_tgt_prefix="sp_de",
        model_export_dir=model_export_dir
    )

if __name__ == "__main__":
    main()
