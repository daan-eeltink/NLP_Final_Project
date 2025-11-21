from data import load_jsonl
from metrics import compute_comet, compute_m_eta, harmonic_mean



def run_validation(eval_path, model):

    # Load eval dataset
    eval_data = load_jsonl(eval_path)

    # Initialize lists to track source, target, entities, and predictions
    source_texts = []
    target_texts = []
    entity_mentions = []
    predictions = []

    print(f"Validating on {len(eval_data)} samples...")

    # Iterate over eval samples
    for i, sample in enumerate(eval_data, start=1):
        print(f"Evaluating sample {i}/{len(eval_data)}", end="\r")

        source = sample["source"]
        target = sample["targets"][0]["translation"]
        mention = sample["targets"][0]["mention"]

        # Generate prediction
        pred = model.translate(source)

        source_texts.append(source)
        target_texts.append(target)
        entity_mentions.append(mention)
        predictions.append(pred)

    # Compute M-ETA score
    m_eta = compute_m_eta(predictions, entity_mentions)
    print(f"M-ETA: {m_eta:.4f}")

    # Compute COMET score
    comet_score = compute_comet(source_texts, predictions, target_texts)
    print(f"COMET: {comet_score:.4f}")

    # Compute harmonic mean of M-ETA and COMET
    final_score = harmonic_mean(comet_score, m_eta)
    print(f"FINAL SCORE (harmonic mean): {final_score:.4f}")