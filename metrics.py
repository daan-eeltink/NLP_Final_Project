from comet import download_model, load_from_checkpoint



def compute_m_eta(predictions, entity_mentions):
    
    # Initialize counters
    correct_mentions = 0
    total_mentions = len(entity_mentions)

    # Check correctness of each entity_mention
    for pred, mention in zip(predictions, entity_mentions):
        if mention.lower() in pred.lower():
            correct_mentions += 1

    # Compute percentage of correctly predicted entity mentions
    score = correct_mentions / total_mentions if total_mentions > 0 else 0.0

    return float(score)



def compute_comet(sources, predictions, references):
    
    # Load COMET model
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    # Prepare input data for COMET
    comet_inputs = [
        {"src": s, "mt": p, "ref": r}
        for s, p, r in zip(sources, predictions, references)
    ]

    # Get COMET score
    score = model.predict(comet_inputs, gpus=0)["mean"]

    return float(score)



def harmonic_mean(a, b):
    
    # Handle edge case where either metric is zero
    if a == 0 or b == 0:
        return 0.0
    
    # Compute the harmonic mean
    harmonic_mean = 2 * (a * b) / (a + b)

    return float(harmonic_mean)