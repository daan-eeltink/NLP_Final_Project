import json
from difflib import SequenceMatcher
from flair.models import SequenceTagger
from flair.data import Sentence
from data import load_jsonl
import csv

# Load language-specific NER models
NER_eng = SequenceTagger.load("ner-large")
NER_de = SequenceTagger.load("de-ner-large")

# Fetch the appropriate NER model based on locale
def get_ner_model(locale):
    locale = locale.lower()
    if locale.startswith("en"):
        return NER_eng
    elif locale.startswith("de"):
        return NER_de
    else:
        raise ValueError(f"No NER model defined for locale: {locale}")

# Extract NER spans from text using the appropriate model
def get_spans(text, locale):
    model = get_ner_model(locale)
    sentence = Sentence(text)
    model.predict(sentence)
    return sentence.get_spans("ner")

# Compute similarity between two spans
def span_similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

# Replace specific span in text with provided tag
def replace_span(text, span, tag):
    start = span.tokens[0].start_position
    end = span.tokens[-1].end_position
    return text[:start] + tag + text[end:]

# Replace all spans in text based on tag-mapping
def replace_with_alignment(text, spans, mapping):

    # Sort spans from right to left to ensure correct indexing
    spans_sorted = sorted(spans, key=lambda s: s.tokens[0].start_position, reverse=True)
    
    # Replace each span with its corresponding tag
    new_text = text
    for s in spans_sorted:
        if s in mapping:
            new_text = replace_span(new_text, s, mapping[s])

    return new_text

# Process JSONL file for NER replacement
def process_jsonl(input_path, output_jsonl_path, mode="train", output_csv_path=None):
    data = load_jsonl(input_path)

    # Prepare CSV writer if output path is provided
    csv_writer = None
    if output_csv_path is not None:
        csvfile = open(output_csv_path, "w", encoding="utf8", newline="")
        csv_writer = csv.writer(csvfile)
        if mode == "train":
            csv_writer.writerow(["sentence_id", "entity_tag", "entity_src", "entity_tgt"])
        else:
            csv_writer.writerow(["sentence_id", "entity_tag", "entity_category", "entity_src"])


    # Open output JSONL file for writing
    with open(output_jsonl_path, "w", encoding="utf8") as f_out:
        for obj in data:

            # Extract source text and perform NER
            src = obj["source"]
            src_loc = obj.get("source_locale", "en")
            src_spans = get_spans(src, src_loc)

            # Train mode --> also handle target text
            if mode == "train":
                threshold = 0.35

                # Extract target text and perform NER
                tgt = obj.get("target")
                tgt_loc = obj.get("target_locale", "de")
                tgt_spans = get_spans(tgt, tgt_loc)

                # Keep track of used target spans to avoid multiple alignments
                used_tgt = set()
                alignments = []

                # For each source span, find the best matching target span (if any)
                for s in src_spans:
                    best_tgt = None
                    best_score = 0.0

                    # Compare with each target span
                    for t in tgt_spans:

                        # Ensure target span is not already used
                        if t in used_tgt:
                            continue

                        # Compute similarity and check tag match
                        score = span_similarity(s.text, t.text)
                        if s.tag == t.tag or score >= threshold:
                            
                            # Update best match if score is higher
                            if score > best_score:
                                best_score = score
                                best_tgt = t
                    
                    # Record alignment if a suitable target span is found
                    if best_tgt is not None:
                        used_tgt.add(best_tgt)
                        alignments.append((s, best_tgt))

                # Keep track of tag mappings for source and target
                mapping_src = {}
                mapping_tgt = {}

                # Assign unique tags to each aligned entity pair
                for i, (s, t) in enumerate(alignments, start=1):
                    tag = f"<entity{i}>"
                    mapping_src[s] = tag
                    mapping_tgt[t] = tag

                    if csv_writer:
                        csv_writer.writerow([obj["id"], tag, s.text, t.text])

                # Replace spans in source and target texts
                new_src = replace_with_alignment(src, src_spans, mapping_src)
                new_tgt = replace_with_alignment(tgt, tgt_spans, mapping_tgt) if tgt else tgt
                
                # Update JSON object with new texts
                obj["source"] = new_src
                obj["target"] = new_tgt
        
            # Validation mode --> only handle source text
            elif mode == "val":

                # Extract entity category
                category = obj.get("entity_types", [])

                # Keep track of tag mappings for source
                mapping_src = {}

                # Assign unique tags to each source entity span
                for i, s in enumerate(src_spans, start=1):
                    tag = f"<entity{i}>"
                    mapping_src[s] = tag
                    
                    if csv_writer:
                        csv_writer.writerow([obj["id"], tag, category, s.text])

                # Replace spans in source text
                obj["source"] = replace_with_alignment(src, src_spans, mapping_src)

            # Write updated JSONL
            json.dump(obj, f_out, ensure_ascii=False)
            f_out.write("\n")

    if csv_writer:
        csvfile.close()



# Train data
process_jsonl(
    input_path=r"C:\Users\daane\Desktop\GitHub Repos\NLP_Final_Project\dataset\train\de\train.jsonl",
    output_jsonl_path=r"C:\Users\daane\Desktop\GitHub Repos\NLP_Final_Project\dataset\train\de\train_ner_replaced.jsonl",
    output_csv_path=r"C:\Users\daane\Desktop\GitHub Repos\NLP_Final_Project\dataset\train\de\train_entity_mapping.csv",
    mode="train"
)

# Validation data
process_jsonl(
    input_path=r"C:\Users\daane\Desktop\GitHub Repos\NLP_Final_Project\dataset\validation\de_DE.jsonl",
    output_jsonl_path=r"C:\Users\daane\Desktop\GitHub Repos\NLP_Final_Project\dataset\validation\de_DE_ner_replaced.jsonl",
    output_csv_path=r"C:\Users\daane\Desktop\GitHub Repos\NLP_Final_Project\dataset\validation\de_DE_entity_mapping.csv",
    mode="val"
)