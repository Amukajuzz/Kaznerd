"""Interactive prediction: the user types Kazakh text and gets the recognised entities."""
from pathlib import Path

import pandas as pd

import config
from src import preprocessing
from src.models import load_model
from src.spans import get_spans, span_text


def available_models():
    """Locally trained models plus the fine-tuned transformer, if its folder exists."""
    names = [p.stem for p in sorted(config.MODELS_DIR.glob("*.joblib"))]
    if (config.MODELS_DIR / "transformer" / "config.json").exists():
        names.append("transformer")
    return names


def best_model_name():
    """Model with the highest test F1 in model_comparison.csv that is available on disk."""
    path = config.RESULTS_DIR / "model_comparison.csv"
    avail = available_models()
    if path.exists():
        table = pd.read_csv(path).sort_values("f1_micro", ascending=False)
        for name in table["model"]:
            if name in avail:
                return name
    if not avail:
        raise FileNotFoundError("No trained model found — run menu step 4 first.")
    return "crf" if "crf" in avail else avail[0]


def get_tagger(name):
    if name == "transformer":
        from src.transformer_ner import TransformerTagger
        return TransformerTagger(config.MODELS_DIR / "transformer")
    return load_model(name)


def predict_text(text, tagger):
    """Split into sentences, tokenize, tag. Returns list of {sentence, type, text, start, end}."""
    entities = []
    for k, sentence in enumerate(preprocessing.split_sentences(text)):
        tokens = preprocessing.tokenize(sentence)
        if not tokens:
            continue
        tags = tagger.predict([tokens])[0]
        for etype, s, e in get_spans(tags):
            entities.append({"sentence": k, "type": etype, "text": span_text(tokens, s, e),
                             "start": s, "end": e, "description": config.ENTITY_TYPES.get(etype, "")})
    return entities


def highlight(text, tagger):
    """Inline view: tokens of each entity wrapped as [text](TYPE)."""
    out = []
    for sentence in preprocessing.split_sentences(text):
        tokens = preprocessing.tokenize(sentence)
        tags = tagger.predict([tokens])[0] if tokens else []
        spans = {s: (etype, e) for etype, s, e in get_spans(tags)}
        i, parts = 0, []
        while i < len(tokens):
            if i in spans:
                etype, e = spans[i]
                parts.append(f"[{span_text(tokens, i, e)}]({etype})")
                i = e
            else:
                parts.append(tokens[i])
                i += 1
        out.append(" ".join(parts))
    return " ".join(out)


def save_session(rows, path=None):
    """Append the session's results to a CSV file (persistent storage of user queries)."""
    path = Path(path or config.RESULTS_DIR / "user_predictions.csv")
    df = pd.DataFrame(rows)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)
    return path
