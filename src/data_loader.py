"""Loading KazNERD (CoNLL / IOB2) into memory-efficient pandas structures.

Steps:
  * stream the raw CoNLL files line by line (no full-file read),
  * build one token-level DataFrame with compact dtypes (category / int32 / int16),
  * persist it as Parquet (primary) and CSV (for the format benchmark),
  * regroup tokens into sentences for the models.
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# raw files
# ---------------------------------------------------------------------------
def ensure_raw_data():
    """Download any missing KazNERD split from GitHub. Returns the list of local paths."""
    paths = []
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    for split in config.SPLITS:
        path = config.RAW_DIR / f"IOB2_{split}.txt"
        if not path.exists():
            url = config.KAZNERD_URL.format(split=split)
            print(f"Downloading {url} ...")
            try:
                urllib.request.urlretrieve(url, path)
            except OSError as exc:  # network errors, HTTP errors (URLError is an OSError)
                raise FileNotFoundError(
                    f"{path.name} is missing and could not be downloaded ({exc}). "
                    f"Download it manually from https://github.com/IS2AI/KazNERD into {config.RAW_DIR}"
                ) from exc
        paths.append(path)
    return paths


def read_conll(path):
    """Yield (tokens, tags) for every sentence of a CoNLL file (blank-line separated)."""
    tokens, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                if tokens:
                    yield tokens, tags
                    tokens, tags = [], []
                continue
            token, _, tag = line.rpartition(" ")
            tokens.append(token)
            tags.append(tag)
    if tokens:
        yield tokens, tags


# ---------------------------------------------------------------------------
# token DataFrame
# ---------------------------------------------------------------------------
def build_token_frame(optimize=True):
    """Parse all splits into one DataFrame: sentence_id, split, position, token, tag."""
    ensure_raw_data()
    sent_ids, splits, positions, tokens, tags = [], [], [], [], []
    sid = 0
    for split in config.SPLITS:
        for toks, tgs in read_conll(config.RAW_DIR / f"IOB2_{split}.txt"):
            n = len(toks)
            sent_ids.extend([sid] * n)
            splits.extend([split] * n)
            positions.extend(range(n))
            tokens.extend(toks)
            tags.extend(tgs)
            sid += 1
    df = pd.DataFrame({"sentence_id": sent_ids, "split": splits, "position": positions,
                       "token": tokens, "tag": tags})
    return optimize_dtypes(df) if optimize else df


def optimize_dtypes(df):
    """Downcast integers and turn low-cardinality strings into categoricals."""
    df = df.copy()
    df["sentence_id"] = df["sentence_id"].astype(np.int32)
    df["position"] = df["position"].astype(np.int16)
    df["split"] = df["split"].astype("category")
    df["tag"] = df["tag"].astype("category")
    df["token"] = df["token"].astype("string[pyarrow]")
    return df


def memory_mb(df):
    return df.memory_usage(deep=True).sum() / 2 ** 20


def save_tokens(df):
    """Write Parquet (columnar, compressed) and CSV copies of the token table."""
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.TOKENS_PARQUET, index=False, compression="snappy")
    df.to_csv(config.TOKENS_CSV, index=False)


def load_tokens(fmt="parquet", columns=None):
    """Load the token table; raises FileNotFoundError with a hint if step 1 was not run."""
    path = config.TOKENS_PARQUET if fmt == "parquet" else config.TOKENS_CSV
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found — run menu step 1 (Prepare data) first.")
    if fmt == "parquet":
        return pd.read_parquet(path, columns=columns)
    df = pd.read_csv(path, usecols=columns, keep_default_na=False)
    return df


def tag_counts_chunked(path=None, chunksize=config.CSV_CHUNKSIZE):
    """Count tags in the CSV without loading it whole — demonstrates chunked processing."""
    path = path or config.TOKENS_CSV
    total = pd.Series(dtype="int64")
    n_chunks = 0
    for chunk in pd.read_csv(path, usecols=["tag"], chunksize=chunksize,
                             dtype={"tag": "category"}):
        total = total.add(chunk["tag"].value_counts(), fill_value=0)
        n_chunks += 1
    return total.astype("int64").sort_values(ascending=False), n_chunks


# ---------------------------------------------------------------------------
# sentences
# ---------------------------------------------------------------------------
def to_sentences(df):
    """Group the (sorted) token frame into {split: [(sentence_id, tokens, tags), ...]}."""
    result = {}
    for split in config.SPLITS:
        part = df[df["split"] == split]
        ids = part["sentence_id"].to_numpy()
        toks = part["token"].astype(str).to_numpy()
        tags = part["tag"].astype(str).to_numpy()
        bounds = np.flatnonzero(np.diff(ids)) + 1
        starts = np.concatenate(([0], bounds))
        ends = np.concatenate((bounds, [len(ids)]))
        result[split] = [(int(ids[s]), list(toks[s:e]), list(tags[s:e]))
                         for s, e in zip(starts, ends)]
    return result


def load_sentences():
    return to_sentences(load_tokens("parquet"))
