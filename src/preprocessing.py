"""Text preprocessing for Kazakh: tokenization, stop words, stemming, lemmatization.

Neither NLTK nor spaCy ships a Kazakh stemmer or lemmatizer, so the project uses
  * KazakhSuffixStemmer — a rule-based suffix stripper written as an NLTK StemmerI,
  * SuffixLemmatizer   — a data-driven lemmatizer that learns "strip / append" suffix
                         rules from the gold lemmas of the UD Kazakh-KTB treebank,
  * (optional) Stanza's neural Kazakh lemmatizer, if it is installed locally.
All three are evaluated against held-out UD gold lemmas in compare_normalizers().
"""
import random
import re
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd
from nltk.stem.api import StemmerI

import config

TOKEN_RE = re.compile(r"\d+(?:[.,]\d+)*(?:-\w+)?|\w+(?:-\w+)*|[^\w\s]", re.UNICODE)
SENT_RE = re.compile(r"(?<=[.!?])\s+")
KK_VOWELS = set("аәеёиоөуұүыіэюя")


# ---------------------------------------------------------------------------
# tokenization
# ---------------------------------------------------------------------------
def split_sentences(text):
    return [s for s in SENT_RE.split(text.strip()) if s]


def tokenize(text):
    """Regex word tokenizer matching KazNERD conventions (punctuation is a separate token,
    hyphenated words and numbers with suffixes such as '55-ші' stay whole)."""
    return TOKEN_RE.findall(text)


def is_word(token):
    return any(ch.isalpha() for ch in token)


# ---------------------------------------------------------------------------
# stop words
# ---------------------------------------------------------------------------
def load_stopwords():
    """Kazakh stop words from NLTK; falls back to the bundled copy if the corpus is missing."""
    try:
        from nltk.corpus import stopwords
        words = set(stopwords.words("kazakh"))
    except LookupError:
        try:
            import nltk
            nltk.download("stopwords", quiet=True)
            from nltk.corpus import stopwords
            words = set(stopwords.words("kazakh"))
        except Exception:
            words = set(Path(config.STOPWORDS_FILE).read_text(encoding="utf-8").split())
    return words


# ---------------------------------------------------------------------------
# stemming
# ---------------------------------------------------------------------------
_SUFFIXES = """
ының інің сының сінің ындағы індегі ндағы ндегі дағы дегі тағы тегі
ымыз іміз ыңыз іңіз мыз міз быз біз пыз піз ңыз ңіз
ның нің дың дің тың тің
ға ге қа ке на не
ны ні ды ді ты ті
нда нде да де та те
ндан нден нан нен дан ден тан тен
менен мен бен пен
лар лер дар дер тар тер
ым ім ың ің сы сі ы і
мын мін бын бін пын пін сың сің сыз сіз
ған ген қан кен атын етін йтын йтін ады еді йды йді
ып іп мақ мек бақ бек пақ пек уда уде уы уі
""".split()
SUFFIXES = sorted(set(_SUFFIXES), key=len, reverse=True)


class KazakhSuffixStemmer(StemmerI):
    """Greedy longest-suffix stripping, repeated up to `max_passes` times.

    Guards against over-stemming: the remaining stem must keep >= `min_stem`
    characters (>= min_stem + 1 for one-letter suffixes) and contain a vowel.
    """

    def __init__(self, min_stem=3, max_passes=3):
        self.min_stem = min_stem
        self.max_passes = max_passes

    def stem(self, token):
        word = token.lower()
        if not word.isalpha():
            return word
        for _ in range(self.max_passes):
            for suf in SUFFIXES:
                if word.endswith(suf):
                    stem = word[: -len(suf)]
                    need = self.min_stem + (1 if len(suf) == 1 else 0)
                    if len(stem) >= need and KK_VOWELS & set(stem):
                        word = stem
                        break
            else:
                break
        return word


# ---------------------------------------------------------------------------
# lemmatization
# ---------------------------------------------------------------------------
def ensure_ud_data():
    config.UD_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for split, url in config.UD_URLS.items():
        path = config.UD_DIR / Path(url).name
        if not path.exists():
            try:
                urllib.request.urlretrieve(url, path)
            except OSError as exc:
                raise FileNotFoundError(f"Cannot download UD Kazakh-KTB ({exc}); "
                                        f"place {path.name} into {config.UD_DIR}") from exc
        paths.append(path)
    return paths


def read_conllu(path):
    """Return sentences as lists of (form, lemma, upos) from a CoNLL-U file."""
    sentences, cur = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                if cur:
                    sentences.append(cur)
                    cur = []
                continue
            if line.startswith("#"):
                continue
            cols = line.split("\t")
            if "-" in cols[0] or "." in cols[0]:   # multi-word / empty nodes
                continue
            cur.append((cols[1], cols[2], cols[3]))
    if cur:
        sentences.append(cur)
    return sentences


def load_ud_split(test_share=0.2, seed=config.SEED):
    sentences = []
    for path in ensure_ud_data():
        sentences.extend(read_conllu(path))
    rng = random.Random(seed)
    rng.shuffle(sentences)
    cut = int(len(sentences) * (1 - test_share))
    return sentences[:cut], sentences[cut:]


class SuffixLemmatizer:
    """Learns (strip, append) edit rules keyed by word endings from gold (form, lemma) pairs.

    Prediction: exact-form lookup first; otherwise the rule attached to the longest
    known word ending (seen at least `min_count` times) is applied.
    """

    def __init__(self, max_context=3, min_count=2):
        self.max_context = max_context
        self.min_count = min_count
        self.lexicon = {}
        self.rules = {}

    def fit(self, pairs):
        lex = defaultdict(Counter)
        rules = defaultdict(Counter)
        for form, lemma in pairs:
            f, l = form.lower(), lemma.lower()
            if not f.isalpha():
                continue
            lex[f][l] += 1
            p = 0
            while p < min(len(f), len(l)) and f[p] == l[p]:
                p += 1
            strip, add = f[p:], l[p:]
            for k in range(max(len(strip), 1), min(len(f), len(strip) + self.max_context) + 1):
                rules[f[-k:]][(len(strip), add)] += 1
        self.lexicon = {f: c.most_common(1)[0][0] for f, c in lex.items()}
        self.rules = {k: c.most_common(1)[0] for k, c in rules.items()}
        self.max_key = max((len(k) for k in self.rules), default=0)
        return self

    def lemmatize(self, token):
        f = token.lower()
        if not f.isalpha():
            return f
        if f in self.lexicon:
            return self.lexicon[f]
        for k in range(min(self.max_key, len(f) - 1), 0, -1):
            hit = self.rules.get(f[-k:])
            if hit and hit[1] >= self.min_count:
                (n_strip, add), _ = hit
                if len(f) - n_strip >= 2:
                    return f[: len(f) - n_strip] + add
        return f


def stanza_lemmatizer():
    """Return a token->lemma function backed by Stanza's Kazakh model, or None if unavailable."""
    try:
        import logging
        import stanza
        logging.getLogger("stanza").setLevel(logging.CRITICAL)
        nlp = stanza.Pipeline("kk", processors="tokenize,lemma", tokenize_pretokenized=True,
                              download_method=None, logging_level="CRITICAL")
    except Exception:
        return None

    def lemmatize_batch(tokens):
        doc = nlp([list(tokens)])
        return [w.lemma or w.text for w in doc.sentences[0].words]
    return lemmatize_batch


_default_stemmer = KazakhSuffixStemmer()
_default_lemmatizer = None


def get_lemmatizer():
    """Train (once) the suffix lemmatizer on the UD training part and cache it."""
    global _default_lemmatizer
    if _default_lemmatizer is None:
        train, _ = load_ud_split()
        _default_lemmatizer = SuffixLemmatizer().fit((f, l) for s in train for f, l, _ in s)
    return _default_lemmatizer


def stem(token):
    return _default_stemmer.stem(token)


# ---------------------------------------------------------------------------
# comparison
# ---------------------------------------------------------------------------
def compare_normalizers(kaznerd_tokens):
    """Stemming vs lemmatization: accuracy against held-out UD gold lemmas,
    vocabulary reduction on KazNERD and speed. Returns a DataFrame."""
    train, test = load_ud_split()
    lemmatizer = SuffixLemmatizer().fit((f, l) for s in train for f, l, _ in s)
    stemmer = KazakhSuffixStemmer()
    train_forms = {f.lower() for s in train for f, _, _ in s}
    gold = [(f, l.lower()) for s in test for f, l, _ in s if f.isalpha()]

    methods = {
        "identity (lowercase)": str.lower,
        "suffix stemmer (rule-based)": stemmer.stem,
        "suffix lemmatizer (UD-trained)": lemmatizer.lemmatize,
    }
    stanza_fn = stanza_lemmatizer()
    if stanza_fn is not None:
        methods["stanza lemmatizer (neural)"] = lambda t: stanza_fn([t])[0].lower()

    words = [t for t in kaznerd_tokens if is_word(t)]
    unique_words = sorted({w.lower() for w in words})
    rows = []
    for name, fn in methods.items():
        preds = [fn(f) for f, _ in gold]
        correct = [p == l for p, (_, l) in zip(preds, gold)]
        oov = [c for c, (f, _) in zip(correct, gold) if f.lower() not in train_forms]
        t0 = time.perf_counter()
        sample = unique_words if "stanza" not in name else unique_words[:2000]
        vocab = {fn(w) for w in sample}
        secs = time.perf_counter() - t0
        rows.append({
            "method": name,
            "ud_accuracy": sum(correct) / len(correct),
            "ud_oov_accuracy": sum(oov) / max(len(oov), 1),
            "kaznerd_vocab": len(vocab) if "stanza" not in name else None,
            "vocab_reduction_pct": 100 * (1 - len(vocab) / len(sample)),
            "ms_per_1k_words": 1000 * secs / len(sample) * 1000,
        })
    df = pd.DataFrame(rows)
    df.attrs["ud_test_tokens"] = len(gold)
    df.attrs["ud_test_oov"] = sum(1 for f, _ in gold if f.lower() not in train_forms)
    return df
