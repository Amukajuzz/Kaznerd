"""Feature extraction: token representations (BoW, TF-IDF, Word2Vec) and CRF features."""
import re
from functools import lru_cache

import numpy as np
from scipy import sparse
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import config
from src.preprocessing import stem as _stem

CONTEXT = (-2, -1, 1, 2)


@lru_cache(maxsize=500_000)
def cached_stem(token):
    return _stem(token)


def word_shape(token):
    """'Астана' -> 'Xx', '2024' -> 'd', '55-ші' -> 'd-x' (repeated classes collapsed)."""
    shape = re.sub(r"[^\W\d_]", lambda m: "X" if m.group().isupper() else "x", token)
    shape = re.sub(r"\d", "d", shape)
    return re.sub(r"(.)\1+", r"\1", shape)


# ---------------------------------------------------------------------------
# CRF features
# ---------------------------------------------------------------------------
def crf_token_features(tokens, i):
    w = tokens[i]
    lw = w.lower()
    f = {
        "bias": 1.0, "w": lw, "stem": cached_stem(w), "suf2": lw[-2:], "suf3": lw[-3:],
        "suf4": lw[-4:], "pre3": lw[:3], "shape": word_shape(w),
        "title": w.istitle(), "upper": w.isupper(), "digit": w.isdigit(),
        "hasdigit": any(c.isdigit() for c in w), "hyphen": "-" in w,
    }
    for off in CONTEXT:
        j = i + off
        if 0 <= j < len(tokens):
            o = tokens[j]
            f[f"{off}:w"] = o.lower()
            f[f"{off}:stem"] = cached_stem(o)
            f[f"{off}:suf3"] = o.lower()[-3:]
            f[f"{off}:title"] = o.istitle()
            f[f"{off}:shape"] = word_shape(o)
        else:
            f[f"{off}:pad"] = "BOS" if j < 0 else "EOS"
    # bigram feature around the token
    if i > 0:
        f["-1:w|w"] = tokens[i - 1].lower() + "|" + lw
    if i < len(tokens) - 1:
        f["w|+1:w"] = lw + "|" + tokens[i + 1].lower()
    return f


def crf_sentence_features(tokens):
    return [crf_token_features(tokens, i) for i in range(len(tokens))]


# ---------------------------------------------------------------------------
# window representations for the token-classification (SGD) taggers
# ---------------------------------------------------------------------------
def _ortho(tokens, i):
    """Orthographic block shared by all three representations (keeps the comparison fair)."""
    w = tokens[i]
    return {"title": w.istitle(), "upper": w.isupper(), "digit": w.isdigit(),
            "hasdigit": any(c.isdigit() for c in w), "punct": not any(c.isalnum() for c in w),
            "bos": i == 0, "eos": i == len(tokens) - 1, "prev_title": i > 0 and tokens[i - 1].istitle()}


def _context_words(low, i):
    return [low[i + o] for o in CONTEXT if 0 <= i + o < len(low)]


class WindowVectorizer:
    """Turns sentences into one feature row per token.

    kind = 'bow'      — one-hot current word + bag of context words (counts)
           'tfidf'    — char 2-4-gram TF-IDF of the current word + word TF-IDF of context
           'word2vec' — concatenated Word2Vec vectors of positions -2..+2 (trained on train)
    """

    def __init__(self, kind):
        assert kind in ("bow", "tfidf", "word2vec")
        self.kind = kind
        self.ortho = DictVectorizer()

    # -- helpers --
    @staticmethod
    def _flatten(sentences):
        rows_ortho, cur, ctx, windows = [], [], [], []
        for tokens in sentences:
            low = [t.lower() for t in tokens]
            for i in range(len(tokens)):
                rows_ortho.append(_ortho(tokens, i))
                cur.append(low[i])
                ctx.append(_context_words(low, i))
                windows.append([low[i + o] if 0 <= i + o < len(low) else None for o in (-2, -1, 0, 1, 2)])
        return rows_ortho, cur, ctx, windows

    def fit_transform(self, sentences):
        rows_ortho, cur, ctx, windows = self._flatten(sentences)
        o = self.ortho.fit_transform(rows_ortho)
        if self.kind == "bow":
            self.vec = DictVectorizer()
            x = self.vec.fit_transform(self._bow_dicts(cur, ctx))
        elif self.kind == "tfidf":
            self.char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2,
                                            sublinear_tf=True, max_features=200_000)
            self.ctx_vec = TfidfVectorizer(token_pattern=r"\S+", min_df=2, sublinear_tf=True,
                                           lowercase=False)
            x = sparse.hstack([self.char_vec.fit_transform(cur),
                               self.ctx_vec.fit_transform([" ".join(c) for c in ctx])])
        else:
            from gensim.models import Word2Vec
            corpus = [[t.lower() for t in s] for s in sentences]
            self.w2v = Word2Vec(corpus, vector_size=config.W2V_DIM, window=config.W2V_WINDOW,
                                min_count=2, sg=1, epochs=config.W2V_EPOCHS, workers=2,
                                seed=config.SEED)
            return np.hstack([self._w2v_matrix(windows), o.toarray().astype(np.float32)])
        return sparse.hstack([x, o]).tocsr()

    def transform(self, sentences):
        rows_ortho, cur, ctx, windows = self._flatten(sentences)
        o = self.ortho.transform(rows_ortho)
        if self.kind == "bow":
            x = self.vec.transform(self._bow_dicts(cur, ctx))
        elif self.kind == "tfidf":
            x = sparse.hstack([self.char_vec.transform(cur),
                               self.ctx_vec.transform([" ".join(c) for c in ctx])])
        else:
            return np.hstack([self._w2v_matrix(windows), o.toarray().astype(np.float32)])
        return sparse.hstack([x, o]).tocsr()

    @staticmethod
    def _bow_dicts(cur, ctx):
        dicts = []
        for w, c in zip(cur, ctx):
            d = {"w0=" + w: 1}
            for cw in c:
                d["ctx=" + cw] = d.get("ctx=" + cw, 0) + 1
            dicts.append(d)
        return dicts

    def _w2v_matrix(self, windows):
        kv = self.w2v.wv
        dim = kv.vector_size
        x = np.zeros((len(windows), dim * 5 + 1), dtype=np.float32)
        for r, win in enumerate(windows):
            for k, w in enumerate(win):
                if w is not None and w in kv.key_to_index:
                    v = kv[w]
                    x[r, k * dim:(k + 1) * dim] = v / (np.linalg.norm(v) + 1e-8)
            x[r, -1] = float(win[2] not in kv.key_to_index)   # OOV flag for the current word
        return x  # dense: a CSR copy of a dense block would triple memory

    @property
    def n_features(self):
        if self.kind == "bow":
            return len(self.vec.vocabulary_)
        if self.kind == "tfidf":
            return len(self.char_vec.vocabulary_) + len(self.ctx_vec.vocabulary_)
        return self.w2v.wv.vector_size * 5 + 1
