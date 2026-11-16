"""NER taggers sharing one interface: fit(sentences) / predict(token_lists) / save / load.

  * LookupBaseline  — most frequent tag per word form seen in training (else 'O')
  * WindowSGDTagger — per-token linear classifier on BoW / TF-IDF / Word2Vec windows
  * CRFTagger       — linear-chain CRF (python-crfsuite) with hand-crafted features
The transformer model lives in src/transformer_ner.py (needs a GPU).
"""
import random
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import pycrfsuite
from sklearn.linear_model import SGDClassifier

import config
from src.features import WindowVectorizer, crf_sentence_features


class LookupBaseline:
    name = "baseline_lookup"
    description = "Most frequent tag per word form (exact, then lowercase), else O"

    def fit(self, sentences):
        exact, lower = defaultdict(Counter), defaultdict(Counter)
        for tokens, tags in sentences:
            for t, g in zip(tokens, tags):
                exact[t][g] += 1
                lower[t.lower()][g] += 1
        self.exact = {t: c.most_common(1)[0][0] for t, c in exact.items()}
        self.lower = {t: c.most_common(1)[0][0] for t, c in lower.items()}
        return self

    def predict(self, token_lists):
        return [[self.exact.get(t) or self.lower.get(t.lower(), "O") for t in tokens]
                for tokens in token_lists]

    def save(self, path):
        joblib.dump(self, path, compress=3)


class WindowSGDTagger:
    """Independent per-token classification; same classifier for every representation."""

    def __init__(self, kind, max_sentences=config.LINEAR_TRAIN_SENTENCES):
        self.kind = kind
        self.name = f"sgd_{kind}"
        self.description = {"bow": "SGD logistic regression on Bag-of-Words window",
                            "tfidf": "SGD logistic regression on char/word TF-IDF window",
                            "word2vec": "SGD logistic regression on Word2Vec window"}[kind]
        self.max_sentences = max_sentences

    def fit(self, sentences):
        sentences = list(sentences)
        if self.max_sentences and len(sentences) > self.max_sentences:
            sentences = random.Random(config.SEED).sample(sentences, self.max_sentences)
        self.train_sentences = len(sentences)
        self.vectorizer = WindowVectorizer(self.kind)
        x = self.vectorizer.fit_transform([t for t, _ in sentences])
        y = np.array([g for _, tags in sentences for g in tags])
        self.clf = SGDClassifier(loss="log_loss", alpha=1e-6 if self.kind == "word2vec" else 1e-5,
                                 max_iter=30, tol=1e-4, n_jobs=-1, random_state=config.SEED)
        self.clf.fit(x, y)
        return self

    def predict(self, token_lists, batch=5000):
        out = []
        for b in range(0, len(token_lists), batch):
            chunk = token_lists[b:b + batch]
            flat = self.clf.predict(self.vectorizer.transform(chunk))
            pos = 0
            for tokens in chunk:
                out.append(list(flat[pos:pos + len(tokens)]))
                pos += len(tokens)
        return out

    def save(self, path):
        joblib.dump(self, path, compress=3)


class CRFTagger:
    name = "crf"
    description = "Linear-chain CRF (L-BFGS, L1+L2) with lexical, suffix, stem and shape features"

    def __init__(self, model_file="crf.crfsuite", c1=config.CRF_C1, c2=config.CRF_C2,
                 max_iterations=config.CRF_MAX_ITERATIONS):
        self.model_file = model_file      # stored relative to MODELS_DIR so the project is movable
        self.params = {"c1": c1, "c2": c2, "max_iterations": max_iterations,
                       "feature.possible_transitions": True}
        self._tagger = None

    @property
    def model_path(self):
        return str(config.MODELS_DIR / self.model_file)

    def fit(self, sentences):
        trainer = pycrfsuite.Trainer(verbose=False)
        trainer.set_params(self.params)
        for tokens, tags in sentences:      # features are generated sentence by sentence
            trainer.append(crf_sentence_features(tokens), tags)
        Path(self.model_path).parent.mkdir(parents=True, exist_ok=True)
        trainer.train(self.model_path)
        self.train_log = getattr(trainer.logparser, "last_iteration", None)
        self._tagger = None
        return self

    @property
    def tagger(self):
        if self._tagger is None:
            if not Path(self.model_path).exists():
                raise FileNotFoundError(f"{self.model_path} not found — train the CRF first.")
            self._tagger = pycrfsuite.Tagger()
            self._tagger.open(self.model_path)
        return self._tagger

    def predict(self, token_lists):
        return [self.tagger.tag(crf_sentence_features(tokens)) if tokens else []
                for tokens in token_lists]

    def top_features(self, k=20):
        info = self.tagger.info()
        state = sorted(info.state_features.items(), key=lambda kv: -kv[1])
        trans = sorted(info.transitions.items(), key=lambda kv: -kv[1])
        return state[:k], trans[:k]

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tagger"] = None          # the C tagger is not picklable; reopened lazily
        return state

    def save(self, path):
        joblib.dump(self, path)


def build_models():
    """All locally trainable models, baseline first."""
    return [LookupBaseline(), WindowSGDTagger("bow"), WindowSGDTagger("tfidf"),
            WindowSGDTagger("word2vec"), CRFTagger()]


def model_path(name):
    return config.MODELS_DIR / f"{name}.joblib"


def load_model(name):
    path = model_path(name)
    if not path.exists():
        raise FileNotFoundError(f"Model '{name}' not found at {path} — train models first (menu 4).")
    return joblib.load(path)
