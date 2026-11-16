"""Exploratory analysis: frequencies, n-grams, entity statistics, word clouds."""
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
from nltk.util import ngrams

import config
from src import plotting, preprocessing
from src.spans import get_spans, span_text

FONT_PATH = plotting.matplotlib.get_data_path() + "/fonts/ttf/DejaVuSans.ttf"


def token_frequencies(sentences, stopwords=None, lowercase=True, words_only=True):
    """Counter of tokens over [(sid, tokens, tags)], optionally without stop words / punctuation."""
    counts = Counter()
    for _, tokens, _ in sentences:
        for t in tokens:
            if words_only and not preprocessing.is_word(t):
                continue
            w = t.lower() if lowercase else t
            if stopwords and w in stopwords:
                continue
            counts[w] += 1
    return counts


def ngram_frequencies(sentences, n, stopwords=None):
    """Counter of word n-grams; n-grams containing punctuation or stop words are skipped."""
    counts = Counter()
    for _, tokens, _ in sentences:
        words = [t.lower() for t in tokens]
        for gram in ngrams(words, n):
            if all(preprocessing.is_word(w) and not (stopwords and w in stopwords) for w in gram):
                counts[" ".join(gram)] += 1
    return counts


def entity_mentions(sentences):
    """List of dicts {type, text, length} for every gold entity."""
    rows = []
    for sid, tokens, tags in sentences:
        for etype, s, e in get_spans(tags):
            rows.append({"sentence_id": sid, "type": etype, "text": span_text(tokens, s, e),
                         "length": e - s})
    return pd.DataFrame(rows)


def context_keywords(sentences, stopwords, window=3, top_k=15):
    """TF-IDF keywords of the words around each entity type (one 'document' per type).

    Answers: which words typically signal a PERSON, a POSITION, a DATE...?"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    docs = {t: [] for t in config.ENTITY_TYPES}
    for _, tokens, tags in sentences:
        low = [t.lower() for t in tokens]
        for etype, s, e in get_spans(tags):
            ctx = low[max(0, s - window):s] + low[e:e + window]
            docs[etype].extend(w for w in ctx if w.isalpha() and w not in stopwords)
    types = [t for t in docs if docs[t]]
    vec = TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True, min_df=1)
    mat = vec.fit_transform([" ".join(docs[t]) for t in types])
    vocab = vec.get_feature_names_out()
    rows = []
    for i, t in enumerate(types):
        row = mat[i].toarray().ravel()
        for rank, j in enumerate(row.argsort()[::-1][:top_k], 1):
            rows.append({"type_code": t, "rank": rank, "term": vocab[j], "score": float(row[j])})
    return pd.DataFrame(rows)


def make_wordclouds(mentions, path, types=("PERSON", "GPE", "ORGANISATION", "POSITION")):
    from wordcloud import WordCloud
    plotting.setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, etype in zip(axes.ravel(), types):
        freqs = Counter(mentions.loc[mentions["type"] == etype, "text"])
        wc = WordCloud(width=800, height=450, background_color=plotting.SURFACE,
                       font_path=FONT_PATH, colormap=plotting.TEXT_CMAP, max_words=80,
                       prefer_horizontal=0.95, random_state=config.SEED)
        wc.generate_from_frequencies(dict(freqs.most_common(300)))
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(etype)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_sentence_lengths(sentences, path):
    plotting.setup_style()
    lengths = [len(t) for _, t, _ in sentences]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(lengths, bins=range(0, 61, 2), color=plotting.SERIES[0], edgecolor=plotting.SURFACE)
    ax.set_title("Sentence length (tokens), train split")
    ax.set_xlabel("tokens per sentence")
    ax.set_ylabel("sentences")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run_eda(sentences_by_split, top_n=20):
    """Produce all EDA figures and tables. Returns a dict of summary numbers."""
    fig_dir, res_dir = config.FIGURES_DIR, config.RESULTS_DIR
    stop = preprocessing.load_stopwords()
    train = sentences_by_split["train"]
    all_sents = [s for split in config.SPLITS for s in sentences_by_split[split]]

    raw = token_frequencies(train)
    clean = token_frequencies(train, stopwords=stop)
    bigrams = ngram_frequencies(train, 2, stop)
    trigrams = ngram_frequencies(train, 3, stop)

    plotting.barh(*zip(*raw.most_common(top_n)), "Top words, train (with stop words)",
                  fig_dir / "top_words_raw.png", "frequency")
    plotting.barh(*zip(*clean.most_common(top_n)), "Top words, train (stop words removed)",
                  fig_dir / "top_words_clean.png", "frequency")
    plotting.barh(*zip(*bigrams.most_common(top_n)), "Top bigrams (no stop words / punctuation)",
                  fig_dir / "top_bigrams.png", "frequency", figsize=(9, 6))
    plotting.barh(*zip(*trigrams.most_common(15)), "Top trigrams (no stop words / punctuation)",
                  fig_dir / "top_trigrams.png", "frequency", figsize=(10, 5))

    mentions = entity_mentions(all_sents)
    type_counts = mentions["type"].value_counts()
    plotting.barh(type_counts.index, type_counts.values, "Entity mentions per class (all splits)",
                  fig_dir / "entity_distribution.png", "mentions", figsize=(8, 8))
    make_wordclouds(mentions, fig_dir / "wordcloud_entities.png")
    plot_sentence_lengths(train, fig_dir / "sentence_lengths.png")

    pd.DataFrame(raw.most_common(200), columns=["word", "count"]).to_csv(res_dir / "freq_words_raw.csv", index=False)
    pd.DataFrame(clean.most_common(200), columns=["word", "count"]).to_csv(res_dir / "freq_words_clean.csv", index=False)
    pd.DataFrame(bigrams.most_common(100), columns=["bigram", "count"]).to_csv(res_dir / "freq_bigrams.csv", index=False)
    pd.DataFrame(trigrams.most_common(100), columns=["trigram", "count"]).to_csv(res_dir / "freq_trigrams.csv", index=False)
    type_counts.rename_axis("type").reset_index(name="mentions").to_csv(res_dir / "entity_counts.csv", index=False)

    total_tokens = sum(raw.values())
    stop_tokens = total_tokens - sum(clean.values())
    summary = {
        "sentences": {s: len(sentences_by_split[s]) for s in config.SPLITS},
        "tokens": {s: sum(len(t) for _, t, _ in sentences_by_split[s]) for s in config.SPLITS},
        "train_word_types": len(raw),
        "train_word_tokens": total_tokens,
        "stopword_token_share": stop_tokens / total_tokens,
        "entity_mentions": int(len(mentions)),
        "entity_classes": int(mentions["type"].nunique()),
        "mean_entity_length": float(mentions["length"].mean()),
    }
    return summary
