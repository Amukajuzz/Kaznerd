"""Evaluation: seqeval metrics, entity-type confusion matrix, span-level error analysis."""
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from seqeval.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score

from src import plotting
from src.spans import get_spans, span_text


def evaluate(y_true, y_pred):
    """Entity-level (seqeval, conlleval-style) micro/macro scores plus token accuracy."""
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    per_class = (pd.DataFrame(report).T.drop(index=["micro avg", "macro avg", "weighted avg"],
                                             errors="ignore")
                 .rename(columns={"f1-score": "f1"}).reset_index(names="entity_type"))
    per_class["support"] = per_class["support"].astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_micro": precision_score(y_true, y_pred, zero_division=0),
        "recall_micro": recall_score(y_true, y_pred, zero_division=0),
        "f1_micro": f1_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "per_class": per_class.sort_values("support", ascending=False),
    }


def _strip(tag):
    return tag[2:] if tag != "O" else "O"


def type_confusion(y_true, y_pred):
    """Token-level confusion matrix over entity types (B-/I- merged), rows = gold."""
    pairs = Counter((_strip(g), _strip(p)) for gs, ps in zip(y_true, y_pred) for g, p in zip(gs, ps))
    gold_freq = Counter()
    for (g, _), n in pairs.items():
        gold_freq[g] += n
    labels = ["O"] + [l for l, _ in gold_freq.most_common() if l != "O"]
    mat = pd.DataFrame(0, index=labels, columns=labels, dtype=np.int64)
    for (g, p), n in pairs.items():
        if g in mat.index and p in mat.columns:
            mat.loc[g, p] += n
    return mat


def plot_confusion(mat, title, path, drop_o_diagonal=True):
    """Row-normalised heatmap; the dominant O->O cell is hidden so entity errors stay visible."""
    norm = mat.div(mat.sum(axis=1).replace(0, 1), axis=0)
    data = norm.to_numpy().copy()
    if drop_o_diagonal:
        data[0, 0] = np.nan
    plotting.setup_style()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(data, cmap=plotting.SEQ_CMAP, vmin=0, vmax=1)
    ax.grid(False)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(mat.index)))
    ax.set_yticklabels(mat.index, fontsize=8)
    ax.set_xlabel("predicted type")
    ax.set_ylabel("gold type")
    ax.set_title(title)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if not np.isnan(v) and v >= 0.01:
                ax.text(j, i, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=6,
                        color="white" if v > 0.55 else plotting.INK)
    fig.colorbar(im, ax=ax, fraction=0.03, label="share of gold tokens")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def error_analysis(sentences, predictions):
    """Classify every span-level error.

    error_type: type_error (same span, other type), boundary_error (overlap, same type),
                type_and_boundary, missed (no overlapping prediction), spurious (no overlapping gold).
    """
    rows = []
    for (sid, tokens, gold_tags), pred_tags in zip(sentences, predictions):
        gold = get_spans(gold_tags)
        pred = get_spans(pred_tags)
        pred_set = set(pred)
        for g in gold:
            if g in pred_set:
                continue
            overlaps = [p for p in pred if p[1] < g[2] and g[1] < p[2]]
            if not overlaps:
                kind, p = "missed", None
            else:
                p = max(overlaps, key=lambda s: min(s[2], g[2]) - max(s[1], g[1]))
                same_bounds = (p[1], p[2]) == (g[1], g[2])
                if same_bounds:
                    kind = "type_error"
                elif p[0] == g[0]:
                    kind = "boundary_error"
                else:
                    kind = "type_and_boundary"
            rows.append(_err_row(sid, tokens, kind, g, p))
        gold_set = set(gold)
        for p in pred:
            if p in gold_set:
                continue
            if not any(g[1] < p[2] and p[1] < g[2] for g in gold):
                rows.append(_err_row(sid, tokens, "spurious", None, p))
    return pd.DataFrame(rows, columns=["sentence_id", "error_type", "gold_type", "pred_type",
                                       "gold_text", "pred_text", "context"])


def _err_row(sid, tokens, kind, g, p):
    lo = min(x[1] for x in (g, p) if x)
    hi = max(x[2] for x in (g, p) if x)
    return {
        "sentence_id": sid, "error_type": kind,
        "gold_type": g[0] if g else None, "pred_type": p[0] if p else None,
        "gold_text": span_text(tokens, g[1], g[2]) if g else None,
        "pred_text": span_text(tokens, p[1], p[2]) if p else None,
        "context": span_text(tokens, max(0, lo - 4), min(len(tokens), hi + 4)),
    }


def predicted_entities(sentences, predictions):
    """Rows (sentence_id, type, start, end, text) for storing predictions in SQLite."""
    rows = []
    for (sid, tokens, _), tags in zip(sentences, predictions):
        for etype, s, e in get_spans(tags):
            rows.append((sid, etype, s, e, span_text(tokens, s, e)))
    return rows
