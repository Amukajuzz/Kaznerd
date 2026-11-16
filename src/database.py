"""SQLite layer: schema creation, corpus loading, model results, analytical queries."""
import sqlite3
from datetime import datetime

import pandas as pd

import config
from src import preprocessing
from src.spans import get_spans, span_text
from src.sql_queries import QUERIES

MODEL_TABLES = ("model_scores", "class_scores", "predicted_entities", "errors")


def connect(path=config.DB_PATH):
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def create_schema(conn, reset=False):
    if reset:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        conn.execute("PRAGMA foreign_keys = OFF")
        for t in tables:
            conn.execute(f"DROP TABLE IF EXISTS {t}")
        conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript((config.SQL_DIR / "schema.sql").read_text(encoding="utf-8"))


def load_corpus(conn, df, sentences_by_split, keywords_df=None, chunksize=config.CSV_CHUNKSIZE):
    """Fill entity_types, sentences, tokens (with stems/lemmas/stop flags), entities, keywords."""
    conn.executemany("INSERT OR REPLACE INTO entity_types VALUES (?, ?)", config.ENTITY_TYPES.items())

    sent_rows, ent_rows = [], []
    for split in config.SPLITS:
        for sid, tokens, tags in sentences_by_split[split]:
            sent_rows.append((sid, split, len(tokens), " ".join(tokens)))
            for etype, s, e in get_spans(tags):
                ent_rows.append((sid, etype, s, e, e - s, span_text(tokens, s, e)))
    conn.executemany("INSERT INTO sentences VALUES (?, ?, ?, ?)", sent_rows)
    conn.executemany("INSERT INTO entities (sentence_id, type_code, start_pos, end_pos, n_tokens, text) "
                     "VALUES (?, ?, ?, ?, ?, ?)", ent_rows)

    # normalise each distinct form once, then map (1.3M tokens -> ~150k forms)
    stop = preprocessing.load_stopwords()
    lemmatizer = preprocessing.get_lemmatizer()
    tok = df[["sentence_id", "position", "token", "tag"]].copy()
    tok["token"] = tok["token"].astype(str)
    tok["lower"] = tok["token"].str.lower()
    uniq = pd.Series(tok["lower"].unique())
    stem_map = dict(zip(uniq, uniq.map(preprocessing.stem)))
    lemma_map = dict(zip(uniq, uniq.map(lemmatizer.lemmatize)))
    tok["stem"] = tok["lower"].map(stem_map)
    tok["lemma"] = tok["lower"].map(lemma_map)
    tok["is_stopword"] = tok["lower"].isin(stop).astype("int8")
    tok["tag"] = tok["tag"].astype(str)
    cols = ["sentence_id", "position", "token", "lower", "stem", "lemma", "is_stopword", "tag"]
    tok[cols].to_sql("tokens", conn, if_exists="append", index=False, chunksize=chunksize)

    if keywords_df is not None and len(keywords_df):
        keywords_df[["type_code", "rank", "term", "score"]].to_sql(
            "keywords", conn, if_exists="append", index=False)
    conn.commit()
    return {"sentences": len(sent_rows), "entities": len(ent_rows), "tokens": len(tok)}


def table_counts(conn):
    tables = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")]
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in tables}


def store_model_results(conn, name, description, info, metrics, pred_rows, errors_df):
    """Replace all stored results of model `name` with the new run."""
    row = conn.execute("SELECT model_id FROM models WHERE name = ?", (name,)).fetchone()
    if row:
        for t in MODEL_TABLES:
            conn.execute(f"DELETE FROM {t} WHERE model_id = ?", (row[0],))
        conn.execute("DELETE FROM user_query_entities WHERE query_id IN "
                     "(SELECT query_id FROM user_queries WHERE model_id = ?)", (row[0],))
        conn.execute("DELETE FROM user_queries WHERE model_id = ?", (row[0],))
        conn.execute("DELETE FROM models WHERE model_id = ?", (row[0],))
    cur = conn.execute(
        "INSERT INTO models (name, description, train_sentences, n_features, train_time_s, "
        "predict_time_s, peak_mem_mb, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (name, description, info.get("train_sentences"), info.get("n_features"),
         info.get("train_time_s"), info.get("predict_time_s"), info.get("peak_mem_mb"),
         datetime.now().isoformat(timespec="seconds")))
    mid = cur.lastrowid
    conn.execute("INSERT INTO model_scores VALUES (?, ?, ?, ?, ?, ?)",
                 (mid, metrics["accuracy"], metrics["precision_micro"], metrics["recall_micro"],
                  metrics["f1_micro"], metrics["f1_macro"]))
    conn.executemany("INSERT INTO class_scores VALUES (?, ?, ?, ?, ?, ?)",
                     [(mid, r.entity_type, r.precision, r.recall, r.f1, int(r.support))
                      for r in metrics["per_class"].itertuples()])
    conn.executemany("INSERT INTO predicted_entities (model_id, sentence_id, type_code, start_pos, "
                     "end_pos, text) VALUES (?, ?, ?, ?, ?, ?)",
                     [(mid, *r) for r in pred_rows])
    if len(errors_df):
        e = errors_df.astype(object).where(errors_df.notna(), None)
        conn.executemany("INSERT INTO errors (model_id, sentence_id, error_type, gold_type, pred_type, "
                         "gold_text, pred_text, context) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                         [(mid, *r) for r in e.itertuples(index=False)])
    conn.commit()
    return mid


def run_query(conn, key):
    title, sql = QUERIES[key]
    return title, pd.read_sql_query(sql, conn)


def run_all_queries(conn, save=True, verbose=True):
    results = {}
    for key in QUERIES:
        try:
            title, df = run_query(conn, key)
        except (sqlite3.Error, pd.errors.DatabaseError) as exc:
            print(f"[{key}] failed: {exc}")
            continue
        results[key] = df
        if save:
            df.to_csv(config.RESULTS_DIR / f"sql_{key}.csv", index=False)
        if verbose:
            print(f"\n--- {key}: {title} ---")
            print(df.head(15).to_string(index=False, max_colwidth=60) if len(df) else "(no rows)")
    return results


def log_user_query(conn, model_name, text, entities):
    row = conn.execute("SELECT model_id FROM models WHERE name = ?", (model_name,)).fetchone()
    cur = conn.execute("INSERT INTO user_queries (model_id, input_text, created_at) VALUES (?, ?, ?)",
                       (row[0] if row else None, text, datetime.now().isoformat(timespec="seconds")))
    conn.executemany("INSERT INTO user_query_entities (query_id, type_code, text) VALUES (?, ?, ?)",
                     [(cur.lastrowid, e["type"], e["text"]) for e in entities])
    conn.commit()
