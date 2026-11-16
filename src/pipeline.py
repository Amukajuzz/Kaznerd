"""Pipeline steps called from the console menu (main.py)."""
import pandas as pd

import config
from src import benchmarks, data_loader, database, eda, evaluation, plotting, preprocessing
from src.models import build_models, model_path
from src.utils import ResourceMonitor, print_header, save_json

COMPARISON_CSV = config.RESULTS_DIR / "model_comparison.csv"


# ---------------------------------------------------------------------------
# 1. data
# ---------------------------------------------------------------------------
def step_prepare():
    print_header("1. Prepare data: CoNLL -> optimised DataFrame -> Parquet + CSV")
    with ResourceMonitor() as mon:
        df = data_loader.build_token_frame()
        data_loader.save_tokens(df)
    summary = {
        "rows": len(df), "sentences": int(df["sentence_id"].nunique()),
        "rows_per_split": df["split"].value_counts().reindex(config.SPLITS).to_dict(),
        "tags": int(df["tag"].nunique()), "in_memory_mb": data_loader.memory_mb(df),
        "seconds": mon.seconds,
    }
    save_json(summary, config.RESULTS_DIR / "data_summary.json")
    print(f"{summary['rows']:,} token rows, {summary['sentences']:,} sentences, {summary['tags']} tags; "
          f"{summary['in_memory_mb']:.1f} MB in memory; {mon.seconds:.1f} s")
    print(f"Saved {config.TOKENS_PARQUET.name} and {config.TOKENS_CSV.name} in {config.PROCESSED_DIR}")
    return summary


# ---------------------------------------------------------------------------
# 2. database
# ---------------------------------------------------------------------------
def step_build_db():
    print_header("2. Build SQLite database")
    df = data_loader.load_tokens("parquet")
    sentences = data_loader.to_sentences(df)
    keywords = eda.context_keywords(sentences["train"], preprocessing.load_stopwords())
    conn = database.connect()
    try:
        with ResourceMonitor() as mon:
            database.create_schema(conn, reset=True)
            database.load_corpus(conn, df, sentences, keywords)
        counts = database.table_counts(conn)
    finally:
        conn.close()
    for t, n in counts.items():
        print(f"  {t:<22}{n:>12,}")
    print(f"Database {config.DB_PATH.name}: {config.DB_PATH.stat().st_size / 2**20:.1f} MB, built in {mon.seconds:.1f} s")
    print("Note: model tables are filled by step 4 (training).")
    return counts


# ---------------------------------------------------------------------------
# 3. EDA + preprocessing comparison
# ---------------------------------------------------------------------------
def step_eda():
    print_header("3. Preprocessing & frequency analysis")
    sentences = data_loader.load_sentences()
    summary = eda.run_eda(sentences)
    norm = preprocessing.compare_normalizers([t for _, toks, _ in sentences["train"] for t in toks])
    norm.to_csv(config.RESULTS_DIR / "stemming_vs_lemmatization.csv", index=False)
    summary["ud_test_tokens"] = norm.attrs["ud_test_tokens"]
    summary["ud_test_oov"] = norm.attrs["ud_test_oov"]
    save_json(summary, config.RESULTS_DIR / "eda_summary.json")
    plotting.grouped_bars(list(norm["method"]),
                          {"accuracy vs UD gold lemmas": list(norm["ud_accuracy"]),
                           "accuracy on unseen words": list(norm["ud_oov_accuracy"]),
                           "vocabulary reduction (share)": list(norm["vocab_reduction_pct"] / 100)},
                          "Stemming vs lemmatization (Kazakh)", config.FIGURES_DIR / "stem_vs_lemma.png",
                          ylim=(0, 1))
    print(f"Train: {summary['train_word_tokens']:,} word tokens, {summary['train_word_types']:,} types; "
          f"stop words = {summary['stopword_token_share']:.1%} of word tokens")
    print(f"{summary['entity_mentions']:,} gold entities in {summary['entity_classes']} classes")
    print("\nStemming vs lemmatization:")
    print(norm.round(3).to_string(index=False))
    print(f"\nFigures saved to {config.FIGURES_DIR}")
    return summary, norm


# ---------------------------------------------------------------------------
# 4. models
# ---------------------------------------------------------------------------
def evaluate_and_record(tagger, test_sentences, info):
    """Predict the test split, score it, analyse errors, store everything (CSV + SQLite)."""
    tokens = [t for _, t, _ in test_sentences]
    gold = [g for _, _, g in test_sentences]
    with ResourceMonitor() as mon:
        pred = tagger.predict(tokens)
    info["predict_time_s"] = mon.seconds
    metrics = evaluation.evaluate(gold, pred)
    errors = evaluation.error_analysis(test_sentences, pred)
    name = tagger.name

    metrics["per_class"].to_csv(config.RESULTS_DIR / f"per_class_{name}.csv", index=False)
    errors.to_csv(config.RESULTS_DIR / f"errors_{name}.csv", index=False)
    conf = evaluation.type_confusion(gold, pred)
    conf.to_csv(config.RESULTS_DIR / f"confusion_{name}.csv")
    evaluation.plot_confusion(conf, f"Token-level class confusion — {name} (row-normalised)",
                              config.FIGURES_DIR / f"confusion_{name}.png")

    conn = database.connect()
    try:
        database.store_model_results(conn, name, tagger.description, info, metrics,
                                     evaluation.predicted_entities(test_sentences, pred), errors)
    finally:
        conn.close()

    row = {"model": name, "description": tagger.description,
           **{k: metrics[k] for k in ("accuracy", "precision_micro", "recall_micro", "f1_micro", "f1_macro")},
           **{k: info.get(k) for k in ("train_sentences", "n_features", "train_time_s",
                                       "predict_time_s", "peak_mem_mb")}}
    table = pd.read_csv(COMPARISON_CSV) if COMPARISON_CSV.exists() else pd.DataFrame()
    if len(table):
        table = table[table["model"] != name]
    table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    table.to_csv(COMPARISON_CSV, index=False)
    plot_comparison(table)
    print(f"  {name:<18} F1 micro {metrics['f1_micro']:.4f} | F1 macro {metrics['f1_macro']:.4f} | "
          f"acc {metrics['accuracy']:.4f} | train {info.get('train_time_s', 0):.0f} s | "
          f"predict {mon.seconds:.1f} s | peak +{info.get('peak_mem_mb', 0):.0f} MB | errors {len(errors):,}")
    return metrics, errors


def plot_comparison(table):
    order = ["baseline_lookup", "sgd_bow", "sgd_tfidf", "sgd_word2vec", "crf", "transformer"]
    t = table.set_index("model").reindex([m for m in order if m in set(table["model"])])
    plotting.grouped_bars(list(t.index), {"F1 micro": list(t["f1_micro"]), "F1 macro": list(t["f1_macro"]),
                                          "precision": list(t["precision_micro"]),
                                          "recall": list(t["recall_micro"])},
                          "Test-set scores by model (entity level, seqeval)",
                          config.FIGURES_DIR / "model_comparison.png", ylim=(0, 1.05))


def step_train(only=None):
    print_header("4. Train and evaluate models (test split)")
    sentences = data_loader.load_sentences()
    train = [(t, g) for _, t, g in sentences["train"]]
    for model in build_models():
        if only and model.name not in only:
            continue
        print(f"Training {model.name} ...", flush=True)
        with ResourceMonitor() as mon:
            model.fit(train)
        info = {"train_sentences": getattr(model, "train_sentences", len(train)),
                "n_features": getattr(getattr(model, "vectorizer", None), "n_features", None),
                "train_time_s": mon.seconds, "peak_mem_mb": mon.peak_mb}
        if model.name == "crf":
            info["n_features"] = len(model.tagger.info().state_features)
        if model.name == "baseline_lookup":
            info["n_features"] = len(model.exact)
        model.save(model_path(model.name))
        evaluate_and_record(model, sentences["test"], info)
        del model
    print(f"\nComparison table: {COMPARISON_CSV}")


# ---------------------------------------------------------------------------
# 5-7. SQL, benchmarks, error analysis
# ---------------------------------------------------------------------------
def step_sql():
    print_header("5. Analytical SQL queries (results also saved as CSV)")
    conn = database.connect()
    try:
        return database.run_all_queries(conn)
    finally:
        conn.close()


def step_benchmarks():
    print_header("6. Benchmarks: CSV vs Parquet, dtypes, Pandas vs SQL")
    fmt, dtypes, pvs = benchmarks.run_benchmarks()
    print(fmt.round(3).to_string(index=False))
    print()
    print(dtypes.round(1).to_string(index=False))
    print()
    print(pvs.round(4).to_string(index=False))
    return fmt, dtypes, pvs


def step_error_report(model_name=None):
    print_header("7. Error analysis")
    if not COMPARISON_CSV.exists():
        raise FileNotFoundError("No results yet — run step 4 first.")
    table = pd.read_csv(COMPARISON_CSV).sort_values("f1_micro", ascending=False)
    name = model_name or table.iloc[0]["model"]
    errors = pd.read_csv(config.RESULTS_DIR / f"errors_{name}.csv").fillna("—")
    per_class = pd.read_csv(config.RESULTS_DIR / f"per_class_{name}.csv")
    print(f"Model: {name}  ({len(errors):,} span errors on the test split)\n")
    print(errors["error_type"].value_counts().rename("count").to_frame()
          .assign(share=lambda d: (d["count"] / d["count"].sum()).round(3)).to_string())
    conf = errors[errors["error_type"].isin(["type_error", "type_and_boundary"])]
    print("\nTop class confusions (gold -> predicted):")
    print(conf.groupby(["gold_type", "pred_type"]).size().sort_values(ascending=False).head(10).to_string())
    print("\nWeakest classes:")
    print(per_class.sort_values("f1").head(8)[["entity_type", "precision", "recall", "f1", "support"]]
          .round(3).to_string(index=False))
    print("\nExamples:")
    for kind in ("missed", "spurious", "type_error", "boundary_error"):
        ex = errors[errors["error_type"] == kind].head(3)
        for r in ex.itertuples():
            print(f"  [{kind}] gold={r.gold_type}:{r.gold_text} | pred={r.pred_type}:{r.pred_text}\n"
                  f"      … {r.context} …")
    print(f"\nConfusion matrix: {config.FIGURES_DIR / f'confusion_{name}.png'}")
    if name == "crf":
        from src.models import load_model
        state, trans = load_model("crf").top_features(12)
        print("\nStrongest CRF state features:")
        for (attr, label), w in state:
            print(f"  {w:7.3f}  {label:<16} {attr}")
        print("Strongest CRF transitions:")
        for (a, b), w in trans[:8]:
            print(f"  {w:7.3f}  {a} -> {b}")
    return errors


def step_summary():
    print_header("Results summary")
    if not COMPARISON_CSV.exists():
        raise FileNotFoundError("No results yet — run step 4 first.")
    t = pd.read_csv(COMPARISON_CSV).sort_values("f1_micro", ascending=False)
    cols = ["model", "f1_micro", "f1_macro", "precision_micro", "recall_micro", "accuracy",
            "train_time_s", "predict_time_s", "peak_mem_mb", "train_sentences"]
    print(t[cols].round(4).to_string(index=False))
    return t
