"""Processing-approach comparisons: CSV vs Parquet, dtype optimisation, Pandas vs SQL."""
import os
import statistics
import time

import pandas as pd

import config
from src import data_loader, database, plotting
from src.utils import ResourceMonitor


def _best_of(fn, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    return result, min(times), statistics.median(times)


def format_benchmark(df):
    """Write/read the token table as CSV and Parquet; measure time, size and memory."""
    rows = []
    tmp = {"csv": config.PROCESSED_DIR / "_bench.csv", "parquet": config.PROCESSED_DIR / "_bench.parquet"}
    writers = {"csv": lambda p: df.to_csv(p, index=False),
               "parquet": lambda p: df.to_parquet(p, index=False, compression="snappy")}
    readers = {"csv": lambda p, cols=None: pd.read_csv(p, usecols=cols, keep_default_na=False),
               "parquet": lambda p, cols=None: pd.read_parquet(p, columns=cols)}
    for fmt, path in tmp.items():
        with ResourceMonitor() as w:
            writers[fmt](path)
        with ResourceMonitor() as r:
            loaded = readers[fmt](path)
        _, cols_best, _ = _best_of(lambda: readers[fmt](path, ["token", "tag"]))
        rows.append({"format": fmt, "file_mb": os.path.getsize(path) / 2 ** 20,
                     "write_s": w.seconds, "read_all_s": r.seconds, "read_2cols_s": cols_best,
                     "read_peak_mem_mb": r.peak_mb,
                     "in_memory_mb": data_loader.memory_mb(loaded)})
        del loaded
    with ResourceMonitor() as c:
        _, n_chunks = data_loader.tag_counts_chunked(tmp["csv"])
    rows.append({"format": f"csv, chunked ({n_chunks} x {config.CSV_CHUNKSIZE:,} rows, tag column)",
                 "file_mb": os.path.getsize(tmp["csv"]) / 2 ** 20, "write_s": None,
                 "read_all_s": c.seconds, "read_2cols_s": None, "read_peak_mem_mb": c.peak_mb,
                 "in_memory_mb": None})
    for path in tmp.values():
        path.unlink(missing_ok=True)
    return pd.DataFrame(rows)


def dtype_benchmark():
    """Memory of the token table with default object dtypes vs optimised dtypes."""
    raw = data_loader.build_token_frame(optimize=False)
    opt = data_loader.optimize_dtypes(raw)
    out = pd.DataFrame({
        "column": list(raw.columns),
        "default_dtype": [str(raw[c].dtype) for c in raw.columns],
        "default_mb": [raw[c].memory_usage(deep=True) / 2 ** 20 for c in raw.columns],
        "optimised_dtype": [str(opt[c].dtype) for c in opt.columns],
        "optimised_mb": [opt[c].memory_usage(deep=True) / 2 ** 20 for c in opt.columns],
    })
    total = pd.DataFrame([{"column": "TOTAL", "default_dtype": "", "optimised_dtype": "",
                           "default_mb": out["default_mb"].sum(), "optimised_mb": out["optimised_mb"].sum()}])
    return pd.concat([out, total], ignore_index=True)


def pandas_vs_sql(df, conn):
    """Same three analytical tasks in Pandas (data in memory) and SQLite (indexed tables)."""
    low = df.assign(lower=df["token"].astype(str).str.lower(), tag=df["tag"].astype(str))

    def pd_mentions():
        t = low["tag"]
        return t[t.str.startswith("B-")].value_counts()

    def pd_sentence_len():
        size = low.groupby("sentence_id", observed=True).agg(split=("split", "first"), n=("token", "size"))
        return size.groupby("split", observed=True)["n"].mean()

    def pd_top_words():
        c = low[low["tag"] != "O"].groupby(["tag", "lower"]).size().rename("n").reset_index()
        return c.sort_values(["tag", "n"], ascending=[True, False]).groupby("tag").head(3)

    sql = {
        "entity mentions per class": "SELECT tag, COUNT(*) AS n FROM tokens WHERE tag LIKE 'B-%' GROUP BY tag",
        "mean sentence length per split":
            "SELECT s.split, AVG(x.n) FROM (SELECT sentence_id, COUNT(*) AS n FROM tokens GROUP BY sentence_id) x "
            "JOIN sentences s USING (sentence_id) GROUP BY s.split",
        "top-3 words per tag (window function)":
            "WITH c AS (SELECT tag, lower, COUNT(*) AS n FROM tokens WHERE tag <> 'O' GROUP BY tag, lower) "
            "SELECT * FROM (SELECT *, ROW_NUMBER() OVER (PARTITION BY tag ORDER BY n DESC) rn FROM c) WHERE rn <= 3",
    }
    pandas_fns = dict(zip(sql, (pd_mentions, pd_sentence_len, pd_top_words)))
    rows = []
    for task, query in sql.items():
        _, pd_best, _ = _best_of(pandas_fns[task])
        _, sql_best, _ = _best_of(lambda: pd.read_sql_query(query, conn))
        rows.append({"task": task, "pandas_s": pd_best, "sqlite_s": sql_best,
                     "faster": "pandas" if pd_best < sql_best else "sqlite"})
    _, load_s, _ = _best_of(lambda: pd.read_parquet(config.TOKENS_PARQUET), repeats=1)
    rows.append({"task": "(one-off) load parquet into pandas", "pandas_s": load_s,
                 "sqlite_s": None, "faster": None})
    return pd.DataFrame(rows)


def run_benchmarks():
    df = data_loader.load_tokens("parquet")
    fmt = format_benchmark(df)
    dtypes = dtype_benchmark()
    conn = database.connect()
    try:
        pvs = pandas_vs_sql(df, conn)
    finally:
        conn.close()
    fmt.to_csv(config.RESULTS_DIR / "bench_formats.csv", index=False)
    dtypes.to_csv(config.RESULTS_DIR / "bench_dtypes.csv", index=False)
    pvs.to_csv(config.RESULTS_DIR / "bench_pandas_vs_sql.csv", index=False)

    main = fmt[fmt["format"].isin(["csv", "parquet"])]
    plotting.grouped_bars(["file size, MB", "read all, s x10", "write, s x10"],
                          {r.format: [r.file_mb, r.read_all_s * 10, r.write_s * 10] for r in main.itertuples()},
                          "CSV vs Parquet: size and speed (lower is better)",
                          config.FIGURES_DIR / "bench_formats.png", fmt="{:.1f}")
    p = pvs.dropna(subset=["sqlite_s"])
    plotting.grouped_bars(list(p["task"]), {"pandas": list(p["pandas_s"] * 1000),
                                            "sqlite": list(p["sqlite_s"] * 1000)},
                          "Pandas vs SQLite, same task (ms, lower is better)",
                          config.FIGURES_DIR / "bench_pandas_vs_sql.png", ylabel="ms", fmt="{:.0f}")
    return fmt, dtypes, pvs
