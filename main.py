"""Kazakh NER on KazNERD — interactive console menu.

    python main.py              # interactive menu
    python main.py --run-all    # steps 1-7 without prompts
    python main.py --step 4     # a single step
"""
import argparse
import sqlite3
import sys

from src import pipeline
from src.utils import ensure_dirs, print_header

MENU = """
 1 - Prepare data (CoNLL -> Parquet/CSV, dtype optimisation)
 2 - Build SQLite database (corpus, entities, keywords)
 3 - Preprocessing & frequency analysis (stop words, stemming vs lemmatization, n-grams, word clouds)
 4 - Train & evaluate models (baseline, BoW / TF-IDF / Word2Vec, CRF)
 5 - Run analytical SQL queries
 6 - Benchmarks (CSV vs Parquet, Pandas vs SQL)
 7 - Error analysis & confusion matrix
 8 - Recognise entities in your own text
 9 - Show results summary
 0 - Exit"""

STEPS = {
    "1": pipeline.step_prepare,
    "2": pipeline.step_build_db,
    "3": pipeline.step_eda,
    "4": pipeline.step_train,
    "5": pipeline.step_sql,
    "6": pipeline.step_benchmarks,
    "7": pipeline.step_error_report,
    "9": pipeline.step_summary,
}


def choose_model():
    """Let the user pick a trained model; Enter selects the best one."""
    from src import predict
    models = predict.available_models()
    if not models:
        raise FileNotFoundError("No trained model found — run step 4 first.")
    best = predict.best_model_name()
    print("Available models: " + ", ".join(f"{i}={m}" for i, m in enumerate(models, 1)))
    while True:
        raw = input(f"Model number [Enter = best: {best}]: ").strip()
        if not raw:
            return best
        try:
            idx = int(raw)
            if 1 <= idx <= len(models):
                return models[idx - 1]
            print(f"Enter a number between 1 and {len(models)}.")
        except ValueError:
            print("Please enter a number.")


def interactive_prediction():
    from src import database, predict
    print_header("8. Recognise entities — type Kazakh text, empty line to return")
    name = choose_model()
    print(f"Loading {name} ...")
    tagger = predict.get_tagger(name)
    print('Example: Қасым-Жомарт Тоқаев Астанада ҚР Ұлттық банкінің төрағасымен кездесті.')
    session = []
    conn = database.connect()
    try:
        while True:
            text = input("\nText> ").strip()
            if not text:
                break
            entities = predict.predict_text(text, tagger)
            print(predict.highlight(text, tagger))
            if entities:
                for e in entities:
                    print(f"  {e['type']:<14} {e['text']:<35} {e['description']}")
            else:
                print("  (no entities found)")
            try:
                database.log_user_query(conn, name, text, entities)
            except sqlite3.Error as exc:
                print(f"  (not saved to the database: {exc})")
            session.extend({"model": name, "input": text, **e} for e in entities)
    finally:
        conn.close()
    if session:
        print(f"Saved {len(session)} entities to {predict.save_session(session)}")


def run_step(key):
    if key == "8":
        interactive_prediction()
    else:
        STEPS[key]()


def menu():
    while True:
        print(MENU)
        choice = input("Choose an option: ").strip()
        if choice == "0":
            print("Bye!")
            return
        if choice not in STEPS and choice != "8":
            print("Unknown option, please enter 0-9.")
            continue
        try:
            run_step(choice)
        except FileNotFoundError as exc:
            print(f"\n[!] {exc}")
        except sqlite3.Error as exc:
            print(f"\n[!] Database error: {exc}. Try step 2 (rebuild the database).")
        except KeyboardInterrupt:
            print("\n[!] Interrupted, back to menu.")
        except Exception as exc:   # keep the menu alive on unexpected errors
            print(f"\n[!] {type(exc).__name__}: {exc}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Kazakh NER (KazNERD) project")
    parser.add_argument("--run-all", action="store_true", help="run steps 1-7 non-interactively")
    parser.add_argument("--step", choices=sorted(STEPS), help="run one step and exit")
    args = parser.parse_args(argv)
    ensure_dirs()
    if args.run_all:
        for key in ("1", "2", "3", "4", "5", "6", "7", "9"):
            run_step(key)
    elif args.step:
        run_step(args.step)
    else:
        menu()
    return 0


if __name__ == "__main__":
    sys.exit(main())
