-- KazNERD NER project — SQLite schema
PRAGMA foreign_keys = ON;

-- Reference table: the 25 KazNERD entity classes
CREATE TABLE IF NOT EXISTS entity_types (
    type_code   TEXT PRIMARY KEY,
    description TEXT NOT NULL
);

-- Corpus: one row per sentence
CREATE TABLE IF NOT EXISTS sentences (
    sentence_id INTEGER PRIMARY KEY,
    split       TEXT    NOT NULL CHECK (split IN ('train', 'valid', 'test')),
    n_tokens    INTEGER NOT NULL,
    text        TEXT    NOT NULL
);

-- Corpus: one row per token, with preprocessing results
CREATE TABLE IF NOT EXISTS tokens (
    sentence_id INTEGER NOT NULL REFERENCES sentences (sentence_id),
    position    INTEGER NOT NULL,
    token       TEXT    NOT NULL,
    lower       TEXT    NOT NULL,
    stem        TEXT    NOT NULL,
    lemma       TEXT    NOT NULL,
    is_stopword INTEGER NOT NULL CHECK (is_stopword IN (0, 1)),
    tag         TEXT    NOT NULL,          -- gold IOB2 tag
    PRIMARY KEY (sentence_id, position)
) WITHOUT ROWID;

-- Gold entity spans (end_pos is exclusive)
CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY,
    sentence_id INTEGER NOT NULL REFERENCES sentences (sentence_id),
    type_code   TEXT    NOT NULL REFERENCES entity_types (type_code),
    start_pos   INTEGER NOT NULL,
    end_pos     INTEGER NOT NULL,
    n_tokens    INTEGER NOT NULL,
    text        TEXT    NOT NULL
);

-- Context keywords per entity type (TF-IDF over words around mentions)
CREATE TABLE IF NOT EXISTS keywords (
    type_code TEXT    NOT NULL REFERENCES entity_types (type_code),
    rank      INTEGER NOT NULL,
    term      TEXT    NOT NULL,
    score     REAL    NOT NULL,
    PRIMARY KEY (type_code, rank)
);

-- Trained models and their resource usage
CREATE TABLE IF NOT EXISTS models (
    model_id        INTEGER PRIMARY KEY,
    name            TEXT UNIQUE NOT NULL,
    description     TEXT,
    train_sentences INTEGER,
    n_features      INTEGER,
    train_time_s    REAL,
    predict_time_s  REAL,
    peak_mem_mb     REAL,
    created_at      TEXT NOT NULL
);

-- Overall test-set scores (one row per model)
CREATE TABLE IF NOT EXISTS model_scores (
    model_id        INTEGER PRIMARY KEY REFERENCES models (model_id),
    accuracy        REAL,
    precision_micro REAL,
    recall_micro    REAL,
    f1_micro        REAL,
    f1_macro        REAL
);

-- Per-class test-set scores
CREATE TABLE IF NOT EXISTS class_scores (
    model_id  INTEGER NOT NULL REFERENCES models (model_id),
    type_code TEXT    NOT NULL REFERENCES entity_types (type_code),
    precision REAL, recall REAL, f1 REAL, support INTEGER,
    PRIMARY KEY (model_id, type_code)
);

-- Entities predicted on the test split
CREATE TABLE IF NOT EXISTS predicted_entities (
    pred_id     INTEGER PRIMARY KEY,
    model_id    INTEGER NOT NULL REFERENCES models (model_id),
    sentence_id INTEGER NOT NULL REFERENCES sentences (sentence_id),
    type_code   TEXT    NOT NULL REFERENCES entity_types (type_code),
    start_pos   INTEGER NOT NULL,
    end_pos     INTEGER NOT NULL,
    text        TEXT    NOT NULL
);

-- Span-level errors on the test split
CREATE TABLE IF NOT EXISTS errors (
    error_id    INTEGER PRIMARY KEY,
    model_id    INTEGER NOT NULL REFERENCES models (model_id),
    sentence_id INTEGER NOT NULL REFERENCES sentences (sentence_id),
    error_type  TEXT    NOT NULL CHECK (error_type IN
                ('missed', 'spurious', 'type_error', 'boundary_error', 'type_and_boundary')),
    gold_type   TEXT REFERENCES entity_types (type_code),
    pred_type   TEXT REFERENCES entity_types (type_code),
    gold_text   TEXT,
    pred_text   TEXT,
    context     TEXT
);

-- Texts typed by the user in the console and the entities found
CREATE TABLE IF NOT EXISTS user_queries (
    query_id   INTEGER PRIMARY KEY,
    model_id   INTEGER REFERENCES models (model_id),
    input_text TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS user_query_entities (
    id        INTEGER PRIMARY KEY,
    query_id  INTEGER NOT NULL REFERENCES user_queries (query_id),
    type_code TEXT    NOT NULL REFERENCES entity_types (type_code),
    text      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tokens_lower      ON tokens (lower);
CREATE INDEX IF NOT EXISTS idx_entities_type     ON entities (type_code, text);
CREATE INDEX IF NOT EXISTS idx_entities_sentence ON entities (sentence_id);
CREATE INDEX IF NOT EXISTS idx_pred_lookup       ON predicted_entities (model_id, sentence_id, start_pos, end_pos);
CREATE INDEX IF NOT EXISTS idx_errors_model      ON errors (model_id, error_type);
