"""Central configuration: paths, dataset URLs and experiment settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# --- directories -----------------------------------------------------------
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"            # original KazNERD CoNLL files
UD_DIR = DATA_DIR / "ud_kazakh"       # UD Kazakh-KTB (gold lemmas)
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
RESULTS_DIR = REPORTS_DIR / "results"
SQL_DIR = ROOT / "sql"
DOCS_DIR = ROOT / "docs"

DB_PATH = DATA_DIR / "kaznerd.db"
TOKENS_PARQUET = PROCESSED_DIR / "tokens.parquet"
TOKENS_CSV = PROCESSED_DIR / "tokens.csv"
STOPWORDS_FILE = DATA_DIR / "stopwords_kk.txt"

# --- data sources ----------------------------------------------------------
SPLITS = ("train", "valid", "test")
KAZNERD_URL = "https://raw.githubusercontent.com/IS2AI/KazNERD/main/KazNERD/IOB2_{split}.txt"
UD_URLS = {
    "train": "https://raw.githubusercontent.com/UniversalDependencies/UD_Kazakh-KTB/master/kk_ktb-ud-train.conllu",
    "test": "https://raw.githubusercontent.com/UniversalDependencies/UD_Kazakh-KTB/master/kk_ktb-ud-test.conllu",
}

# --- experiment settings ---------------------------------------------------
SEED = 42
CSV_CHUNKSIZE = 200_000           # rows per chunk when reading CSV / inserting into SQLite
LINEAR_TRAIN_SENTENCES = 30_000   # subsample for BoW / TF-IDF / Word2Vec taggers (memory bound)
CRF_MAX_ITERATIONS = 100
CRF_C1, CRF_C2 = 0.1, 0.1
W2V_DIM = 100
W2V_WINDOW = 5
W2V_EPOCHS = 10

# Short English descriptions of the 25 KazNERD classes (from the annotation guidelines)
ENTITY_TYPES = {
    "ADAGE": "Proverbs, sayings and idioms",
    "ART": "Titles of works of art, books, songs, films",
    "CARDINAL": "Numerals that are not another numeric type",
    "CONTACT": "Addresses, phone numbers, e-mails, URLs",
    "DATE": "Absolute or relative dates and periods",
    "DISEASE": "Diseases and medical conditions",
    "EVENT": "Named events: wars, festivals, sports events",
    "FACILITY": "Buildings, airports, roads, bridges",
    "GPE": "Geo-political entities: countries, cities, regions",
    "LANGUAGE": "Named languages",
    "LAW": "Named laws, codes, treaties, conventions",
    "LOCATION": "Non-GPE locations: rivers, mountains, districts",
    "MISCELLANEOUS": "Other named entities",
    "MONEY": "Monetary values with unit",
    "NON_HUMAN": "Named animals and other non-human beings",
    "NORP": "Nationalities, religious or political groups",
    "ORDINAL": "Ordinal numbers",
    "ORGANISATION": "Companies, agencies, institutions",
    "PERCENTAGE": "Percentages",
    "PERSON": "People, including fictional",
    "POSITION": "Job titles and positions",
    "PRODUCT": "Objects, vehicles, foods, brands",
    "PROJECT": "Named projects, programmes, initiatives",
    "QUANTITY": "Measurements: weight, distance, area",
    "TIME": "Times shorter than a day",
}
