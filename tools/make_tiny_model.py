"""Build a tiny, randomly initialised BERT + WordPiece tokenizer from the KazNERD corpus.

It is NOT a useful NER model. Its only purpose is to smoke-test the fine-tuning
pipeline (sub-word label alignment, training, evaluation, saving, inference)
without internet access and without a GPU:

    python tools/make_tiny_model.py
    python -m src.transformer_ner --model models/tiny_bert --epochs 1 \
        --max-train-sentences 2000 --max-eval-sentences 500

Delete models/transformer and re-run step 4 afterwards: the smoke run overwrites
the 'transformer' row in the results.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors, trainers  # noqa: E402
from transformers import BertConfig, BertForTokenClassification, PreTrainedTokenizerFast  # noqa: E402

import config  # noqa: E402
from src.data_loader import ensure_raw_data, read_conll  # noqa: E402


def main(vocab_size=8000, n_sentences=30000):
    target = config.MODELS_DIR / "tiny_bert"
    target.mkdir(parents=True, exist_ok=True)
    ensure_raw_data()
    corpus = [" ".join(t) for t, _ in read_conll(config.RAW_DIR / "IOB2_train.txt")][:n_sentences]

    tok = Tokenizer(models.WordPiece(unk_token="[UNK]"))
    tok.normalizer = normalizers.Lowercase()
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.train_from_iterator(corpus, trainers.WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]))
    tok.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]",
                                   cls_token="[CLS]", sep_token="[SEP]", mask_token="[MASK]")
    fast.save_pretrained(str(target))

    cfg = BertConfig(vocab_size=vocab_size, hidden_size=64, num_hidden_layers=2,
                     num_attention_heads=2, intermediate_size=128, max_position_embeddings=512)
    BertForTokenClassification(cfg).save_pretrained(str(target))
    print(f"Tiny model saved to {target}")


if __name__ == "__main__":
    main()
