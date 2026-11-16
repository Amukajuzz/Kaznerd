"""Fine-tuning a multilingual transformer (default: XLM-RoBERTa base) for KazNERD.

Needs a CUDA GPU for reasonable speed (≈1 h per epoch on CPU is not practical).

    python -m src.transformer_ner --model xlm-roberta-base --epochs 3 --batch-size 16

After training the model is saved to models/transformer/ and evaluated on the test split;
its scores, predictions and errors are written to the same CSV files and SQLite tables as
the classical models, so the menu (options 5, 7, 8, 9) picks it up automatically.
"""
import argparse
import random

import numpy as np

import config

LABEL_PAD = -100


def build_label_list(sentences):
    labels = sorted({t for _, _, tags in sentences for t in tags} - {"O"})
    return ["O"] + labels


class NERDataset:
    """Tokenizes pre-split words and aligns IOB2 labels to the first sub-token of each word."""

    def __init__(self, sentences, tokenizer, label2id, max_length=256):
        self.items = []
        for _, tokens, tags in sentences:
            enc = tokenizer(tokens, is_split_into_words=True, truncation=True, max_length=max_length)
            labels, prev = [], None
            for wid in enc.word_ids():
                if wid is None or wid == prev:
                    labels.append(LABEL_PAD)
                else:
                    labels.append(label2id[tags[wid]])
                prev = wid
            enc["labels"] = labels
            self.items.append(dict(enc))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class TransformerTagger:
    """Inference wrapper with the same predict(token_lists) interface as the other models."""

    name = "transformer"

    def __init__(self, model_dir, batch_size=32, max_length=256, device=None):
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.batch_size, self.max_length = batch_size, max_length
        self.description = f"Fine-tuned {self.model.config.name_or_path or 'transformer'}"

    def predict(self, token_lists):
        out = []
        for b in range(0, len(token_lists), self.batch_size):
            batch = token_lists[b:b + self.batch_size]
            non_empty = [t if t else ["."] for t in batch]
            enc = self.tokenizer(non_empty, is_split_into_words=True, truncation=True,
                                 max_length=self.max_length, padding=True, return_tensors="pt")
            with self.torch.no_grad():
                logits = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits
            pred_ids = logits.argmax(-1).cpu().numpy()
            for k, tokens in enumerate(batch):
                tags = ["O"] * len(tokens)
                seen = set()
                for pos, wid in enumerate(enc.word_ids(k)):
                    if wid is not None and wid not in seen and wid < len(tokens):
                        tags[wid] = self.id2label[int(pred_ids[k, pos])]
                        seen.add(wid)
                out.append(tags)
        return out


def train(args):
    import torch
    from seqeval.metrics import f1_score
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, Trainer, TrainingArguments)

    from src import data_loader, pipeline
    from src.utils import ResourceMonitor

    random.seed(config.SEED)
    np.random.seed(config.SEED)
    torch.manual_seed(config.SEED)

    sentences = data_loader.load_sentences()
    train_s, valid_s, test_s = sentences["train"], sentences["valid"], sentences["test"]
    if args.max_train_sentences:
        train_s = random.Random(config.SEED).sample(train_s, min(args.max_train_sentences, len(train_s)))
    if args.max_eval_sentences:
        valid_s = valid_s[:args.max_eval_sentences]
        test_s = test_s[:args.max_eval_sentences]

    labels = build_label_list(sentences["train"])
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = dict(enumerate(labels))

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model, num_labels=len(labels), id2label=id2label, label2id=label2id,
        ignore_mismatched_sizes=True)   # allows starting from a checkpoint with another label set

    train_ds = NERDataset(train_s, tokenizer, label2id, args.max_length)
    valid_ds = NERDataset(valid_s, tokenizer, label2id, args.max_length)

    def compute_metrics(eval_pred):
        logits, gold = eval_pred
        pred = logits.argmax(-1)
        y_true, y_pred = [], []
        for p_row, g_row in zip(pred, gold):
            keep = g_row != LABEL_PAD
            y_true.append([id2label[int(i)] for i in g_row[keep]])
            y_pred.append([id2label[int(i)] for i in p_row[keep]])
        return {"f1": f1_score(y_true, y_pred)}

    ckpt_dir = config.MODELS_DIR / "transformer_checkpoints"
    total_steps = int(np.ceil(len(train_ds) / args.batch_size) * args.epochs)
    warmup_steps = int(0.1 * total_steps)       # 10 % linear warm-up (works on transformers 4.x and 5.x)
    training_args = TrainingArguments(
        output_dir=str(ckpt_dir), learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size * 2,
        weight_decay=0.01, warmup_steps=warmup_steps, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", save_total_limit=1,
        logging_steps=200, fp16=torch.cuda.is_available() and not args.no_fp16,
        report_to="none", seed=config.SEED, dataloader_num_workers=args.workers,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=valid_ds,
                      data_collator=DataCollatorForTokenClassification(tokenizer),
                      processing_class=tokenizer, compute_metrics=compute_metrics)
    with ResourceMonitor() as mon:
        trainer.train()
    out_dir = config.MODELS_DIR / "transformer"
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    gpu_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if torch.cuda.is_available() else None
    print(f"Training done in {mon.seconds / 60:.1f} min; model saved to {out_dir}")

    tagger = TransformerTagger(out_dir, batch_size=args.batch_size * 2, max_length=args.max_length)
    tagger.description = f"Fine-tuned {args.model} ({args.epochs} epochs, lr {args.lr})"
    info = {"train_sentences": len(train_s), "n_features": sum(p.numel() for p in model.parameters()),
            "train_time_s": mon.seconds, "peak_mem_mb": gpu_mb if gpu_mb is not None else mon.peak_mb,
            "memory_note": "GPU peak allocated" if gpu_mb is not None else "process RSS"}
    pipeline.evaluate_and_record(tagger, test_s, info)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="xlm-roberta-base",
                   help="HuggingFace model id or local folder (e.g. xlm-roberta-large, bert-base-multilingual-cased)")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--max-train-sentences", type=int, default=None, help="subsample train (quick runs)")
    p.add_argument("--max-eval-sentences", type=int, default=None, help="truncate valid/test (smoke tests only)")
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--no-fp16", action="store_true")
    return p.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
