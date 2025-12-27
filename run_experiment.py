import os
import numpy as np
import evaluate
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    TrainingArguments,
    Trainer,
)

from experiment_config import ExperimentConfig
from utils import set_global_seed, ensure_dir, format_float
from modeling_peft import apply_lora_seqcls, apply_bitfit, print_trainable_summary

# --- Make HF downloading more tolerant (optional but helpful) ---
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ETAG_TIMEOUT"] = "60"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"


def build_datasets(cfg: ExperimentConfig, seed: int):
    ds = load_dataset(cfg.dataset_name)

    train_ds = ds["train"].shuffle(seed=seed)
    eval_ds = ds["validation"].shuffle(seed=seed) if "validation" in ds else ds["test"].shuffle(seed=seed)

    if cfg.train_size is not None:
        train_ds = train_ds.select(range(min(cfg.train_size, len(train_ds))))
    if cfg.eval_size is not None:
        eval_ds = eval_ds.select(range(min(cfg.eval_size, len(eval_ds))))

    return train_ds, eval_ds


def tokenize_datasets(cfg: ExperimentConfig, train_ds, eval_ds, tokenizer):
    def tok(batch):
        return tokenizer(
            batch[cfg.text_col],
            truncation=True,
            max_length=cfg.max_length,
        )

    train_ds = train_ds.map(tok, batched=True, remove_columns=[cfg.text_col])
    eval_ds = eval_ds.map(tok, batched=True, remove_columns=[cfg.text_col])
    return train_ds, eval_ds


def build_model(cfg: ExperimentConfig, method: str):
    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=cfg.num_labels)

    if method == "full":
        print_trainable_summary(model, prefix="[FULL] ")
        return model

    if method == "lora":
        model = apply_lora_seqcls(
            model,
            r=cfg.lora_r,
            alpha=cfg.lora_alpha,
            dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
        )
        print_trainable_summary(model, prefix="[LoRA] ")
        return model

    if method == "bitfit":
        model = apply_bitfit(model)
        print_trainable_summary(model, prefix="[BitFit] ")
        return model

    raise ValueError(f"Unknown method: {method}")


def make_trainer(cfg: ExperimentConfig, method: str, seed: int, train_ds, eval_ds, tokenizer, model, out_dir: str):
    metric_acc = evaluate.load("accuracy")
    metric_f1 = evaluate.load("f1")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = metric_acc.compute(predictions=preds, references=labels)["accuracy"]
        f1 = metric_f1.compute(predictions=preds, references=labels)["f1"]  # binary F1
        return {"accuracy": acc, "f1": f1}

    fp16 = bool(torch.cuda.is_available() and cfg.use_fp16_if_cuda)
    lr = cfg.lr_full if method == "full" else cfg.lr_peft

    # ✅ transformers 4.57.3 uses `eval_strategy` (NOT evaluation_strategy)
    args = TrainingArguments(
        output_dir=out_dir,
        eval_strategy=cfg.eval_strategy,         # <- fixed
        save_strategy=cfg.eval_strategy,
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=lr,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=min(cfg.batch_size * 2, 64),
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        fp16=fp16,
        report_to=cfg.report_to,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        seed=seed,
        data_seed=seed,
        save_total_limit=1,
    )

    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )
    return trainer


def summarize_results(all_results: dict):
    print("\n" + "=" * 70)
    print("FINAL SUMMARY (mean ± std over seeds)")
    print("=" * 70)
    header = f"{'Method':10s} | {'Acc(mean)':10s} {'Acc(std)':10s} | {'F1(mean)':10s} {'F1(std)':10s}"
    print(header)
    print("-" * len(header))

    for method, runs in all_results.items():
        accs = np.array([r["accuracy"] for r in runs], dtype=np.float32)
        f1s = np.array([r["f1"] for r in runs], dtype=np.float32)
        print(
            f"{method:10s} | "
            f"{format_float(float(accs.mean())):10s} {format_float(float(accs.std())):10s} | "
            f"{format_float(float(f1s.mean())):10s} {format_float(float(f1s.std())):10s}"
        )
    print("=" * 70 + "\n")


def main():
    cfg = ExperimentConfig()
    ensure_dir(cfg.output_root)

    print("Device:", "cuda" if torch.cuda.is_available() else "cpu")
    print("Transformers version check (runtime):")
    try:
        import transformers
        print("transformers =", transformers.__version__)
    except Exception:
        pass

    print("Config:", cfg)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    all_results = {m: [] for m in cfg.methods}

    for method in cfg.methods:
        for seed in cfg.seeds:
            print("\n" + "#" * 70)
            print(f"RUN: method={method} | seed={seed}")
            print("#" * 70)

            set_global_seed(seed, deterministic=cfg.deterministic)

            train_ds, eval_ds = build_datasets(cfg, seed=seed)
            train_ds, eval_ds = tokenize_datasets(cfg, train_ds, eval_ds, tokenizer)

            model = build_model(cfg, method=method)

            run_dir = os.path.join(cfg.output_root, f"{method}_seed{seed}")
            ensure_dir(run_dir)

            trainer = make_trainer(
                cfg=cfg,
                method=method,
                seed=seed,
                train_ds=train_ds,
                eval_ds=eval_ds,
                tokenizer=tokenizer,
                model=model,
                out_dir=run_dir,
            )

            trainer.train()
            metrics = trainer.evaluate()

            acc = float(metrics.get("eval_accuracy", 0.0))
            f1 = float(metrics.get("eval_f1", 0.0))
            all_results[method].append({"seed": seed, "accuracy": acc, "f1": f1})

            print(f"Result: acc={acc:.4f}, f1={f1:.4f}")

    summarize_results(all_results)


if __name__ == "__main__":
    main()
