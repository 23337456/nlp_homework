import torch
from peft import LoraConfig, get_peft_model, TaskType


def apply_lora_seqcls(
    model,
    r: int = 8,
    alpha: int = 16,
    dropout: float = 0.05,
    target_modules=("query", "value"),
):
    """
    LoRA for sequence classification.
    We freeze base model parameters and train only LoRA adapters (+ classifier head remains trainable by default).
    """
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(target_modules),
        bias="none",
    )
    model = get_peft_model(model, config)
    return model


def apply_bitfit(model):
    """
    BitFit: train only bias terms (+ classifier head weights & bias).
    Implementation: freeze everything, then unfreeze:
      - all parameters with ".bias" in name
      - classifier head (often named "classifier" in BERT seqcls)
    """
    for p in model.parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if name.endswith(".bias"):
            p.requires_grad = True

    # Unfreeze classifier head fully (weights + bias)
    # For BERT sequence classification, typically model.classifier exists.
    if hasattr(model, "classifier"):
        for p in model.classifier.parameters():
            p.requires_grad = True

    return model


def count_trainable_params(model):
    total = 0
    trainable = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def print_trainable_summary(model, prefix=""):
    trainable, total = count_trainable_params(model)
    ratio = 100.0 * trainable / max(total, 1)
    print(f"{prefix}Trainable params: {trainable:,} / {total:,} ({ratio:.2f}%)")
