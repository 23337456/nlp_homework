from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    # ===== Core experiment setup =====
    model_name: str = "bert-base-uncased"
    dataset_name: str = "stanfordnlp/sst2"
    text_col: str = "sentence"
    label_col: str = "label"
    num_labels: int = 2

    # ===== Reproducibility =====
    seeds: tuple = (13, 21, 42)   # 论文复现常见：多seed取均值
    deterministic: bool = True

    # ===== Data budget (论文式：可以做 full vs low-resource) =====
    train_size: int | None = 5000   # None 表示用全量；先用 5k 快速复现更稳
    eval_size: int | None = 1000

    # ===== Training hyperparams =====
    max_length: int = 128
    epochs: float = 3.0
    lr_full: float = 2e-5          # full fine-tuning 通常更小
    lr_peft: float = 2e-4          # PEFT 通常更大
    batch_size: int = 16           # 4060 8GB 通常稳
    grad_accum: int = 2            # 等效 batch = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    eval_strategy: str = "epoch"

    # ===== Output =====
    output_root: str = "./outputs_sst2_repro"
    report_to: str = "none"        # 不用 wandb/tensorboard

    # ===== Methods to compare =====
    methods: tuple = ("full", "lora", "bitfit")

    # ===== LoRA hyperparams (paper-like knobs) =====
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("query", "value")  # BERT attention

    # ===== Speed/Memory =====
    use_fp16_if_cuda: bool = True
