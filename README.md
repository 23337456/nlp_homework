# SST-2 Sentiment Classification with Parameter-Efficient Fine-Tuning

This repository contains experiments for sentiment classification on the SST-2 dataset,
comparing different fine-tuning strategies:

- Full Fine-Tuning
- LoRA (Low-Rank Adaptation)
- BitFit

## Dataset
We use the SST-2 dataset provided by Hugging Face Datasets:
`stanfordnlp/sst2`.

The dataset is automatically downloaded and cached when running the code.

## Environment
- Python >= 3.9
- PyTorch
- transformers >= 4.57
- datasets
- peft
- evaluate

Install dependencies:
```bash
pip install -r requirements.txt
