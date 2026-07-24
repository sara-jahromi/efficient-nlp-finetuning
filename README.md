# Efficient NLP Fine-Tuning with LoRA and PEFT

Fine-tuning `distilbert-base-uncased` for sentiment classification using 
parameter-efficient LoRA via the PEFT library. Only ~1% of model parameters 
are trained, making this approach practical for resource-constrained settings.


## Model on Hugging Face Hub
https://huggingface.co/Sara-1990/distilbert-lora-sst2


## What this project covers
- Fine-tuning a pre-trained transformer with LoRA (Low-Rank Adaptation)
- Using Hugging Face `transformers`, `peft`, `datasets`, and `evaluate`
- Tracking eval accuracy across epochs
- Publishing a model to the Hugging Face Hub with a model card
- Building a Gradio inference demo

## Results
| Metric | Value |
|--------|-------|
| Dataset | SST-2 |
| Base model | distilbert-base-uncased |
| Trainable params | ~0.8% (LoRA r=8) |

## Setup
```bash
git clone https://github.com/sara-jahromi/efficient-nlp-finetuning.git
cd efficient-nlp-finetuning
pip install -r requirements.txt
python train.py
```

## Tech stack
Python · Hugging Face Transformers · PEFT · Datasets · Gradio

## Author
Sara — PhD in Electrical Engineering (NJIT), specializing in Reliable ML, distributed ML systems, and Semantic Information Retrieval.
