from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("./results/checkpoint-12630")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

model.push_to_hub("Sara-1990/distilbert-lora-sst2")
tokenizer.push_to_hub("Sara-1990/distilbert-lora-sst2")
print("Done!")