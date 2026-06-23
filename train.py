from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np

# 1. Load dataset (SST-2 is a clean starter: sentiment on sentences)
dataset = load_dataset("stanfordnlp/sst2")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch["sentence"], truncation=True, padding="max_length", max_length=128)

tokenized = dataset.map(tokenize, batched=True)

# 2. Load model + wrap with LoRA
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,               # rank — keep low to start
    lora_alpha=16,
    lora_dropout=0.1,
    target_modules=["q_lin", "v_lin"]   # DistilBERT attention layers
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()   # should show ~1% of params — that's the point

# 3. Define metrics
accuracy = evaluate.load("accuracy")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return accuracy.compute(predictions=preds, references=labels)

# 4. Train
args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()

# 5. Save + push to Hub
model.push_to_hub("YOUR-HF-USERNAME/distilbert-lora-sst2")
tokenizer.push_to_hub("YOUR-HF-USERNAME/distilbert-lora-sst2")
print("Done! Model is live on Hugging Face Hub.")