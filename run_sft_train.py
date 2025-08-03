
import json
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer

model_name = "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
output_dir = "slm-finetuned"
sft_data_file = "sft_training_data.json"

with open(sft_data_file, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list([
    {"text": f"### Instruction:\n{item['instruction']}\n### Response:\n{item['output']}"}
    for item in data[:50]
])

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def preprocess(example):
    return tokenizer(example["text"], truncation=True, max_length=512)

tokenized = dataset.map(preprocess)

args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    per_device_train_batch_size=4,
    save_steps=500,
    logging_steps=100,
    learning_rate=2e-5,
    fp16=True
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
)
trainer.train()
tokenizer.save_pretrained(output_dir)
model.save_pretrained(output_dir)
print("SFT training complete.")
