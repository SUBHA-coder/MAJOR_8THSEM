# step2_data_prep_qwen.py
import os
from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer

# --- Configuration ---
# Using the 4B tokenizer ensures it covers the smaller models in the same family
MODEL_NAME = "Qwen/Qwen1.5-4B-Chat" 
OUTPUT_DIR = "./qwen_tokenized"

os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load dataset (keeping your 1% logic)
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train[:1%]")
eval_dataset = load_dataset("cnn_dailymail", "3.0.0", split="validation[:1%]")

def preprocess_function(examples):
    # Qwen-specific prompt format (ChatML style is best for Qwen)
    prompts = [f"<|im_start|>user\nSummarize this: {a}<|im_end|>\n<|im_start|>assistant\n" for a in examples["article"]]
    targets = [f"{h}<|im_end|>" for h in examples["highlights"]]
    
    model_inputs = tokenizer(prompts, max_length=512, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=512, truncation=True, padding="max_length")

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_train = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)
tokenized_val = eval_dataset.map(preprocess_function, batched=True, remove_columns=eval_dataset.column_names)

tokenized_train.save_to_disk(os.path.join(OUTPUT_DIR, "train"))
tokenized_val.save_to_disk(os.path.join(OUTPUT_DIR, "validation"))
print("Done! Data ready for Qwen training.")