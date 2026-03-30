import os
import torch
import pandas as pd
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import evaluate
from tqdm import tqdm

# Configuration
MODELS = ["Qwen/Qwen1.5-0.5B", "Qwen/Qwen1.5-1.8B", "Qwen/Qwen1.5-4B"]
ADAPTER_DIR = "./qwen_results"
DATASET_PATH = "./qwen_tokenized/validation" # Using validation set for testing
rouge = evaluate.load("rouge")

# def generate_summary(model, tokenizer, text):
#     inputs = tokenizer(f"Summarize: {text}", return_tensors="pt", max_length=512, truncation=True).to("cuda")
#     with torch.no_grad():
#         outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
#     return tokenizer.decode(outputs[0], skip_special_tokens=True)
def generate_summary(model, tokenizer, text):
    # Prompting Qwen correctly for summarization
    prompt = f"<|im_start|>system\nYou are a helpful assistant that summarizes news.<|im_end|>\n<|im_start|>user\nSummarize this article in 3 bullet points: {text}<|im_end|>\n<|im_start|>assistant\n"
    
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=128, 
            num_beams=4,           # <--- Beam search improves ROUGE significantly
            length_penalty=2.0,    # <--- Encourages more complete summaries
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).split("assistant\n")[-1]

results = []

for base_id in MODELS:
    name = base_id.split("/")[-1]
    adapter_path = f"{ADAPTER_DIR}/{name}_final_lora"
    print(f"\n🧐 Evaluating {name}...")

    # 1. Load Model + Adapter
    tokenizer = AutoTokenizer.from_pretrained(base_id)
    base_model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    
    # 2. Load Validation Sample (using first 50 for speed; increase for final paper)
    val_data = load_from_disk(DATASET_PATH).select(range(50))
    
    predictions = []
    references = []

    for example in tqdm(val_data):
        # We need the original text, not tokens, for ROUGE
        # If your step2 saved only tokens, we decode them here
        text = tokenizer.decode(example['input_ids'], skip_special_tokens=True)
        ref = tokenizer.decode(example['labels'], skip_special_tokens=True)
        
        pred = generate_summary(model, tokenizer, text)
        predictions.append(pred)
        references.append(ref)

    # 3. Calculate Scores
    score = rouge.compute(predictions=predictions, references=references)
    score['model'] = name
    results.append(score)
    
    # Cleanup to save VRAM
    del model, base_model
    torch.cuda.empty_cache()

# Save results
df = pd.DataFrame(results)
df.to_csv("qwen_rouge_results.csv", index=False)
print("\n✅ ROUGE evaluation complete! Scores saved to qwen_rouge_results.csv")