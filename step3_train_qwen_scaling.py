import os
import torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer, 
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from codecarbon import EmissionsTracker

# 1. Configuration - Consistent with your KIIT Report methodology
MODELS = ["Qwen/Qwen1.5-0.5B", "Qwen/Qwen1.5-1.8B", "Qwen/Qwen1.5-4B"]
DATASET_BASE_PATH = "./qwen_tokenized"
OUTPUT_DIR = "./qwen_results"

# LoRA rank r=8 is consistent with your previous Phi-2 study
lora_config = LoraConfig(
    r=8, 
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

def train_model(model_id):
    model_name = model_id.split("/")[-1]
    print(f"\n🚀 Starting Fine-Tuning for: {model_name}")
    
    tracker = EmissionsTracker(project_name=f"FineTune_{model_name}", output_dir=OUTPUT_DIR)
    tracker.start()
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Updated quantization config to avoid the deprecation warning
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        model = prepare_model_for_kbit_training(model)
        model = get_peft_model(model, lora_config)
        
        # --- FIXED LOADING LOGIC ---
        # We load the splits individually since they were saved individually
        train_data = load_from_disk(os.path.join(DATASET_BASE_PATH, "train"))
        # ----------------------------

        # training_args = TrainingArguments(
        #     output_dir=f"{OUTPUT_DIR}/{model_name}",
        #     per_device_train_batch_size=2, 
        #     gradient_accumulation_steps=4,
        #     warmup_steps=10,
        #     max_steps=100, 
        #     learning_rate=2e-4,
        #     fp16=True,
        #     logging_steps=10,
        #     save_strategy="no",
        #     report_to="none"
        # )
        training_args = TrainingArguments(
              output_dir=f"{OUTPUT_DIR}/{model_name}",
              per_device_train_batch_size=2, 
              gradient_accumulation_steps=4,
              warmup_steps=50,
              max_steps=500,        # <--- Increase from 100 to 500
              learning_rate=5e-5,   # <--- Slightly lower learning rate for stability
              fp16=True,
              logging_steps=10,
              save_strategy="no",
              report_to="none"
          )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)
        )

        trainer.train()
        model.save_pretrained(f"{OUTPUT_DIR}/{model_name}_final_lora")
        print(f"✅ Finished training {model_name}")

    finally:
        tracker.stop()

for m_id in MODELS:
    train_model(m_id)

print("\n📊 All models trained! Check 'emissions.csv' for your Carbon Footprint Analysis.")