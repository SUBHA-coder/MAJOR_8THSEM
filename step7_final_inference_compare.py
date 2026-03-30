import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Configuration - Raw Models Only
MODELS = {
    "0.5B": "Qwen/Qwen1.5-0.5B",
    "1.8B": "Qwen/Qwen1.5-1.8B",
    "4B":   "Qwen/Qwen1.5-4B"
}

# The actual Emissions you calculated earlier (kg CO2)
EMISSIONS = {
    "0.5B": 0.0039,
    "1.8B": 0.0173,
    "4B":   0.0379
}

input_text = """The space agency NASA has successfully landed its latest rover on the surface of Mars. 
The mission, which cost approximately $2.7 billion, aims to search for signs of ancient 
microbial life and collect rock samples that will eventually be returned to Earth. 
Engineers at the Jet Propulsion Laboratory celebrated as the first high-resolution 
images beamed back across the solar system, showing a dusty, cratered landscape."""

def get_zero_shot_summary(model_id):
    print(f"🔄 Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )

    # Professional Zero-Shot Prompt
    messages = [
        {"role": "system", "content": "You are a professional news editor. Summarize the text in one clear sentence."},
        {"role": "user", "content": input_text}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_new_tokens=50,
            do_sample=False, 
            repetition_penalty=1.1
        )

    # Extract response
    generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)]
    summary = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    del model
    torch.cuda.empty_cache()
    return summary.strip()

# Execution
results = {}
for label, m_id in MODELS.items():
    results[label] = get_zero_shot_summary(m_id)

print("\n" + "="*80)
print("🌍 FINAL THESIS TABLE: QUALITY VS. ENVIRONMENTAL IMPACT")
print("="*80)
print(f"{'Model':<10} | {'Carbon (kg)':<12} | {'Summary Result'}")
print("-" * 80)

for label, summary in results.items():
    carbon = EMISSIONS[label]
    print(f"{label:<10} | {carbon:<12} | {summary}")
    print("-" * 80)