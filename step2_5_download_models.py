import os
from huggingface_hub import snapshot_download

# List of models we need
MODELS = ["Qwen/Qwen1.5-0.5B", "Qwen/Qwen1.5-1.8B", "Qwen/Qwen1.5-4B"]

print("📥 Starting robust model download...")

for model in MODELS:
    print(f"\nChecking/Downloading {model}...")
    success = False
    while not success:
        try:
            # This will resume where it left off if it fails
            snapshot_download(
                repo_id=model,
                repo_type="model",
                resume_download=True,
                max_workers=4  # Parallel downloading
            )
            print(f"✅ {model} is fully downloaded!")
            success = True
        except Exception as e:
            print(f"❌ Download interrupted for {model}. Retrying in 5 seconds... Error: {e}")
            import time
            time.sleep(5)

print("\n🚀 All models are ready! Now you can run Step 3 safely.")