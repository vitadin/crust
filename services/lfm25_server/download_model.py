from huggingface_hub import snapshot_download

MODEL_ID = "LiquidAI/LFM2.5-1.2B-Thinking-MLX-8bit"
LOCAL_DIR = "./models/LFM2.5-1.2B-Thinking-MLX-8bit"

print(f"Downloading {MODEL_ID} → {LOCAL_DIR}")
path = snapshot_download(repo_id=MODEL_ID, local_dir=LOCAL_DIR)
print(f"Done. Model saved to: {path}")
