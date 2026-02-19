import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser(description="Download a HuggingFace model")
    parser.add_argument("--repo-id", default="LiquidAI/LFM2.5-1.2B-Thinking-MLX-8bit", help="HuggingFace repository ID")
    parser.add_argument("--local-dir", default="./models/LFM2.5-1.2B-Thinking-MLX-8bit", help="Local directory to save the model")
    
    args = parser.parse_args()
    
    print(f"Downloading {args.repo_id} → {args.local_dir}")
    path = snapshot_download(repo_id=args.repo_id, local_dir=args.local_dir)
    print(f"Done. Model saved to: {path}")

if __name__ == "__main__":
    main()
