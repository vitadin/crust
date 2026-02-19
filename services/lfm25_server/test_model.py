from mlx_lm import load, generate

MODEL_PATH = "./models/LFM2.5-1.2B-Thinking-MLX-8bit"

print("Loading model...")
model, tokenizer = load(MODEL_PATH)

prompt = "What is 15 + 27? Think step by step."
messages = [{"role": "user", "content": prompt}]

formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

print(f"\nPrompt: {prompt}\n")
print("Response:")
response = generate(model, tokenizer, prompt=formatted, max_tokens=512, verbose=True)
