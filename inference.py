import argparse
import os

import torch
import yaml
from transformers import AutoTokenizer

# --- Import your custom model implementation ---
try:
    from custom_llama_model import CustomLlamaConfig, CustomLlamaModel 
except ImportError:
    print("Error: Could not import CustomLlamaConfig and CustomLlamaModel.")
    print("Please ensure 'custom_llama_model.py' is in the current directory or Python path.")
    exit()

# --- Load Configuration ---
def load_config(config_path='config.yaml'):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

# Load default configuration
config = load_config()

# --- Configuration & Hyperparameters ---
TOKENIZER_PATH = config['paths']['tokenizer_path']
OUTPUT_DIR = config['paths']['output_dir']
FINAL_MODEL_FILE = os.path.join(OUTPUT_DIR, config['paths']['final_model_file'])

# Inference Defaults
DEFAULT_PROMPT = config['inference']['default_prompt']
DEFAULT_MAX_NEW_TOKENS = config['inference']['max_new_tokens']

# --- Inference Function ---
def run_inference(model, tokenizer, device, prompt="太阳从", max_new_tokens=50,
                  temperature=1.0, top_k=0, top_p=1.0, repetition_penalty=1.0):
    print(f"\n--- Running Inference ---")
    print(f"Prompt: '{prompt}'")
    print(f"Parameters: temp={temperature}, top_k={top_k}, top_p={top_p}, rep_penalty={repetition_penalty}")
    model.config.vocab_size = len(tokenizer.get_vocab())
    model.eval() # Set model to evaluation mode

    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    generated_ids = input_ids

    # --- Generation Loop with Sampling ---
    print("Generated: ", end="")
    with torch.no_grad():
        for i in range(max_new_tokens):
            outputs = model(generated_ids)
            next_token_logits = outputs[:, -1, :] # Shape: (batch_size, vocab_size)

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(generated_ids[0].tolist()):
                     if next_token_logits[0, token_id] > 0:
                         next_token_logits[0, token_id] /= repetition_penalty
                     else: 
                         next_token_logits[0, token_id] *= repetition_penalty


            # Apply temperature
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply Top-K filtering
            if top_k > 0:
                top_k_values, top_k_indices = torch.topk(next_token_logits, top_k)
                # Create a mask for filtering
                mask = torch.full_like(next_token_logits, -float('inf'))
                mask.scatter_(1, top_k_indices, top_k_values)
                next_token_logits = mask

            # Apply Top-P (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Scatter sorted tensors to original indexing
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                next_token_logits[indices_to_remove] = -float('inf')

            # Sample from the filtered distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1)

            # Append the new token
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            # Decode and print the new token
            try:
                new_token_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)
                print(new_token_text, end="", flush=True)
            except Exception as e:
                print(f"<Decode Error: {e}>", end="")

            # Stop if EOS token is generated
            if next_token_id.item() == tokenizer.eos_token_id:
                print("\n[EOS token generated]")
                break

    print()
    print("--- Inference Complete ---")

# --- Main execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference with a custom Llama model.")
    parser.add_argument('--gpu_id', type=int, default=2, help='GPU ID to use (e.g., 0, 1, 2)')
    parser.add_argument('--tokenizer_path', type=str, default=TOKENIZER_PATH, help='Path to the tokenizer directory')
    parser.add_argument('--checkpoint_path', type=str, default=FINAL_MODEL_FILE, help='Path to the model checkpoint file')
    parser.add_argument('--prompt', type=str, default=DEFAULT_PROMPT, help='Prompt for generation')
    parser.add_argument('--max_new_tokens', type=int, default=DEFAULT_MAX_NEW_TOKENS, help='Maximum number of new tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8, help='Sampling temperature (e.g., 0.8)')
    parser.add_argument('--top_k', type=int, default=50, help='Top-K filtering (e.g., 50)')
    parser.add_argument('--top_p', type=float, default=0.9, help='Nucleus sampling (Top-P) (e.g., 0.9)')
    parser.add_argument('--repetition_penalty', type=float, default=1.1, help='Repetition penalty (e.g., 1.1)')
    
    args = parser.parse_args()

    # 1. Device Selection
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Tokenizer
    print(f"Loading tokenizer from: {args.tokenizer_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        print("Tokenizer loaded.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        exit()

    # 3. Load Checkpoint and Config
    checkpoint_path = args.checkpoint_path
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        # Try the default checkpoint if final wasn't found and wasn't specified
        default_ckpt = os.path.join(OUTPUT_DIR, "custom_llama_checkpoint.pt")
        if checkpoint_path == FINAL_MODEL_FILE and os.path.exists(default_ckpt):
            print(f"Trying default checkpoint: {default_ckpt}")
            checkpoint_path = default_ckpt
        else:
            exit()

    print(f"Loading checkpoint from: {checkpoint_path}")
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False) # Load to CPU first
        model_config = checkpoint['model_config'] # Correct key for model configuration
        print("Checkpoint loaded.")
        print("Model Configuration from checkpoint:")
        # Ensure model_config is the actual config object or can be used to instantiate one
        if not isinstance(model_config, CustomLlamaConfig):
            # This case assumes 'model_config' might be a dict, adjust if it's always an object
            print("Warning: 'model_config' in checkpoint is not a CustomLlamaConfig object. Assuming it's a dict for instantiation.")
            # If it's a dict of parameters for CustomLlamaConfig:
            # model_config = CustomLlamaConfig(**model_config) 
            # If it's already an object, the above line is not needed and might be an error depending on what's stored.
            # For now, we will assume it is a CustomLlamaConfig object as saved by train_distributed.py
            # If it was saved as a dict, the line above should be uncommented and adapted.
            pass # Assuming model_config is already a CustomLlamaConfig instance

        print(model_config)
    except Exception as e:
        print(f"Error loading checkpoint or model_config: {e}")
        exit()

    # 4. Initialize Model
    # Ensure CustomLlamaModel can accept the model_config object directly
    model = CustomLlamaModel(model_config).to(device)
    print(f"Model instantiated on {device}.")

    # 5. Load Model State Dict
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model weights loaded.")
    except Exception as e:
        print(f"Error loading model state_dict: {e}")
        # Potentially add strict=False if needed, but investigate mismatches first
        exit()

    # 6. Run Inference
    run_inference(model, tokenizer, device, args.prompt, args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty) 
