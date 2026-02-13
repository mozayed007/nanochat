"""
Query Script for Live Memory (Phase 3)

Goal: Query a frozen model using a pre-populated memory state.
Usage: python -m scripts.query --model_path outputs/d8/model_000100.pt --state_path cairo_memory_state.pt
"""

import os
import torch
import argparse
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import build_model
from nanochat.common import print0

def query(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    print0(f"Loading model from {args.model_path}...")
    checkpoint_dir = os.path.dirname(args.model_path)
    step = int(os.path.basename(args.model_path).split('_')[1].split('.')[0])
    model, tokenizer, _ = build_model(checkpoint_dir, step, torch.device(device), phase="eval")
    model.eval()

    # 2. Load State
    print0(f"Loading memory state from {args.state_path}...")
    state = torch.load(args.state_path, map_location=device)
    print0(f"Loaded state with {len(state.get('traces', []))} traces.")

    # 3. Interactive Loop
    print0("\nReady for queries! (Type 'exit' to quit)")
    print0("-" * 50)
    
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Prepare input
        # Note: We do NOT prepend BOS here if we assume this is a continuation
        # But for a fresh query, BOS is usually good. Let's stick to standard encoding.
        tokens = tokenizer.encode(user_input, prepend="<|bos|>")
        tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        # Generate
        print0("Assistant: ", end="", flush=True)
        
        # We need a custom generate loop that passes 'state'
        # The standard model.generate() in gpt.py is too simple (doesn't take state arg yet)
        # So we write a simple one here or rely on forward()
        
        # For simplicity, let's just generate 20 tokens
        curr_tokens = tokens
        with torch.inference_mode():
            for _ in range(args.max_new_tokens):
                # Forward with state
                logits, _ = model(curr_tokens, state=state, return_state=True)
                
                # Greedy decode
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                
                # Print token
                decoded = tokenizer.decode(next_token[0].tolist())
                print(decoded, end="", flush=True)
                
                # Append
                curr_tokens = torch.cat([curr_tokens, next_token], dim=1)
                
                if next_token.item() == tokenizer.eot_token_id: # Assuming EOT exists/is handled
                    break
        print("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to pretrained model checkpoint")
    parser.add_argument("--state_path", type=str, required=True, help="Path to memory state file")
    parser.add_argument("--max_new_tokens", type=int, default=50, help="Max tokens to generate")
    args = parser.parse_args()
    
    query(args)
