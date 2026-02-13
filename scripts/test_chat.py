import torch
import torch.nn.functional as F
import os
import sys
from nanochat.tokenizer import get_tokenizer
from nanochat.checkpoint_manager import load_model_from_dir, find_last_step
from nanochat.common import get_base_dir, compute_init

def test_memory():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    base_dir = get_base_dir()
    checkpoints_dir = os.path.join(base_dir, "base_checkpoints")
    # Assuming run name "dummy" or "gpt2-then-local-d12"
    run_name = "dummy"
    run_dir = os.path.join(checkpoints_dir, run_name)
    
    if not os.path.exists(run_dir):
        print(f"Checkpoint directory not found: {run_dir}")
        print("Please ensure training has completed at least one checkpoint save.")
        return

    try:
        step = find_last_step(run_dir)
        print(f"Loading checkpoint from step {step}...")
        model, tokenizer, meta, state = load_model_from_dir(checkpoints_dir, device, phase="eval", model_tag=run_name, step=step, load_state=True)
    except Exception as e:
        print(f"Failed to load checkpoint: {e}")
        return

    print("Model loaded successfully.")
    
    # Init state if None
    if state is None:
        print("No saved state found, initializing new state.")
        state = {'traces': []}
    else:
        print(f"Loaded state with {len(state.get('traces', []))} traces.")

    # Test loop
    conversation = [
        "Nile sunset",
        "Where am I?",
        "What do I like?",
        "قهوة سوداء"
    ]
    
    print("\n--- Starting Conversation Test ---")
    model.eval()
    
    for prompt in conversation:
        print(f"\nUser: {prompt}")
        
        # Tokenize
        ids = tokenizer.encode(prompt, prepend=tokenizer.get_bos_token_id())
        ids = torch.tensor([ids], dtype=torch.long, device=device)
        
        # Forward with state
        with torch.no_grad():
            # Pass return_state=True to get updated state
            logits, state = model(ids, state=state, return_state=True)
            
        # Inspect state
        traces = state.get('traces', [])
        print(f"System: Processed. Memory traces count: {len(traces)}")
        if traces:
            last_trace = traces[-1]
            print(f"Last trace mean: {last_trace.mean().item():.4f}")
            
        # Check consolidation
        if 'consolidated' in state:
            print(f"Consolidated memory norm: {torch.norm(state['consolidated']).item():.4f}")

    print("\n--- Test Complete ---")
    print("Memory persistence verified if traces count increased.")

if __name__ == "__main__":
    test_memory()
