# Changelog: Fix Memory Mechanisms (Retrieval & Granularity)

## Date: 2026-02-13

## Author: Trae (AI Assistant)

### Summary

Addressed critical architectural flaws identified in "Second Loop Critique" (Loop 2). The model now uses proper attention for memory retrieval (fixing the "Memory Blurring" issue) and implements a buffering mechanism for ingestion (fixing the "Granularity Mismatch" between batch and interactive modes).

### Changes

#### 1. `nanochat/gpt.py`

* **Modified `HybridTHENAttention`**:
  * **Retrieval**: Replaced `torch.mean` (averaging all traces) with `F.scaled_dot_product_attention`.
    * **Query**: `self.dsa(x)`
    * **Key/Value**: `torch.stack(state['traces'])`
    * **Result**: The model can now "attend" to specific past moments rather than receiving a blurred average of its entire life.
  * **Ingestion (Granularity)**: Added `chunk_size` (default 16) and a `state['buffer']`.
    * Incoming tokens are accumulated in `buffer`.
    * Only when `buffer` size >= `chunk_size` is a new trace created (via mean pooling) and appended to `state['traces']`.
    * **Impact**: Prevents RAM explosion during interactive chat (where tokens arrive 1-by-1) and ensures consistency with batch ingestion.

#### 2. `scripts/base_train.py`

* **Updated TBPTT Logic**: Added logic to detach `state['buffer']` along with `state['traces']` to ensure gradients are properly truncated during stateful pretraining.

### Verification

* **Granularity**: Confirmed that feeding 16 tokens one-by-one now results in **1 trace** (same as feeding 16 tokens at once), whereas previously it would have created 16 traces.
* **Retrieval**: Confirmed that the attention mechanism logic is syntactically correct and uses the standard PyTorch functional API.

### Next Steps

* **Performance Profiling**: Check the speed impact of `torch.stack` and Attention on the CPU during inference with large memory states.
* **Stage 2 (Hardware-Native)**: The current `buffer` is still in RAM/VRAM. The next major step is spilling this to Disk/NVMe.
