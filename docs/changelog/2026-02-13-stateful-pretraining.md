# Changelog: Stateful Pretraining Implementation

## Date: 2026-02-13

## Author: Trae (AI Assistant)

### Summary

Implemented "Stage 1: Stateful Pretraining" to bridge the gap between training (Phase 1) and inference (Phase 2/3). This ensures the `THENGPT` model learns to utilize its memory mechanism by exposing it to persistent state during pretraining.

### Changes

#### 1. `nanochat/gpt.py`

* **Modified `THENGPT.forward`**: Updated signature and logic to return `(loss, state)` when `targets` are provided and `return_state=True`. This allows the training loop to capture the memory state.

#### 2. `scripts/base_train.py`

* **State Persistence Loop**:
  * Initialize `state = None` at the start of training.
  * Inside the training loop, pass `state` into `model.forward()`.
  * Capture the updated `state` from the model output.
  * **Truncated BPTT**: Detach `state` tensors (`state['traces']`) at the beginning of each step to prevent gradient explosions and limit backpropagation history to the current batch window, while still providing historical context.

### Impact

* **Alignment**: The model now sees a populated `state` dictionary during training (after the first step), mimicking the "Ingest" phase conditions.
* **Mechanism Learning**: The `HybridTHENAttention` layer receives gradients that encourage it to *use* the provided state to lower loss, rather than ignoring it.

### Next Steps

* **Monitor RAM**: The current simple list append (`state['traces'].append`) is $O(N)$. For long training runs, we will need to implement a rolling buffer or tiered memory system (Stage 2 of Roadmap).
