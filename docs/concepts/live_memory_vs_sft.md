# Conceptual Guide: Live Memory vs. Continued Pretraining

This document clarifies the architectural and functional differences between the **Live Memory (THEN)** workflow and standard **Continued Pretraining** or **Supervised Fine-Tuning (SFT)** used by frontier labs.

## 1. The Core Distinction: Weights vs. State

The fundamental difference lies in **where** the new information is stored and **how** it is written.

| Feature | Continued Pretraining / SFT | Live Memory (THEN) |
| :--- | :--- | :--- |
| **Storage Mechanism** | **Synaptic Weights** (Parameters) | **Episodic State** (Buffer/Cache) |
| **Write Operation** | `loss.backward()` + Optimizer Step | `state['traces'].append()` |
| **Model Mode** | `model.train()` | `model.eval()` (Frozen) |
| **Permanence** | Permanent (until overwritten) | Volatile (unless saved to disk) |
| **Speed** | Slow (requires gradient calculation) | Instant (forward pass only) |
| **Forgetting** | **Catastrophic Forgetting** (Old data lost) | **No Forgetting** (Additive state) |

---

## 2. Standard Approach: Continued Pretraining / SFT

When frontier labs "pause and continue" training on new or higher-quality data, they are modifying the model's **weights**.

### The Process

1. **Input**: New text data (e.g., higher quality books).
2. **Compute**: Calculate loss (prediction error) and gradients.
3. **Update**: Adjust billions of floating-point numbers (weights) slightly to minimize error on the new data.

### The Problem

* **Destructive**: To learn $B$, the model often overwrites parts of $A$. This is "catastrophic forgetting."
* **Opaque**: The knowledge is diffused across the entire network. You cannot point to a specific neuron and say, "This is the memory of user ID 42."
* **Static**: Once training stops, the model is frozen. It cannot learn *during* a conversation without a retraining cycle.

---

## 3. Our Approach: Live Memory (Ingest)

The **THEN (Temporal History Episodic Network)** architecture treats memory as a distinct **state object**, separate from the processing **weights**.

### The Live Memory Process

1. **Phase 1 (Pretrain)**: Train weights *once* to learn the **mechanism** of memory (how to compress, store, and retrieve).
2. **Phase 2 (Ingest)**: Feed episodic data into the frozen model.
    * Instead of updating weights, the `HybridTHENAttention` layer compresses the input into **traces**.
    * These traces are appended to a `state` dictionary.
3. **Phase 3 (Query)**: The model attends to this `state` to answer questions.

### Code Reference (`nanochat/gpt.py`)

In `HybridTHENAttention.forward`:

```python
# SFT would update self.kda.weight here via backprop.
# Live Memory instead does this:

if layer_idx % (self.ratio + 1) < self.ratio:
    # Compress input into a trace
    compressed = self.kda(x)
    trace = torch.mean(compressed, dim=1)
    
    # WRITE to State (Instant Memory)
    state['traces'].append(trace) 
    
    return compressed, state
```

### The Advantage

* **Instant**: As soon as the user speaks, it is in the state. No training run required.
* **Surgeon-Precise**: We can delete a specific memory by removing its trace from the list.
* **Stable**: The core reasoning capabilities (weights) are never touched, so the model doesn't get "dumber" or forget language rules while learning new facts.

## Summary

| Metaphor | Continued Pretraining | Live Memory |
| :--- | :--- | :--- |
| **Human Analogy** | **Brain Plasticity**: Physically rewiring neurons to learn a skill (slow, hard to reverse). | **Working Memory**: Writing a note in a notebook (fast, easy to edit). |
| **Computer Analogy** | **Firmware Update**: Flashing the ROM. | **RAM/Disk Write**: Saving a file. |

**Your "Live Memory" is a dynamic, stateful buffer that the model learns to read/write to, whereas SFT is physically changing the model's brain.**
