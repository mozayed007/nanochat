# NanoChat-THEN Changelog & Setup Log

**Date:** 2026-02-13
**Project:** Live Memory (THEN Phase)
**Base Repo:** NanoChat (Karpathy)
**Target Hardware:** NVIDIA H100 (Cloud), RTX 4070 Ti (Local Dev)

## 1. Project Initialization

- **Fork**: Established `nanochat-then` as a fork of the `nanochat` repository.
- **Environment**: Configured for `conda` environment `ml` (local) and `livemem` (cloud).

## 2. Architecture Modifications

### `nanochat/gpt.py`

- **New Class `THENGPT`**:
  - Subclassed `GPT` to introduce stateful processing.
  - Added `state` parameter to `forward()` method to accept and return episodic memory traces.
  - Integrated `HybridTHENAttention` as the core attention mechanism.
- **New Module `HybridTHENAttention`**:
  - Implemented a hybrid attention layer that interleaves:
    - **KDA (Knowledge Distillation Attention)** simulation.
    - **DSA (Dense Sparse Attention)** simulation.
  - Designed to handle long-term episodic memory injection.

### `nanochat/checkpoint_manager.py`

- **State Persistence**:
  - Updated `save_checkpoint` and `load_checkpoint` to handle the `state` dictionary.
  - Ensures memory traces are preserved alongside model weights for resumption.

## 3. Workflow Pivot: "Ingest, Don't Train"

*Major architectural decision to align with "Live Memory" goals.*

### Strategy Update

- **Deprecated**: Training directly on episodic data (which "bakes" memories into weights).
- **Adopted**: 3-Phase Pipeline:
    1. **Pretrain**: Meta-learning on generic data to teach memory mechanics.
    2. **Ingest**: Freeze weights, feed episodes, populate `state`.
    3. **Query**: Recall facts from frozen `state`.

### New Scripts

- **`scripts/ingest.py`**:
  - Loads frozen model.
  - Feeds episodic text in chunks.
  - Accumulates and saves `state` (memory traces) to disk.
- **`scripts/query.py`**:
  - Loads frozen model + saved `state`.
  - Runs interactive chat to test recall from memory.

## 4. Data Pipeline Enhancements

### `dev/gen_cairo_data.py`

- **Coherent Narratives**:
  - Shifted from random independent samples to a **continuous timeline (t=1 to t=10000)**.
  - Implemented persistent user state (location, mood, favorite drink) that evolves slowly over time.
  - Ensures the model must track state changes (e.g., "I liked tea at t=500, but coffee at t=600") rather than just pattern matching.

### Training Enhancements (`scripts/base_train.py`)

- **Model Switching**: Added `--model-class` argument to toggle between standard `GPT` and `THENGPT`.
- **Compilation Control**: Added `--no-compile` flag to bypass `torch.compile` issues with stateful inference.
- **FP8 Support**: Integrated flags for FP8 training on H100.
- **FA3 Support**: Added automatic fallback to SDPA for local development in `nanochat/flash_attention.py`.

## 5. Cloud Deployment Setup

- **Documentation**:
  - Updated `CLOUD_SETUP.md` to reflect the new 3-phase workflow.
  - Detailed H100 hardware requirements and FA3 installation steps.
- **Dependencies**:
  - Created `requirements.txt` listing all necessary packages.

## 6. Current Status

- **Local**: Validated on RTX 4070 Ti (using SDPA fallback).
- **Data**: Coherent Cairo timeline ready for ingestion.
- **Cloud**: Ready for upload. `CLOUD_SETUP.md` contains the exact commands for the correct Pretrain -> Ingest -> Query workflow.
