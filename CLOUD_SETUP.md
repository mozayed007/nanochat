# Live Memory: THEN Project - Cloud Setup & Execution Guide

This guide details the setup and execution of the **THEN (Temporal History Episodic Network)** project in a cloud environment. This project extends the NanoChat architecture to support episodic memory through `HybridTHENAttention` and stateful inference, as outlined in the Live Memory project documentation.

## 1. Hardware Requirements

To utilize the full capabilities of the "THEN" architecture, specifically **Flash Attention 3 (FA3)**, the following hardware is required:

- **GPU**: NVIDIA H100 (Hopper Architecture).
- **VRAM**: 80GB recommended for efficient training of deeper models.
- **Storage**: Fast NVMe SSD for data loading.

*Note: The codebase includes a fallback to SDPA (Flash Attention 2) for non-Hopper GPUs (e.g., A100, RTX 4090), but FA3 is strictly required for the intended performance profile.*

## 2. Environment Setup

### Base Environment

We recommend using `conda` or `uv` to manage the environment.

```bash
# Create and activate environment
conda create -n livemem python=3.10 -y
conda activate livemem

# Install PyTorch (ensure compatibility with your CUDA version, e.g., 12.1 or 12.4)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Dependencies

Install the project dependencies using the provided `requirements.txt`:

```bash
pip install -r requirements.txt
```

### Flash Attention 3

Flash Attention 3 must be installed separately to ensure it targets the correct GPU architecture.

```bash
# Install Flash Attention (verify version supports FA3)
pip install flash-attn --no-build-isolation
```

## 3. Project Architecture: THEN

The core of this project is the **THENGPT** model, which introduces:

- **HybridTHENAttention**: A custom attention mechanism interleaving **KDA** (Knowledge Distillation Attention) and **DSA** (Dense Sparse Attention) simulations.
- **Stateful Inference**: The model maintains an episodic `state` (memory traces) across forward passes, unlike standard stateless Transformers.

### Key Files

- `nanochat/gpt.py`: Contains the `THENGPT` and `HybridTHENAttention` classes.
- `nanochat/checkpoint_manager.py`: Handles saving/loading of both model weights and memory state.

## 4. Data Preparation

The project currently uses **synthetic episodic data** for prototyping.

### Generate Data

Run the Cairo-themed synthetic data generator to create the training dataset:

```bash
# Generate synthetic episodes
python dev/gen_cairo_data.py
```

This will create `data/synthetic-cairo-episodes.txt` (or similar, check output).

*Note: For large-scale pretraining (e.g., FineWeb-Edu), use `dev/repackage_data_reference.py` to process external datasets.*

## 5. Execution Pipeline (Corrected for Live Memory)

The Live Memory project follows a 3-phase pipeline: **Pretrain → Ingest → Query**.
You should **NOT** train on the Cairo episodes directly.

### Phase 1: Pretraining (Meta-Learning)

Train the `THENGPT` model on generic data (e.g., FineWeb or a small synthetic base) to teach it *how* to use the memory mechanism.

```bash
# Example: Train on default data (FineWeb-Edu subset)
# This produces the initial weights (Phase 1)
python -m scripts.base_train \
    --model-class THENGPT \
    --depth 8 \
    --total-batch-size 524288 \
    --wandb-project "livemem-then" \
    --run-name "phase1-pretrain" \
    --save-every 1000
```

*Output*: Checkpoint at `base_checkpoints/d8/model_001000.pt` (example).

### Phase 2: Ingestion (The "Live" Part)

Freeze the model and feed the Cairo episodes to populate the internal `state`.

```bash
# Run ingestion script
python -m scripts.ingest \
    --model_path base_checkpoints/d8/model_001000.pt \
    --data_path data/synthetic-cairo-episodes.txt \
    --output_path cairo_memory_state.pt
```

*Output*: `cairo_memory_state.pt` (contains the memory traces).

### Phase 3: Recall & Query

Query the frozen model using the populated state.

```bash
# Interactive query
python -m scripts.query \
    --model_path base_checkpoints/d8/model_001000.pt \
    --state_path cairo_memory_state.pt
```

*Action*: Ask "What did the user prefer at t=7?" and verify the recall.

## 6. Verification

Before launching a full run, verify that Flash Attention 3 is active and the Live Memory flow is working:

```bash
# 1. Verify GPU and FA3
python scripts/test_chat.py

# 2. Verify Live Memory Logic
python -m tests.test_live_memory
```

Check the output logs for "Using Flash Attention 3" and "OK" for the tests.

## 7. Roadmap & Plans (`docs/plans/plan0.md`)

This setup fulfills the **Phase 1: Prototype** requirements of `plan0.md`:

1. **Setup**: H100 environment prepared.
2. **Code**: `THENGPT` implemented.
3. **Data**: Synthetic pipeline ready.
4. **Training**: Scripts configured for stateful pretraining.

You are now ready to upload this codebase to the cloud instance and begin the "THEN" phase of Live Memory.
