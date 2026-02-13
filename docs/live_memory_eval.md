# Live Memory Evaluation Strategy

This document outlines the testing and evaluation procedures for the Live Memory "Ingest, Don't Train" workflow.

## 1. Mechanical Verification

**Goal:** Ensure the software pipeline correctly handles state persistence, passing, and growth.

**Method:** Unit tests in `tests/test_live_memory.py`.
**Metrics:**

- **State Growth:** Does the memory state size increase as more tokens are ingested?
- **State Persistence:** Can the state be saved to disk and loaded back identically?
- **State Injection:** Does the `query.py` script successfully load and use the state during inference?

**Run Command:**

```bash
python -m tests.test_live_memory
```

## 2. Functional Recall Evaluation (Accuracy)

**Goal:** Measure the model's ability to recall specific facts ingested during Phase 2.

**Method:**

1. **Ingest:** Feed a synthetic dataset containing specific key-value pairs (e.g., "At 10:00, the user drank coffee.").
2. **Query:** Ask questions targeting those facts (e.g., "What did the user drink at 10:00?").
3. **Score:** Exact match or semantic similarity of the answer.

**Prerequisites:**

- A pretrained `THENGPT` model (Phase 1 complete). Random weights will produce near-zero recall accuracy.

## 3. Coherence and Consistency Check

**Goal:** Verify that the ingested narrative remains consistent over long timelines (Phase 2).

**Method:**

- Use `dev/gen_cairo_data.py` to generate "Coherent" data.
- **Check:** The generator enforces logic (e.g., User cannot be in two places at once; Mood changes gradually).
- **Validation:** During ingestion, monitor the "State traces" count in the logs. It should grow linearly with the number of compressed segments (controlled by `ratio`).

## Current Status

- **Mechanical Verification:** Implemented and passing (`tests/test_live_memory.py`).
- **Data Coherence:** Implemented in `dev/gen_cairo_data.py` (persistent user state).
- **Recall Accuracy:** Pending full pretraining of the `THENGPT` model.
