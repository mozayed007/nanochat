# The THEN Architecture: A "Notebook" for Language Models

## The Problem: "Amnesia" in AI

Most Large Language Models (LLMs) suffer from **anterograde amnesia**.

*   **Training (Phase 1)**: They learn general knowledge (e.g., "The sky is blue").
*   **Inference (Chat)**: They cannot "learn" new things permanently. If you tell them "My name is Sarah," they only remember it for the duration of the context window. Once the chat closes or the window fills up, the information is gone forever.

### Why Current Solutions Fail

1.  **SFT / Fine-Tuning**:
    *   **The Mechanism**: Re-training the model's weights on new data.
    *   **The Pain**: It's slow, expensive, and destructive. Teaching the model "Sarah likes coffee" might overwrite "Sarah is a programmer" (Catastrophic Forgetting). It creates a static snapshot, not a living memory.

2.  **RAG (Retrieval-Augmented Generation)**:
    *   **The Mechanism**: Using a search engine (Vector DB) to find documents and pasting them into the prompt.
    *   **The Pain**: It is **disembodied**. The model doesn't "know" the memory; it just reads a text snippet you pasted.
        *   **No Intuition**: It can't form subconscious connections or "feel" the history.
        *   **Latency**: Requires an external database lookup and massive prompt stuffing.
        *   **Context Choking**: Pasting 10k words of history into the prompt makes the model slower, dumber, and wildly expensive per token.

3.  **Plugins / Tools**:
    *   **The Mechanism**: Giving the model a "Save to File" tool.
    *   **The Pain**: It is **manual and clunky**. The model has to *decide* to call a tool, format a JSON, and hope it wrote the right thing. It's not a natural cognitive process; it's a bureaucratic task.

## The Solution: Temporal History Episodic Network (THEN)

The **THEN** architecture (implemented in this fork) separates **Processing** (Weights) from **Memory** (State).

### 1. The "Notebook" Analogy

Imagine the model is a student taking a test.

* **Weights (Frozen)**: The student's brain. It knows how to read, write, and reason. We train this *once* to be smart.
* **State (Live)**: A notebook on the desk.
* **Ingest Phase**: When you tell the model something new ("My name is Sarah"), it doesn't change its brain. Instead, it **writes a note** in the notebook.
* **Query Phase**: When you ask "What is my name?", the model uses its brain to **read the note** and answer "Sarah."

### 2. How It Works (Simplified)

Instead of updating the neural network's parameters (which requires a massive GPU cluster), we append compressed vectors to a `state` file.

* **Ingest**: `User Input -> [Compress] -> Memory Trace`
* **Recall**: `Query -> [Attention] -> Retrieve Trace -> Answer`

### 3. Key Benefits

1. **Instant Learning**: New facts are available immediately. No waiting for a training run.
2. **Zero Forgetting**: Since we don't touch the brain (weights), the model never forgets how to speak English or write code.
3. **Privacy Control**: The "notebook" is just a file. You can delete specific pages (memories) or burn the whole book without affecting the model's core intelligence.
4. **Low Cost**: You can serve 10,000 users with **one** frozen model and 10,000 small notebook files.

---
*This architecture allows us to move from "Static Intelligence" to "Evolving Intelligence" without the massive compute costs of continuous training.*
