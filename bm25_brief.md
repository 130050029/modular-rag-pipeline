# BM25 & Hybrid Search Architecture

## 1. Core BM25 Formula
BM25 ranks documents based on term relevance, preventing long documents or stuffed keywords from warping scores.

$$\text{Score}(D, Q) = \sum_{i=1}^{n} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$

*   **IDF (Inverse Document Frequency):** Measures term rarity globally. Calculated asynchronously; pulled via $O(1)$ lookups at query time.
*   **TF Saturation ($k_1$):** Caps the impact of repeated words. Controlled by a saturation constant (typically `1.2` to `2.0`).
*   **Length Normalization ($b$):** Penalizes long documents and rewards short, concise matches (typically set to `0.75`).

---

## 2. Index Structure & Block-WAND
To evaluate queries at scale, storage engines bypass document-by-document calculations using a block-skipping approach.

### Memory & Storage Layout
*   **Segments:** The index is divided into independent, immutable files.
*   **Blocks:** Inside segments, posting lists are chopped into chunks of **exactly 128 DocIDs**.
*   **Compression:** DocIDs are delta-encoded and compressed using bit-packing for CPU cache alignment.
*   **Skip List Headers:** Every block stores a metadata header containing its `LastDocID` and its `MaxTermScore` (the highest BM25 score any single document inside that block can achieve).

### Block-WAND Search Flow
1.  **Align Pointers:** Engine places cursor pointers on the corresponding block headers for each query term.
2.  **Calculate Upper Bound:** The engine sums the `MaxTermScore` values of all active block pointers.
3.  **Evaluate vs. Threshold:** 
    *   If **Upper Bound > Current Top-K Threshold**, the engine decompresses the 128-document block and scores individual rows.
    *   If **Upper Bound $\le$ Current Top-K Threshold**, the engine **skips the entire block** without reading its documents or performing floating-point calculations.

---

## 3. Concurrency, Insertion & Deletion
Extensions align with Postgres MVCC to ensure lock-free execution during active writes.

*   **Insertions:** New documents enter an in-memory **Memtable** buffer. When full, it flushes out to disk as a new immutable segment. Global stats are updated asynchronously.
*   **Deletions:** Handled via a **Tombstone Bitmap** (Roaring Bitmap). The actual index blocks remain unchanged; deleted DocIDs are filtered out dynamically during query runtime.
*   **Compaction:** Background threads merge smaller segments into unified ones, purging tombstone documents and cleanly recalculating exact block-level `MaxTermScore` properties.

---

## 4. Distributed Hybrid Fusion (Postgres BM25 + Milvus Vector)
When text and vectors reside in separate databases, raw scores cannot be directly combined. Applications use **Reciprocal Rank Fusion (RRF)** to merge results by relative rank order.

$$\text{RRF\_Score}(d \in D) = \sum_{m \in M} \frac{1}{k + r_m(d)}$$

### Runtime Architecture Flow
1.  **Parallel Queries:** App layer triggers text search on Postgres (`LIMIT K`) and vector search on Milvus (`LIMIT K`).
2.  **Extract Ranks:** Discards raw BM25 scores and vector distances; captures only the integer position rank ($r_m$) of each document from both outputs.
3.  **Compute Consensus:** Evaluates the RRF equation using a constant smoothing factor ($k = 60$). Items matching strongly across both pipelines naturally float to the top of the combined array.
