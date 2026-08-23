# Golden Set

The golden set is the **ground truth for retrieval experiments**. It should
describe what evidence a query needs, not which implementation-specific
chunk ID happens to contain that evidence today.

That distinction is important because chunk IDs are allowed to change when we
change:

- chunk size or overlap
- fixed vs semantic chunking
- parent/child boundaries
- ingestion order
- embedding models
- vector-store implementations

Therefore, the golden set must **not** use `chunk_id`, FAISS integer IDs, vector
IDs, or parent chunk IDs as its primary ground truth.

## Current state

`queries.json` is still the original v0 sanity set.

It is useful for a basic smoke test, but it is too coarse for serious retrieval
experiments because source-level relevance does not tell us whether the
retrieved passage contains the answer.

## Target golden-set schema

Phase A will migrate the set toward records shaped approximately like:

```json
{
  "id": "amazon_deforestation_001",
  "query": "What causes deforestation in the Amazon?",
  "expected_sources": ["sample_prose.txt"],
  "relevant_evidence": [
    "Agriculture, cattle ranching, logging, and infrastructure development are major causes of deforestation."
  ],
  "reference_answer": "Major causes include agriculture and cattle ranching, with logging and infrastructure also contributing.",
  "answerable": true,
  "difficulty": "medium",
  "tags": ["multi_fact", "paraphrase"]
}
```

The exact schema will be finalized after we inspect every current sample
document and define the evaluation semantics.

### Why keep `expected_sources`?

It is still useful as a coarse diagnostic. If retrieval never reaches the
correct document, there is no reason to inspect passage-level ranking yet.

### Why keep `relevant_evidence`?

It represents the actual information a retriever needs to surface. The
evaluator can later map that evidence to whatever chunk boundaries the current
pipeline happens to produce.

### Why keep `reference_answer`?

Retrieval and generation are different evaluations. A reference answer lets us
later assess whether the retrieved evidence is sufficient for answering the
question without tying the benchmark to a particular chunk.

### Why include unanswerable queries?

A production RAG system must not merely find the most similar document for
every query. Some queries should produce:

```text
insufficient evidence
```

rather than a confident answer based on unrelated context.

The final golden set will therefore contain both answerable and intentionally
unanswerable questions.

## What makes a good golden-set query?

We want a mixture of:

- direct lexical matches
- semantic/paraphrased questions
- rare terms and exact names
- multi-fact questions
- questions whose answer is in a table
- questions requiring the correct passage within a relevant document
- questions where dense retrieval and BM25 are likely to disagree
- deliberately difficult queries
- unanswerable queries

The point is not to create a large number of random questions. The point is to
create a small set of **diagnostic questions** that expose specific retrieval
failures.

## Planned metrics

Once the schema is finalized, Phase A will measure at least:

- Recall@K
- MRR
- NDCG@K

and eventually compare:

```text
dense
BM25
dense + BM25 + RRF
dense + BM25 + RRF + reranker
```

The golden set is a measurement instrument, not a CI pass/fail test.
