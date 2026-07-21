# Golden Set

`queries.json` is a retrieval evaluation set: each entry is a
`(query, expected_source)` pair against the files in
`../manual_test_files/`. After uploading those files, this set can be used
to compute Recall@k and MRR — run each query through `/chat` (or directly
through `retrieve()`), and check whether `expected_source` appears among
the returned sources, and at what rank.

This set is intentionally small (10 queries) — sufficient to sanity-check
that retrieval works and to compare chunking-strategy changes against one
another, but not a statistically rigorous benchmark. A production-scale
golden set would contain a few hundred pairs, oversampling the hardest
document types in the corpus.

## Format

```json
{
  "query": "the question a user might ask",
  "expected_source": "the filename that should be retrieved for it"
}
```

## Manual evaluation via Swagger

1. Upload every file in `manual_test_files/` via `/upload`.
2. For each entry in `queries.json`, POST it to `/chat` and check whether
   `sources` contains the `expected_source` filename.

## Automated evaluation (planned)

A small `eval.py` script would loop over this file: for each pair, call
`rag.retrieval.retrieve(query, top_k=N)`, check whether `expected_source`
appears in the returned sources (Recall@k), and record its rank if so (for
MRR). Not yet implemented.