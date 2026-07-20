# Golden Set

`queries.json` is the evaluation set we discussed: each entry is a
`(query, expected_source)` pair against the files in
`../manual_test_files/`. Upload all of those files first (via Swagger or
curl), then this set can be used to compute Recall@k and MRR the way we
described -- run each query through `/chat` (or directly through
`retrieve()` in a small script), check whether `expected_source` appears
among the returned `sources`, and at what rank.

This is intentionally small (10 queries) -- enough to sanity-check that
retrieval is working at all and to compare chunking-strategy changes
against each other, not a statistically rigorous benchmark. A real golden
set, per what we discussed, would have a few hundred pairs oversampling
your hardest document types.

## Format

```json
{
  "query": "the question a user might ask",
  "expected_source": "the filename that should be retrieved for it"
}
```

## Quick manual check via Swagger

1. Upload every file in `manual_test_files/` via `/upload`.
2. For each entry in `queries.json`, POST it to `/chat` and check whether
   `sources` contains the `expected_source` filename.

## Turning this into an actual Recall@k / MRR script (next step)

Once we cover retrieval evaluation properly, this file is exactly what a
small `eval.py` script would loop over: for each pair, call
`rag.retrieval.retrieve(query, top_k=N)`, check if `expected_source`
appears in the returned sources (Recall@k), and note its rank if so (for
MRR). Not implemented yet -- flagging where it plugs in once we get there.