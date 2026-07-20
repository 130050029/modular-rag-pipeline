# data/

```
data/
├── manual_test_files/   <- upload these via Swagger to exercise each pipeline behavior
└── golden_set/          <- queries.json for retrieval evaluation (Recall@k / MRR)
```

## manual_test_files/ -- what each one is for

| File | Upload via Swagger to test... |
|---|---|
| `sample_prose.txt` | Plain text extraction + fixed/semantic chunking on ordinary prose |
| `sample_markdown.md` | Markdown extraction (headings, bullet list survive as plain text) |
| `sample_page.html` | HTML extraction -- confirm `<script>`/`<footer>` content is stripped, only real content remains |
| `sample_document.pdf` | PDF extraction via PyMuPDF |
| `near_duplicate_a.txt` + `near_duplicate_b.txt` | Upload both (different filenames, near-identical content) -- the second should return `skipped_near_duplicate` |
| `table_sample.txt` | Contains a Markdown table -- check the chunk's `embedding_text` in `rag.db` differs from its `content` (should be a generated description, not the raw table) |
| `versioning_v1.txt` + `versioning_v2.txt` | Versioning + soft-delete -- see below, requires re-using the same filename |

### Testing versioning specifically

Swagger's file picker uses whatever filename the file has on your machine.
To simulate "the same document, updated":
1. Rename (or copy) `versioning_v1.txt` to `policy.txt`, upload it.
2. Rename (or copy) `versioning_v2.txt` to `policy.txt` (same name), upload it.
3. Check `rag.db`: the v1 chunks should now show `is_stale = 1`, and a new
   `documents` row should show `version = 2`.

## golden_set/

See `golden_set/README.md` -- this is the evaluation set from our RAG
design discussion, scoped to these same manual test files so you can check
retrieval quality end-to-end without needing the full SQuAD seed data.