# Manual Test Data

```
data/
├── manual_test_files/   Sample files for exercising each pipeline behavior via Swagger or curl
└── golden_set/          queries.json for retrieval evaluation (Recall@k / MRR)
```

## manual_test_files/

| File | Exercises |
|---|---|
| `sample_prose.txt` | Plain text extraction, fixed/semantic chunking |
| `sample_markdown.md` | Markdown extraction |
| `sample_page.html` | HTML extraction (`<script>`/`<footer>` content should be stripped) |
| `sample_document.pdf` | PDF extraction via PyMuPDF |
| `near_duplicate_a.txt` + `near_duplicate_b.txt` | Upload both. The second returns `skipped_near_duplicate`. |
| `table_sample.txt` | Contains a Markdown table. The table becomes its own chunk with a generated `embedding_text` description, separate from the surrounding prose. |
| `versioning_v1.txt` + `versioning_v2.txt` | Versioning + soft-delete. See below — requires uploading both under the same filename. |

### Testing versioning

Versioning is keyed on filename (`source_uri`), so the two files must be
uploaded under an **identical** filename to be recognized as two versions
of the same document. The simplest way is to override the filename at
upload time with curl, without renaming anything on disk:

```bash
curl -X POST http://127.0.0.1:8000/upload -F "file=@versioning_v1.txt;filename=policy.txt"
curl -X POST http://127.0.0.1:8000/upload -F "file=@versioning_v2.txt;filename=policy.txt"
```

After the second call, inspect `rag.db`: the first upload's chunks should
show `is_stale = 1`, and a new `documents` row should show `version = 2`.

If testing via the Swagger UI instead (which uses the actual filename of
whatever file you select), rename or copy one file to match the other's
name before selecting it in the file picker.

## golden_set/

See `golden_set/README.md`.