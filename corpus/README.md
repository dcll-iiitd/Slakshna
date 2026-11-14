# RAG test corpus

A small two-node corpus for trying out peer-to-peer RAG (see the top-level README).

| Folder | Theme | Documents |
| :-- | :-- | :-- |
| `node1/` | Indian space missions | Chandrayaan-3, Chandrayaan-2, Mars Orbiter Mission, Aditya-L1, Gaganyaan |
| `node2/` | Scientists, a national park, an ancient civilisation | Vikram Sarabhai, Satish Dhawan, A. P. J. Abdul Kalam, Homi J. Bhabha, Kaziranga National Park, Indus Valley Civilisation |

Each node gets only its own folder. The themes are different on purpose, so a question about a node 2 topic cannot be answered by node 1 alone.

## Files

- `node*/**.pdf` — one PDF per article, about 53,000 words in total.
- `manifest.json` — title, source URL, Wikipedia revision id and word count for every PDF.
- `questions.jsonl` — 15 test questions with reference answers and the PDFs that contain them:
  - `node1_only` (6): answerable from node 1's documents;
  - `node2_only` (6): answerable only from node 2's documents, so asking node 1 tests the peer request;
  - `cross_node` (3): need one passage from each node.

Some people in node 2's folder are mentioned by name in node 1's documents (for example the Satish Dhawan Space Centre), but the facts the questions ask for appear only in the listed source files.

## Source and licence

Text is the plain-text extract of English Wikipedia articles, fetched through the MediaWiki API on 2026-10-11, with reference and link sections removed. Wikipedia text is licensed CC BY-SA 4.0; each PDF's first page cites its source URL and revision.

## Rebuilding

```bash
uv run --no-project --with reportlab python scripts/rag_build_corpus.py corpus
```

This fetches the current revision of each article, so the text and `manifest.json` will differ from this snapshot and the answers in `questions.jsonl` should be rechecked.
