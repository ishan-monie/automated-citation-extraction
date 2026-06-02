# Citation Extraction Pipeline v5

A hybrid neural pipeline that extracts **inline citations and document-level metadata** from noisy multilingual OCR scans (English / German / French — Assyriology & ancient Near Eastern studies). Each citation gets a **0–1 confidence score** and exports to **highlighted HTML**, **Zotero-importable CSL-JSON**, and CSV.

---

## Demo

`demo_outputs/` contains results from running the pipeline on `sample_pages/page_4.csv` — a dense German Assyriological footnote page (Borger). Open `demo_outputs/viewer.html` in a browser to see the highlighted output.

**Results on page_4.csv:**
- 74 citations extracted, 64 high-confidence (≥0.7)
- 91% author, 89% year, 68% journal populated
- Cross-references (`a.a.O`, `ibid.`, `ders.`) resolved
- Manuscript refs (`KAR 313`, `CT 13`) correctly kept

---

## Architecture

The LLM is a *specialist consultant*, not the backbone. Purpose-built models do the heavy lifting:

| Stage | Model class | Job |
|---|---|---|
| 1. Segment + classify | rules | tag `body`/`footnote`/`cover`/`abbrev-table` |
| 2. Candidate detection | NER + heuristic | only spans with person/year/known-abbrev anchor survive — kills OCR noise at the source |
| 3. Completion check | regex | extends truncated spans (`AfK 2 (1924` → `AfK 2 (1924/25) Sp. 28-30`) |
| 3a. Abbrev lookup | rule/dict | `OLZ → Orientalistische Literaturzeitung` via RlA Excel + in-doc tables |
| 3b. Cross-ref resolve | rule (+LLM) | `ibid./ebd.`, `op.cit./a.a.O.`, `idem/ders.` with full Latin+German token table |
| 4. Field parser | GROBID → LLM fallback | author/title/journal/volume/year/pages |
| 5. Span trimmer | NER + field-anchored | clips surrounding prose from wide candidate windows |
| 6. Entity linking | SBERT → rapidfuzz | author disambiguation, bibliography matching |
| 7. Critic | LLM | rejects junk below confidence threshold |
| 8. Export | — | HTML · Zotero CSL-JSON + RDF · triples · CSV |

---

## Graceful degradation — runs today, upgrades later

| Stage | Best | Fallback |
|---|---|---|
| Candidate NER | spaCy `xx_ent_wiki_sm` | heuristic anchoring |
| Field parser | GROBID server | LLM |
| Entity linking | sentence-transformers LaBSE | rapidfuzz |

Runs with only `openai + pandas + rapidfuzz`. Each optional component auto-activates when installed.

---

## Setup

```bash
pip install -r requirements.txt

# Optional upgrades
pip install spacy && python -m spacy download xx_ent_wiki_sm
pip install sentence-transformers

# GROBID (best parser — run server first)
docker run -t --rm -p 8070:8070 lfoppiano/grobid:0.8.0
export GROBID_URL=http://localhost:8070
```

Create a `.env` file:
```
LLM_API_KEY=gsk_...
LLM_API_URL=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.1-8b-instant
```

---

## Run

```bash
# Single page
python pipeline.py --input sample_pages/page_4.csv

# Full folder
python pipeline.py \
  --input-dir ./your_pages \
  --abbrev RlA_Bibliograph_2026.xlsx \
  --bib secondary_sources_bibliography.xlsx \
  --min-confidence 0.4
```

---

## Outputs

| File | Contents |
|---|---|
| `viewer.html` | Pages re-rendered with citations colour-coded by type; hover for fields; toggle to dim low-confidence |
| `citations_zotero.json` | CSL-JSON — *File → Import* in Zotero |
| `citations_zotero.rdf` | Zotero RDF (legacy) |
| `triples.jsonld` | RDF triples for factual inline citations |
| `citations.csv` | All fields + `confidence`, `parsed_by`, `xref_kind`, `author_cluster_id` |

---

## Cross-reference tokens handled

Latin and German both:
`ibid./ib./ebd./ebenda` · `op.cit./loc.cit./a.a.O.` · `idem/id./ders./dies./eadem`

---

## Known limitations / open issues

- **Cross-page op.cit.**: `op.cit.` resolves within one page's citation state. A work first cited on page 3 referenced on page 5 won't link unless you run the whole document as one state (easy extension — pass accumulated citations between `run_page` calls).
- **Stale authors on truncated spans**: when OCR clips the start of a citation mid-word (`gyptiennes méconnue`), the author field may carry the fragment. The trimmer removes it from the span but the LLM already set the field — a post-parse field re-extraction on trimmed spans would fix this.
- **Off-the-shelf GROBID**: trained on English STEM, underperforms on Assyriological abbreviations. Fine-tune an `XLM-RoBERTa` token classifier on your gold standard for best domain quality.
- **Duplicate detection**: citations with slightly different spans for the same reference (e.g. one truncated, one complete) occasionally survive dedup. The fingerprint-based dedup handles most cases; fuzzy span similarity (75% threshold) catches the rest.

---

## Files

```
citation_v5/
├── pipeline.py        orchestrator + graph + CLI
├── state.py           CitationState + Citation dataclass
├── candidates.py      stage 1 segment + stage 2 candidate detection
├── completer.py       stage 3 completion check (extends truncated spans)
├── parsing.py         stage 4 GROBID/LLM parse + critic
├── resolution.py      stage 3a abbrev lookup + 3b cross-ref (Latin+German)
├── trimmer.py         stage 5 span boundary trimming
├── linking.py         stage 6 entity linking / author clustering
├── exporters.py       stage 8 HTML / Zotero / CSV exports
├── requirements.txt
├── .gitignore
├── README.md
├── sample_pages/
│   └── page_4.csv     demo input (Borger, dense German footnotes)
└── demo_outputs/
    ├── viewer.html
    ├── citations.csv
    ├── citations_zotero.json
    ├── citations_zotero.rdf
    └── triples.jsonld
```
