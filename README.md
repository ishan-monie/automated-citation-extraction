
# Citation Extraction Pipeline (Multilingual Humanities Texts)

This repository implements a full NLP pipeline for extracting and reconstructing inline citations from scanned, multilingual academic texts. The system is designed to handle structurally inconsistent sources like Assyriological publications in English and French.

---

## Project Structure

```
.
├── combined_text.txt              # Raw OCR text input
├── sentences.csv                 # Sentence-level fragments from OCR text
├── outputs/
│   ├── candidates.json           # Inline citation spans predicted by LLaMA
│   ├── reconstructed.csv        # Fully structured citation metadata
│   └── zotero_export.bib        # Final export file for bibliographic tools
├── extract_sentences.py         # Cleans and segments raw text into sentences
├── extract_candidates_llama.py  # Prompts LLaMA for citation candidate spans
├── ner_bilstm_train.py          # Trains the BiLSTM + NER tagger
├── ner_bilstm_predict.py        # Applies BIO tags to complete citations
└── README.md
```

---

## Step 1: Sentence Segmentation

The pipeline begins with a raw OCR `.txt` file. The script `extract_sentences.py` performs:

- Line break and hyphen cleanup
- Regular expression–based sentence segmentation
- Filtering by sentence length to remove OCR noise

Output: `sentences.csv` — one cleaned sentence per row, used as input to the citation candidate model.

---

## Step 2: Citation Candidate Extraction using Meta-LLaMA-3-8B-Instruct-iMatrix

The script `extract_candidates_llama.py` prompts Meta’s instruction-tuned model using few-shot examples to identify citation spans embedded in prose.

**Model:** `meta-llama/Meta-LLaMA-3-8B-Instruct-iMatrix`

**Prompt Format:**
```
Extract all citation references from the following paragraph.
Return each citation as a separate line.

Text:
<Your Sentence Here>
```

**Inference Parameters:**
- `temperature = 0.0` (for deterministic output)
- `max_new_tokens = 100`
- `do_sample = False` (greedy decoding)
- `top_k = 1` (no randomness)
- Sentences are processed one at a time to avoid truncation

**Output:** `candidates.json`, a list of original sentences paired with extracted citation-like spans.

---

## Step 3: Full Citation Reconstruction with BiLSTM + NER

Once citation spans are extracted, we apply a sequence tagging model to recover structured citation metadata.

### Model Architecture
- **Embedding layer:** 100-dimensional word embeddings trained from scratch on the corpus
- **BiLSTM layer:** 64 hidden units in each direction
- **TimeDistributed dense layer** with softmax output over BIO-formatted tags

### BIO Tags
We use token-level labels such as:
- `B-AUTHOR`, `I-AUTHOR`
- `B-TITLE`, `I-TITLE`
- `B-YEAR`, `B-PAGE`, `B-VOLUME`, `B-NOTE`, etc.
- `O` for outside tokens

### Training Details
- **Dataset:** ~1,200 manually labeled citation fragments
- **Batch size:** 32
- **Sequence length:** 100 tokens
- **Optimizer:** Adam
- **Learning rate:** 0.001
- **Loss function:** Categorical cross-entropy
- **Dropout:** 0.2
- **Epochs:** 10–15 with early stopping on development loss
- **Evaluation:** Token-level and span-level F1 score

**Output:** Labeled token sequences, post-processed into structured fields (Author, Title, Year, Volume, Page, Notes), saved as `reconstructed.csv`.

---

## Step 4: Zotero Export

Final structured outputs are converted into:
- `.csv`: Tabular metadata
- `.bib`: BibTeX entries for LaTeX use
- `.json`: Zotero-compatible records for reference manager import

Where citation fields (e.g., titles or years) are missing, placeholders are used to flag incomplete entries for later manual editing or metadata lookup.

---

## Evaluation Results

**Step 2 – LLaMA (citation fragment extraction):**
- English average similarity: 81%
- French average similarity: 79%

**Step 3 – BiLSTM + NER (structured reconstruction):**
- English average similarity: 96.5%
- French average similarity: 89.4%

Performance was measured using normalized Levenshtein distance, token alignment, and span-level precision.

---

## Notes on Language Performance

The lower French accuracy is attributed to:
- Heavier abbreviation usage in journal names
- OCR degradation in accented characters and ligatures
- More frequent non-standard punctuation and chaining of citations

Future improvements may include abbreviation expansion dictionaries and better sentence segmentation tuned for French-language conventions.

---

## Dependencies

- Python ≥ 3.8
- `transformers`
- `pandas`
- `torch`
- `scikit-learn`
- `sentencepiece`
- GPU recommended for inference and training

---

This pipeline is optimized for multilingual humanities research and supports integration into bibliographic workflows such as Zotero or LaTeX-based citation systems.
