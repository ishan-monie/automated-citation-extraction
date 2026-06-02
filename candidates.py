#!/usr/bin/env python3
"""
Stage 1 — segment + classify (rules)
Stage 2 — candidate detection: only spans with a nearby anchor (person / year /
          known abbreviation) survive. This kills OCR-noise false positives like
          'Koine 5' or 'VIL 86' at the source, instead of scoring them down later.

Uses spaCy NER for person/date/place anchors if available; otherwise a
heuristic anchor detector. A bi-LSTM-CRF or fine-tuned BERT token tagger can be
slotted in here later behind the same `detect_candidates` interface.
"""
import re

# ── optional spaCy NER ───────────────────────────────────────────────────────
_NLP = None
def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            try:
                _NLP = spacy.load("xx_ent_wiki_sm")   # multilingual
            except Exception:
                _NLP = spacy.load("en_core_web_sm")
        except Exception:
            _NLP = False   # not available
    return _NLP


# ── Stage 1: segmentation ────────────────────────────────────────────────────
COVER_SIGNALS = [r"Author\(s\)\s*:", r"Source\s*:", r"Published by", r"Stable URL",
                 r"JSTOR", r"Library of Congress", r"ISBN", r"Cataloging-in-Publication"]

def classify_segment(text: str) -> str:
    t = text.strip()
    low = t.lower()
    for sig in COVER_SIGNALS:
        if re.search(sig, t, re.IGNORECASE):
            return "cover"
    lines = [l for l in t.splitlines() if l.strip()]
    if lines:
        short_caps = sum(1 for l in lines if re.match(r"^[A-Z][A-Za-z]{0,6}\.?\s*$", l.strip()))
        if "abbreviation" in low and short_caps >= 3:
            return "abbrev-table"
        if short_caps >= 6 and short_caps / len(lines) > 0.3:
            return "abbrev-table"
    if t.count("..") > 4 or re.search(r"\.\s?\.\s?\.\s?\.", t):
        return "toc"
    if len(re.findall(r"\b(?:Z\.|Sp\.|S\.|Anm\.)\s*\d", t)) >= 3:
        return "footnote"
    if len(re.findall(r"\b[A-Z][A-Za-z]{1,5}\s+\d+\s*\(\d{4}\)", t)) >= 2:
        return "footnote"
    return "body"


# ── Stage 2: candidate detection ─────────────────────────────────────────────
YEAR_RE      = re.compile(r"\((?:18|19|20)\d{2}\)|\b(?:18|19|20)\d{2}\b")
# A manuscript/tablet ref is only a candidate if its sigla is a KNOWN collection
KNOWN_SIGLA  = {"K", "Rm", "Sm", "DT", "BM", "VAT", "AO", "Bln", "VS", "STT",
                "CT", "KAR", "TCL", "KAH", "ND", "IM", "BE", "CBS", "Ni"}
MS_RE        = re.compile(r"\b([A-Z][A-Za-z]{0,3})\s+(\d{1,6})\b")

def _has_person(text: str) -> bool:
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text[:1000])
        return any(ent.label_ in ("PERSON", "PER") for ent in doc.ents)
    # heuristic: a capitalised surname-like token followed by a comma or initials
    return bool(re.search(r"\b[A-ZÄÖÜ][a-zäöü]{2,}(?:,|\s+[A-Z]\.)", text))


def _tight_window(text: str, start: int, end: int) -> str:
    """
    Clip a candidate to a citation-sized span around [start,end] by stopping at
    sentence punctuation or a preceding capitalised surname, rather than grabbing
    a fixed huge window. Boundaries are refined downstream; this just avoids
    swallowing neighbouring sentences.
    """
    # walk left to the start of the citation: last '. ' or ';' before start,
    # but keep an immediately-preceding capitalised author token if present
    left = text.rfind(". ", max(0, start - 90), start)
    semic = text.rfind("; ", max(0, start - 90), start)
    left = max(left, semic)
    if left == -1:
        left = max(0, start - 60)
    else:
        left += 2
    # extend left back over an author surname directly before the abbrev/year
    head = text[max(0, left - 40): left]
    am = re.search(r"([A-ZÄÖÜ][\wäöü.\-]+(?:\s+(?:und|and|&|et)\s+[A-ZÄÖÜ][\wäöü.\-]+)?),?\s*$", head)
    if am:
        left = max(0, left - (len(head) - am.start()))
    # walk right to the next sentence end after the page range
    right = end
    rm = re.search(r"[.;]\s", text[end: end + 60])
    right = end + (rm.start() + 1 if rm else min(40, len(text) - end))
    return re.sub(r"\s+", " ", text[left:right]).strip(" .;,")


def detect_candidates(segment_text: str, seg_type: str, abbrev_lookup: dict) -> list:
    """
    Return a list of candidate dicts: {span, anchor, hint_type}.
    A span qualifies only if it has at least one anchor:
      - a person name nearby, OR
      - a year, OR
      - a known journal/series abbreviation, OR
      - a known manuscript sigla
    seg_type 'cover' bypasses anchoring (it's structured metadata).
    """
    cands = []
    text = segment_text

    if seg_type == "cover":
        cands.append({"span": text[:1500], "anchor": "cover", "hint_type": "document-level"})
        return cands

    has_person = _has_person(text)

    # journal-abbrev style: Author, ABBR vol (year) pages
    for m in re.finditer(r"([A-ZÄÖÜ][\wäöü.\-]+(?:\s+(?:und|and|&|et)\s+[A-ZÄÖÜ][\wäöü.\-]+)?),?\s+"
                         r"([A-Z][A-Za-z]{1,6})\.?\s+([IVXLC\d]+)\s*\(?((?:18|19|20)\d{2})\)?", text):
        cands.append({"span": m.group(0).strip(), "anchor": "person+year", "hint_type": "journal-abbrev"})

    # known journal abbreviation present with a year close by -> tight window
    for m in re.finditer(r"\b([A-Z][A-Za-z]{1,6})\b", text):
        tok = m.group(1)
        if tok.upper().replace(".", "") in abbrev_lookup:
            window = _tight_window(text, m.start(), m.end())
            if YEAR_RE.search(window):
                cands.append({"span": window, "anchor": "abbrev+year", "hint_type": "journal-abbrev"})

    # manuscript refs — ONLY known sigla, and only counts as candidate
    for m in MS_RE.finditer(text):
        sigla = m.group(1)
        if sigla in KNOWN_SIGLA:
            cands.append({"span": m.group(0).strip(), "anchor": "known-sigla", "hint_type": "manuscript-ref"})
        # unknown sigla like 'Koine 5' / 'VIL 86' are REJECTED here

    # full author-title with a year and a person anchor -> tight window
    if has_person:
        for m in YEAR_RE.finditer(text):
            window = _tight_window(text, m.start(), m.end())
            if _has_person(window):
                cands.append({"span": window, "anchor": "person+year", "hint_type": "author-title-full"})

    # dedupe by normalised span
    seen, out = set(), []
    for c in cands:
        key = re.sub(r"\s+", " ", c["span"].lower())[:120]
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out
