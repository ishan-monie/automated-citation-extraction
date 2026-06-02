#!/usr/bin/env python3
"""
Stage 5 — entity linking + author disambiguation.
Collapses author-name variants (Reade / J. Reade / Reade, J.) into one cluster
and matches each citation against the bibliography DataFrame.

Uses sentence-transformers (LaBSE — multilingual) if installed, otherwise
rapidfuzz string similarity. Same interface either way.
"""
import re

_MODEL = None
def _get_model():
    global _MODEL
    if _MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _MODEL = SentenceTransformer("sentence-transformers/LaBSE")
        except Exception:
            _MODEL = False
    return _MODEL


def _surname(author: str) -> str:
    if not author:
        return ""
    a = author.split(";")[0].split(" und ")[0].split(" and ")[0].strip()
    if "," in a:
        return a.split(",")[0].strip().lower()
    toks = a.split()
    return toks[-1].lower() if toks else ""


def cluster_authors(citations: list):
    """Assign author_cluster_id to each citation by grouping similar surnames."""
    surnames = [_surname(c.author) for c in citations]
    uniq = sorted(set(s for s in surnames if s))
    if not uniq:
        return

    model = _get_model()
    clusters = {}   # surname -> cluster_id
    if model:
        import numpy as np
        embs = model.encode(uniq, normalize_embeddings=True)
        assigned = {}
        cid = 0
        for i, s in enumerate(uniq):
            if s in assigned:
                continue
            assigned[s] = cid
            for j in range(i + 1, len(uniq)):
                if uniq[j] not in assigned:
                    sim = float((embs[i] * embs[j]).sum())
                    if sim >= 0.92:
                        assigned[uniq[j]] = cid
            cid += 1
        clusters = assigned
    else:
        from rapidfuzz import fuzz
        assigned, cid = {}, 0
        for i, s in enumerate(uniq):
            if s in assigned:
                continue
            assigned[s] = cid
            for j in range(i + 1, len(uniq)):
                if uniq[j] not in assigned and fuzz.ratio(s, uniq[j]) >= 88:
                    assigned[uniq[j]] = cid
            cid += 1
        clusters = assigned

    for c, s in zip(citations, surnames):
        c.author_cluster_id = clusters.get(s, -1)


def match_bibliography(citations: list, bib_df):
    """Set bib_matched if the author surname appears in any bibliography column."""
    if bib_df is None or len(bib_df) == 0:
        return
    hay = " ".join(bib_df.astype(str).fillna("").apply(lambda r: " ".join(r), axis=1)).lower()
    for c in citations:
        sn = _surname(c.author)
        if sn and len(sn) > 2 and sn in hay:
            c.bib_matched = True
