#!/usr/bin/env python3
"""
v5 orchestrator — the graph.

Wires the stages with conditional edges:
  1 segment+classify ─► 2 candidate detection ─► 3 parse (GROBID|LLM)
  ─► 3a abbrev lookup ─► 3b cross-ref resolve ─► [4 LLM re-parse if low conf]
  ─► 5 entity linking ─► [6 critic if conf < threshold] ─► score ─► 7 export

Selective LLM use keeps token cost down:
  - GROBID parses well-formed strings; LLM only when GROBID absent/low-conf.
  - The critic runs ONLY on citations below `critic_threshold`.
This is a plain-Python state machine; the same graph maps 1:1 onto LangGraph
nodes/edges if you later want tracing (see README).
"""
import os, re, glob, argparse
from pathlib import Path
import pandas as pd

# load .env first so all env vars are available before any module reads them
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv optional — can still set vars manually

from state import Citation, CitationState
from candidates import classify_segment, detect_candidates
from completer import complete_candidates
from resolution import (build_lookup, resolve_abbrev, resolve_crossref,
                        detect_xref)
from parsing import parse_fields, llm_parse, critic_review, make_llm_opcit_resolver
from linking import cluster_authors, match_bibliography
from trimmer import trim_all
import exporters


# ── helpers to load resources ────────────────────────────────────────────────
def load_abbrev_excel(path):
    if not path or not os.path.exists(path):
        return {}
    try:
        df = pd.read_excel(path)
        df.columns = [str(c).lower().strip() for c in df.columns]
        akey = next((c for c in df.columns if any(s in c for s in ["abbrev","short","sigel","abk"])), df.columns[0])
        aval = next((c for c in df.columns if any(s in c for s in ["title","full","name","expan"])),
                    df.columns[1] if len(df.columns) > 1 else df.columns[0])
        return {str(r[akey]).strip(): str(r[aval]).strip()
                for _, r in df.iterrows() if str(r.get(akey,"")).strip() not in ("","nan")}
    except Exception as e:
        print(f"  [warn] abbrev excel: {e}")
        return {}


def load_bib(path):
    if not path or not os.path.exists(path):
        return None
    try:
        return pd.read_excel(path)
    except Exception:
        return None


def load_segments(csv_path):
    df = pd.read_csv(csv_path)
    # find the text column — prefer 'text', else first column with long strings
    if "text" in df.columns:
        tcol = "text"
    else:
        tcol = next(
            (c for c in df.columns
             if df[c].dropna().astype(str).str.len().mean() > 40),
            df.columns[-1]
        )
    segs = []
    for idx, row in df.iterrows():
        t = str(row.get(tcol, "")).strip()
        if len(t) >= 30:
            segs.append({"segment_id": int(idx), "text": t})
    return segs


# ── confidence scoring ───────────────────────────────────────────────────────
def score(c: Citation) -> float:
    s = 0.0
    if c.author: s += 0.25
    if c.year: s += 0.15
    if c.journal_series or c.title: s += 0.20
    if c.pages or c.volume: s += 0.10
    if len(c.span) > 3 and re.search(r"[A-Za-zÄÖÜäöü]", c.span): s += 0.10
    if c.resolved_abbrev: s += 0.10
    if c.bib_matched: s += 0.10
    if c.parsed_by == "crf": s += 0.05         # GROBID agreement bonus
    if c.xref_kind and c.resolves_to_index >= 0: s += 0.15   # successfully resolved ref
    if c.citation_class == "document-level" and c.author and c.title:
        s = max(s, 0.85)
    if c.cite_type == "manuscript-ref" and c.journal_series:
        s = max(s, 0.55)
    if c.critic_verdict == "reject":
        s = min(s, 0.15)
    return round(min(s, 1.0), 2)


# ── the graph ────────────────────────────────────────────────────────────────
def run_page(csv_path, abbrev_map, bib_df, args):
    st = CitationState(page_id=Path(csv_path).stem, abbrev_map=abbrev_map)
    st.bib_df = bib_df
    st.critic_threshold = args.critic_threshold
    lookup = build_lookup(abbrev_map)
    opcit_resolver = make_llm_opcit_resolver()

    # 1 segment + classify, harvest in-doc abbrevs from abbrev-table segments
    raw_segs = load_segments(csv_path)
    for s in raw_segs:
        s["seg_type"] = classify_segment(s["text"])
    st.segments = raw_segs

    # 2 candidate detection (+ cross-ref candidates kept even without anchors)
    cand_list = []
    for s in raw_segs:
        cs = detect_candidates(s["text"], s["seg_type"], lookup)
        cs = complete_candidates(cs, s["text"])
        # also keep explicit cross-ref tokens as candidates (ibid/op.cit/ders...)
        # capture a tight window: a preceding author surname + the xref token + trailing page
        for m in re.finditer(
                r"([A-ZÄÖÜ][\wäöü.\-]+)?,?\s*"
                r"\b(ibid\.?|ebd\.?|a\.\s?a\.\s?O\.?|op\.\s?cit\.?|loc\.\s?cit\.?|ders\.?|dies\.?|idem)\b"
                r"\.?,?\s*(?:(?:p\.?|S\.?|pp\.?|Sp\.?)\s*\d+(?:[-–]\d+)?)?",
                s["text"], re.IGNORECASE):
            span = m.group(0).strip(" .,")
            if len(span) >= 3:
                cs.append({"span": span, "anchor": "xref", "hint_type": "xref"})
        for c in cs:
            c["segment_id"], c["seg_type"] = s["segment_id"], s["seg_type"]
        cand_list.extend(cs)
    st.candidates = cand_list

    # 3 parse each candidate, in document order
    for cand in cand_list:
        is_xref = detect_xref(cand["span"]) != ""
        if is_xref:
            c = Citation(span=cand["span"], cite_type="xref",
                         seg_type=cand.get("seg_type",""), parsed_by="rule")
        else:
            c = parse_fields(cand, prefer_grobid=bool(os.environ.get("GROBID_URL")))
        c.page_id, c.segment_id = st.page_id, cand.get("segment_id", -1)
        c.seg_type = cand.get("seg_type", "")
        st.add(c)

        # 3a abbrev lookup
        resolve_abbrev(c, lookup)
        # 3b cross-ref resolution against everything seen before
        resolve_crossref(c, st.citations[:c.order], llm_resolver=opcit_resolver)

        # 4 LLM re-parse if low confidence and not already LLM/xref
        if (not is_xref and c.parsed_by != "llm"
                and not c.core_complete() and c.cite_type != "manuscript-ref"):
            fix = llm_parse(c.span, c.cite_type)
            if fix and "_error" not in fix:
                for k in ("author","year","title","journal_series","volume","pages"):
                    if not getattr(c, k) and fix.get(k):
                        setattr(c, k, str(fix[k]))
                c.parsed_by = "llm"
                resolve_abbrev(c, lookup)

    # 5 entity linking
    cluster_authors(st.citations)
    match_bibliography(st.citations, bib_df)

    # 6 critic — ONLY below threshold (saves tokens)
    for c in st.citations:
        c.confidence = score(c)
        if c.confidence < st.critic_threshold and c.cite_type != "xref":
            v = critic_review(c)
            c.critic_verdict = v.get("verdict", "keep")
            fix = v.get("fix") or {}
            for k in ("author","year","title","journal_series","volume","pages"):
                if fix.get(k) and not getattr(c, k):
                    setattr(c, k, str(fix[k]))
            c.confidence = score(c)   # rescore after critic

    # trim spans to citation boundaries using extracted fields as anchors
    trim_all(st.citations)

    # fix stale author fields that came from pre-trim prose
    _clean_stale_authors(st.citations)

    # drop rejects and prose-only candidates, then dedupe
    kept = [c for c in st.citations
            if c.critic_verdict != "reject" and not _is_prose_candidate(c)]
    return _dedupe(kept)


def _clean_stale_authors(citations: list):
    """
    Null out author fields that are clearly pre-trim prose rather than a real
    author name — i.e. the author field contains 5+ words, starts with a
    lowercase letter, starts with a known prose opener, or matches the full span.
    """
    PROSE_OPENERS = re.compile(
        r"^(In |Im |An |Zu |Fir |Für |Von |Der |Die |Das |Als |See |For |The |This |"
        r"Ebelings |Bibliographische |Ganzes|Unicode|An- fang)",
        re.IGNORECASE
    )
    for c in citations:
        if not c.author or c.author in ("nan", ""):
            continue
        auth = c.author.strip()
        # author matches the whole span (LLM dumped span into author)
        if auth == c.span.strip():
            c.author = ""
            continue
        # author starts with prose opener
        if PROSE_OPENERS.match(auth):
            c.author = ""
            continue
        # author is suspiciously long (real authors rarely exceed 5 words)
        if len(auth.split()) > 5:
            c.author = ""
            continue
        # author starts lowercase (mid-word OCR fragment)
        if auth and auth[0].islower():
            c.author = ""


def _dedupe(cits: list) -> list:
    """
    Improved dedup: merge citations with the same author+year+journal
    OR very similar spans (rapidfuzz), keeping highest confidence.
    """
    from rapidfuzz import fuzz

    # first pass: exact normalised span key
    by_span = {}
    for c in cits:
        key = re.sub(r"\s+", " ", c.span.lower().strip())[:120]
        if key not in by_span or c.confidence > by_span[key].confidence:
            by_span[key] = c
    unique = list(by_span.values())

    # second pass: merge by author+year+journal fingerprint
    def fingerprint(c):
        auth = (c.author or "").split(",")[0].split()[0].lower()[:6] if c.author else ""
        yr   = (c.year or "")[:4]
        jnl  = re.sub(r"\s+","", (c.journal_series or "").lower())[:6]
        return f"{auth}|{yr}|{jnl}"

    by_fp = {}
    for c in unique:
        fp = fingerprint(c)
        if fp == "||":   # no fields — keep as-is
            by_fp[id(c)] = c
        elif fp not in by_fp or c.confidence > by_fp[fp].confidence:
            by_fp[fp] = c

    result = list(by_fp.values())

    # third pass: fuzzy span similarity (catches "Schott, OLZ 45 (1942" vs "OLZ 45 (1942) Sp. 165-172")
    final, used = [], set()
    for i, a in enumerate(result):
        if i in used:
            continue
        best = a
        for j, b in enumerate(result):
            if j <= i or j in used:
                continue
            sim = fuzz.token_set_ratio(a.span, b.span)
            if sim >= 75:
                used.add(j)
                if b.confidence > best.confidence:
                    best = b
        final.append(best)

    return final


# ── prose candidate filter ───────────────────────────────────────────────────
_PROSE_PATTERNS = [
    re.compile(r"^(Als ich|In meiner|In HKL|Der Keilschrift|Ebelings Uber|"
               r"Bibliographische|Ganzes\.|Unicode Cons|Herr |An- fang|"
               r"1973/74 habe|Es interessiert)", re.IGNORECASE),
]

def _is_prose_candidate(c) -> bool:
    span = c.span.strip()
    if any(p.match(span) for p in _PROSE_PATTERNS):
        return True
    # span has no journal, no title, no pages — just a year and prose author
    if (not c.journal_series or c.journal_series in ("nan","")) and \
       (not c.title or c.title in ("nan","")) and \
       (not c.pages or c.pages in ("nan","")) and \
       c.cite_type not in ("manuscript-ref", "xref"):
        # and the span looks like a sentence (verb-like tokens)
        if re.search(r"\b(ich|habe|wurde|ist|hat|finden|liegt|beruht|darf)\b", span, re.IGNORECASE):
            return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Hybrid neural citation extraction v5")
    ap.add_argument("--input"); ap.add_argument("--input-dir")
    ap.add_argument("--abbrev", default="RlA_Bibliograph_2026.xlsx")
    ap.add_argument("--bib", default="secondary_sources_bibliography.xlsx")
    ap.add_argument("--output-dir", default="outputs")
    ap.add_argument("--critic-threshold", type=float, default=0.6)
    ap.add_argument("--min-confidence", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    abbrev_map = load_abbrev_excel(args.abbrev)
    bib_df = load_bib(args.bib)
    print(f"abbrev map: {len(abbrev_map)} | bib rows: {0 if bib_df is None else len(bib_df)}")
    print(f"GROBID: {os.environ.get('GROBID_URL') or 'not set (LLM parse)'} | "
          f"LLM: {os.environ.get('LLM_MODEL','llama-3.1-8b-instant')}\n")

    files = [args.input] if args.input else sorted(glob.glob(os.path.join(args.input_dir, "*.csv")))
    all_c = []
    for f in files:
        print(f"Processing {f}")
        cits = run_page(f, abbrev_map, bib_df, args)
        print(f"  -> {len(cits)} citations")
        all_c.extend(cits)

    export_c = [c for c in all_c if c.confidence >= args.min_confidence]
    exporters.export_all(export_c, args.output_dir, files)
    hi = sum(1 for c in all_c if c.confidence >= 0.7)
    print(f"\nTotal {len(all_c)} | high-conf {hi} | exported {len(export_c)}")


if __name__ == "__main__":
    main()
