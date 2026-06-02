#!/usr/bin/env python3
"""
Stage 3a — abbreviation lookup (rule, no model)
Stage 3b — cross-reference resolution: ibid. / op.cit. / idem, Latin AND German

These are deterministic. Most cases are pure rules against accumulated state;
only an ambiguous op.cit. (author with multiple cited works) escalates to the LLM.
"""
import re
from state import Citation

# ── Cross-reference token table (Latin + German) ────────────────────────────
# kind -> behaviour:
#   ibid   : copy ALL fields from the immediately preceding citation (override page if given)
#   op-cit : find most recent citation by the SAME author, copy title/year
#   idem   : carry forward only the author from the previous citation
XREF_TOKENS = {
    # ibid. = same work as immediately preceding
    "ibid":   "ibid", "ibid.": "ibid", "ib.": "ibid",
    "ebd":    "ibid", "ebd.": "ibid",          # ebenda (German)
    "ebenda": "ibid",
    # op. cit. = a work by this author cited earlier
    "op.cit":  "op-cit", "op. cit.": "op-cit", "op.cit.": "op-cit",
    "loc.cit": "op-cit", "loc. cit.": "op-cit", "l.c.": "op-cit",
    "a.a.o":   "op-cit", "a.a.o.": "op-cit",   # am angegebenen Ort (German)
    "a. a. o.": "op-cit",
    # idem / eadem = same author as previous (new work)
    "idem": "idem", "id.": "idem", "eadem": "idem",
    "ders": "idem", "ders.": "idem",            # derselbe (German, masc.)
    "dies": "idem", "dies.": "idem",            # dieselbe (German, fem.)
}

# build a regex that finds any cross-ref token (longest-first so "op. cit." wins over "op")
_XREF_PAT = re.compile(
    r"(?<![A-Za-z])(" +
    "|".join(sorted((re.escape(k) for k in XREF_TOKENS), key=len, reverse=True)) +
    r")(?![A-Za-z])",
    re.IGNORECASE,
)


def detect_xref(span: str) -> str:
    """Return the xref kind (ibid|op-cit|idem) if the span is a cross-reference, else ''."""
    m = _XREF_PAT.search(span)
    if not m:
        return ""
    return XREF_TOKENS.get(m.group(1).lower().replace(" ", ""), "")


# ── 3a: abbreviation lookup ──────────────────────────────────────────────────
def _norm_key(s: str) -> str:
    """Normalise an abbreviation token for matching (OCR adds stray dots/spaces)."""
    return re.sub(r"[\.\s]", "", s).upper()


def build_lookup(abbrev_map: dict) -> dict:
    """Pre-normalise the abbreviation map keys once."""
    return {_norm_key(k): v for k, v in abbrev_map.items() if k}


def resolve_abbrev(cit: Citation, lookup: dict):
    """Fill cit.resolved_abbrev from the journal_series token, fuzzy on punctuation/case."""
    js = cit.journal_series.strip()
    if not js:
        return
    key = _norm_key(js)
    if key in lookup:
        cit.resolved_abbrev = lookup[key]


# ── 3b: cross-reference resolution ───────────────────────────────────────────
def _extract_trailing_page(span: str) -> str:
    """Pull a page number that travels with an ibid/op.cit., e.g. 'ibid., p. 45' -> '45'."""
    m = re.search(r"(?:p\.?|S\.?|pp\.?|Sp\.?)\s*(\d+(?:[-–]\d+)?)", span)
    return m.group(1) if m else ""


def resolve_crossref(cit: Citation, prior: list, llm_resolver=None):
    """
    Resolve a cross-reference citation against the prior citations (document order).
    `prior` is state.citations[:cit.order] (everything seen before this one).
    `llm_resolver(cit, candidates)` is called ONLY for ambiguous op.cit.
    """
    kind = detect_xref(cit.span)
    if not kind:
        return
    cit.xref_kind = kind
    if not prior:
        cit.notes = (cit.notes + " | xref with no antecedent").strip(" |")
        return

    if kind == "ibid":
        # same work as the immediately preceding REAL citation
        ref = _last_real(prior)
        if ref is not None:
            _copy_from(cit, ref)
            cit.resolves_to_index = ref.order
            pg = _extract_trailing_page(cit.span)
            if pg:
                cit.pages = pg

    elif kind == "idem":
        # same author as previous, but a new work — carry author only
        ref = _last_real(prior)
        if ref is not None and ref.author:
            cit.author = ref.author
            cit.resolves_to_index = ref.order

    elif kind == "op-cit":
        # a work by this author cited earlier. Need the author to disambiguate.
        author = cit.author.strip()
        if not author:
            # try the previous citation's author as a fallback
            ref = _last_real(prior)
            author = ref.author if ref else ""
        candidates = [p for p in prior if author
                      and _same_author(p.author, author) and p.core_complete()]
        if len(candidates) == 1:
            _copy_from(cit, candidates[0])
            cit.resolves_to_index = candidates[0].order
        elif len(candidates) > 1 and llm_resolver is not None:
            chosen = llm_resolver(cit, candidates)   # ambiguous -> LLM picks
            if chosen is not None:
                _copy_from(cit, chosen)
                cit.resolves_to_index = chosen.order
                cit.notes = (cit.notes + " | op.cit resolved by LLM").strip(" |")
        elif candidates:
            # multiple, no LLM available: take the most recent as best guess
            _copy_from(cit, candidates[-1])
            cit.resolves_to_index = candidates[-1].order
            cit.notes = (cit.notes + " | op.cit ambiguous, took most recent").strip(" |")


def _last_real(prior: list):
    """Most recent citation that is itself not a cross-reference."""
    for p in reversed(prior):
        if not p.xref_kind:
            return p
    return prior[-1] if prior else None


def _copy_from(cit: Citation, ref: Citation):
    for fld in ("author", "year", "title", "journal_series", "volume",
                "pages", "place", "publisher", "resolved_abbrev"):
        if not getattr(cit, fld):
            setattr(cit, fld, getattr(ref, fld))


def _same_author(a: str, b: str) -> bool:
    if not a or not b:
        return False
    # compare on last name (first comma-token or first whitespace token)
    la = a.split(",")[0].split()[0].lower() if a else ""
    lb = b.split(",")[0].split()[0].lower() if b else ""
    return la == lb and len(la) > 1
