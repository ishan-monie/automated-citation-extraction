#!/usr/bin/env python3
"""
Citation completion checker.

Runs after candidate detection, before parsing. If a candidate span looks
structurally incomplete (open parenthesis, year without close, journal+volume
without page locator), it extends the window forward in the source text until
the citation reaches a natural end.

This is the cheap regex fix for spans like:
  "Ebeling, AfK 2 (1924"  →  "Ebeling, AfK 2 (1924/25) Sp. 28-30"

A bi-LSTM sequence tagger could replace or augment this for the harder cases
where the completion pattern is less formulaic, but for the structured
journal-citation format (Author, ABBR vol (year) pages) regex is sufficient.
"""
import re


# ── Completion signals — what a COMPLETE citation end looks like ──────────────
# Ordered: most specific first
COMPLETE_END_PATTERNS = [
    # page range: Sp./p./pp./S./f./ff. + digits (at least 2) + optional range + boundary
    r"(?:Sp\.|S\.|p\.|pp\.|f\.|ff\.?)\s*\d{2,}(?:[-–]\d+)?(?=\D|$)",
    # bare page range after year: (1942) 165-172 + boundary
    r"\(\d{4}(?:/\d{2,4})?\)\s*\d{2,}(?:[-–]\d+)?(?=\D|$)",
    # year in parens with no page — acceptable end for short refs
    r"\(\d{4}(?:/\d{2,4})?\)\.?\s*$",
]

# Signs that a span is INCOMPLETE
INCOMPLETE_SIGNALS = [
    r"\(\d{4}(?:/\d{2})?\s*$",          # year paren never closed: "(1924"
    r"\(\s*$",                           # open paren at end
    r"\b[A-Z]{2,6}\s+\d+\s*$",          # journal + volume, no year yet: "AfK 2"
    r"\b[A-Z]{2,6}\s+\d+\s*\([^)]*$",  # journal + vol + unclosed year: "AfK 2 (1924"
    r",\s*$",                            # ends with comma (more to come)
    r"\bSp\.?\s*$",                      # truncated Sp. abbreviation
    r"\bpp?\.\s*$",                      # truncated p./pp.
    r"\bS\.\s*$",                        # truncated S. (Seite)
]


def _is_weak_complete(span: str) -> bool:
    """Year-in-parens with no page locator — acceptable only if nothing more follows."""
    return bool(re.search(r"\(\d{4}(?:/\d{2,4})?\)\.?\s*$", span.rstrip()))


def _has_page_locator(span: str) -> bool:
    return bool(re.search(
        r"(?:Sp\.|S\.|p\.|pp\.|f\.|ff\.?|col\.)\s*\d+", span))


def _is_complete(span: str) -> bool:
    s = span.rstrip()
    # strong completion: explicit page locator with digits
    strong = [
        r"(?:Sp\.|S\.|p\.|pp\.|f\.|ff\.?)\s*\d+(?:[-–]\d+)?",
        r"\(\d{4}(?:/\d{2,4})?\)\s*\d+(?:[-–]\d+)?",   # year + bare page range
    ]
    if any(re.search(p, s) for p in strong):
        return True
    # weak completion: year only — accept ONLY if no page-locator abbreviation
    # follows within 25 chars of the end (i.e. nothing more is coming)
    if _is_weak_complete(s):
        return True   # caller must check context; extension will stop here
    return False

# German/Latin page locator abbreviations to look for when extending
PAGE_LOCS = r"(?:Sp\.|S\.|p\.|pp\.|f\.|ff\.?|col\.|cols\.?|Nr\.?)"
YEAR_PAT  = r"\(?\d{4}(?:/\d{2,4})?\)?"


def _is_incomplete(span: str) -> bool:
    s = span.rstrip()
    return any(re.search(p, s) for p in INCOMPLETE_SIGNALS)


def _is_complete(span: str) -> bool:
    s = span.rstrip()
    return any(re.search(p, s) for p in COMPLETE_END_PATTERNS)


def _extend_span(span: str, full_text: str, max_extend: int = 80) -> str:
    """
    Find `span` in `full_text` and extend it forward to capture the complete
    citation structure using a forward-looking regex rather than char-by-char.
    """
    idx = full_text.find(span)
    if idx == -1:
        prefix = span[:min(30, len(span))]
        idx = full_text.find(prefix)
        if idx == -1:
            return span

    end = idx + len(span)
    tail = full_text[end: end + max_extend]

    # Try to grab the remaining citation pieces in one regex:
    # optional year completion, optional page locator + number + optional range
    # Handles: /25) Sp. 165-172  |  ) 323  |  ) S. 15  |  /53) 323
    completion_pat = re.compile(
        r"\.?"                     # optional leading dot (for truncated "Sp" -> ". 165")
        r"(?:/\d{2,4})?"           # optional year suffix: /25 or /53
        r"\)?"                     # optional closing paren for year
        r"(?:"                     # optional page locator block
          r"\s*(?:Sp\.|S\.|p\.|pp\.|f\.|ff\.?|col\.?)?"  # locator abbrev
          r"\s*\d+(?:[-–]\d+)?"   # page number(s)
          r"(?:\s*(?:und|and|,)\s*\d+(?:[-–]\d+)?)?"     # optional second range
        r")?"
    )

    m = completion_pat.match(tail)
    if m and m.group(0) and _is_incomplete(span):
        extended = (span + m.group(0)).rstrip(" .,;:")
        if len(extended) > len(span) + 1:
            return extended

    return span


def complete_candidates(candidates: list, segment_text: str) -> list:
    """
    For each candidate whose span looks incomplete, try to extend it
    forward in segment_text. Returns the same list with spans updated in-place.
    """
    for cand in candidates:
        span = cand["span"]
        if cand.get("hint_type") == "manuscript-ref":
            continue   # manuscript refs are always complete (they're just "K 5923")
        if _is_incomplete(span) and not _is_complete(span):
            extended = _extend_span(span, segment_text)
            if extended != span:
                cand["span"] = extended
                cand["_completed"] = True
    return candidates
