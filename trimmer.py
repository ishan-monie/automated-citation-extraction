#!/usr/bin/env python3
"""
Span trimmer — runs AFTER parsing, BEFORE export.

Uses the already-extracted fields (author, journal_series, year) as anchors
to find where the real citation starts and ends within the wide candidate window.
NER (spaCy) assists when fields are empty.

trim_all(citations) mutates each Citation's span in-place so both the
Zotero export and the HTML highlighter automatically use the cleaned version.
"""
import re

# ── optional spaCy ───────────────────────────────────────────────────────────
_NLP = None
def _get_nlp():
    global _NLP
    if _NLP is None:
        try:
            import spacy
            try:    _NLP = spacy.load("xx_ent_wiki_sm")
            except: _NLP = spacy.load("en_core_web_sm")
        except:
            _NLP = False
    return _NLP


# ── helpers ──────────────────────────────────────────────────────────────────
def _find_anchor_start(span: str, author: str, journal: str) -> int:
    """
    Return the char index where the citation actually starts.
    Strategy (in order):
      1. Find the extracted author surname in the span
      2. Find the journal abbreviation
      3. Ask spaCy NER for the first PERSON entity
      4. Fall back to first capitalised word that looks like a surname
    """
    # 0 — detect mid-word OCR fragment at start (lowercase opener = clipped)
    # if span starts with a lowercase letter or looks like a word fragment,
    # skip straight to journal or NER rather than treating the fragment as author
    starts_mid_word = bool(re.match(r"^[a-zäöüé]", span.strip()))

    # 0b — detect prose prefix (span starts with a connective/article word)
    PROSE_OPENERS = r"^(?:In|Im|An|Zu|Zur|Zum|Für|Fir|Von|Der|Die|Das|Den|Dem|Ein|Eine|Nach|Seit|Über|Uber|Durch|Als|Wie|Auch|So|See|For|The|This|With|From|On|At|By)\b"
    starts_with_prose = bool(re.match(PROSE_OPENERS, span.strip(), re.IGNORECASE))

    # 1 — use extracted author surname (only if span doesn't start mid-word or with prose)
    if not starts_mid_word and not starts_with_prose and author and author.strip() not in ("nan", ""):
        surname = author.split(",")[0].split()[0].strip()
        if len(surname) > 2:
            m = re.search(re.escape(surname), span)
            if m:
                left = m.start()
                # walk left to include co-author — but cap at 35 chars
                prefix = span[max(0, left-35):left]
                co = re.search(r"[A-ZÄÖÜ][\wäöü\-]+(?:\s+(?:und|and|&)\s+)?$", prefix)
                if co and (len(prefix) - co.start()) <= 35:
                    left = left - (len(prefix) - co.start())
                return max(0, left)

    # 2 — use extracted journal abbreviation
    if journal and journal.strip() not in ("nan", ""):
        m = re.search(re.escape(journal.split()[0]), span)
        if m:
            # try to extend left to a preceding author token
            prefix = span[max(0, m.start()-50):m.start()]
            am = re.search(r"[A-ZÄÖÜ][\wäöü\-]+(?:,\s*[A-Z]\.?)?\s*$", prefix)
            if am:
                return max(0, m.start() - (len(prefix) - am.start()))
            return m.start()

    # 3 — spaCy PERSON entity
    nlp = _get_nlp()
    if nlp:
        doc = nlp(span[:500])
        persons = [e for e in doc.ents if e.label_ in ("PERSON","PER")]
        if persons:
            return persons[0].start_char

    # 4 — first capitalised surname-like token (≥3 chars, not a stopword)
    STOPWORDS = {"The","This","In","On","As","At","By","For","From","With",
                 "Das","Die","Der","Ein","Eine","Im","In","Zu","Zur","Zum",
                 "Les","Le","La","Un","Une","Du","De"}
    for m in re.finditer(r"\b([A-ZÄÖÜ][a-zäöü]{2,})\b", span):
        if m.group(1) not in STOPWORDS:
            return m.start()

    return 0   # give up, keep full span


def _find_anchor_end(span: str, pages: str, year: str) -> int:
    """
    Return the char index where the citation ends.
    Stops at the last page range or year, then trims trailing prose.
    """
    end = len(span)

    # find the rightmost year token
    for m in re.finditer(r"\(?\b(?:18|19|20)\d{2}(?:/\d{2,4})?\b\)?", span):
        candidate = m.end()
        # extend past a following page range: "(1942) Sp. 165-172" → keep to 172
        tail = span[candidate:candidate+30]
        page_m = re.match(r"[\s,\.]*(?:Sp\.|S\.|p\.|pp\.|f\.|ff\.?)?\s*\d+(?:[-–]\d+)?", tail)
        if page_m:
            candidate += page_m.end()
        end = max(end if end < len(span) else 0, candidate)

    # also anchor on explicit page field if we have it
    if pages and pages.strip() not in ("nan",""):
        pg = pages.split()[0].split("-")[0].strip()
        if pg.isdigit():
            m = re.search(re.escape(pg), span)
            if m:
                end = max(end, m.end())

    # trim trailing punctuation / prose
    tail = span[end:]
    # stop at next sentence-start (capital letter after punctuation)
    prose_m = re.search(r"[.;,]\s+[A-ZÄÖÜ][a-z]", tail)
    if prose_m:
        end += prose_m.start() + 1

    result = span[:end].rstrip(" .,;:")
    return len(result)


def trim_span(span: str, author: str = "", journal: str = "",
              year: str = "", pages: str = "", cite_type: str = "") -> str:
    """
    Return a trimmed version of `span` that starts and ends at the citation boundary.
    Manuscript refs and short spans are returned unchanged.
    """
    if cite_type == "manuscript-ref" or len(span) < 15:
        return span

    start = _find_anchor_start(span, author, journal)
    end   = _find_anchor_end(span, pages, year)

    # sanity: trimmed span must be at least 8 chars and shorter than original
    if end <= start or (end - start) < 8:
        return span

    trimmed = span[start:end].strip(" .,;:()")

    # reject if we trimmed away so much that the year/author are now missing
    if year and year.strip() not in ("nan","") and year[:4] not in trimmed:
        return span    # trim went wrong, keep original

    return trimmed


def trim_all(citations: list):
    """
    Mutate each Citation's span in-place using the extracted fields as anchors.
    Called once after parsing + resolution, before export.
    """
    for c in citations:
        if c.xref_kind:           # cross-refs are already short
            continue
        original = c.span
        c.span = trim_span(
            span=c.span,
            author=c.author,
            journal=c.journal_series,
            year=c.year,
            pages=c.pages,
            cite_type=c.cite_type,
        )
        if c.span != original:
            # log the trim in notes for audit
            c.notes = (c.notes + f" | trimmed from: {original[:60]}").strip(" |")
