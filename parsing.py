#!/usr/bin/env python3
"""
Stage 3 — reference field parser. Tries GROBID (if a server is reachable),
          falls back to the LLM. A bi-LSTM-CRF/AnyStyle parser can be added
          behind the same `parse_fields` interface.
Stage 4 — LLM hard-case parse (only invoked on low-confidence / garbled spans).
Stage 6 — LLM critic: keep/reject + confidence. Only runs below a threshold.

All LLM calls go through one OpenAI-compatible client so you can point at
Groq / LM Studio / ngrok by changing env vars.
"""
import os, re, json, time
from state import Citation

API_URL  = os.environ.get("LLM_API_URL", "https://api.groq.com/openai/v1")
MODEL    = os.environ.get("LLM_MODEL", "llama-3.1-8b-instant")
GROBID_URL = os.environ.get("GROBID_URL", "")
RATE_LIMIT_SLEEP = float(os.environ.get("RATE_LIMIT_SLEEP", "1.2"))

def _api_key():
    """Read key lazily so setting the env var after import still works."""
    return os.environ.get("LLM_API_KEY", os.environ.get("NGROK_KEY", ""))

_client = None
def _get_client():
    global _client
    key = _api_key()
    if _client is None or not key:
        from openai import OpenAI
        _client = OpenAI(base_url=API_URL, api_key=key)
    return _client


def _parse_json(raw: str):
    if not raw:
        return None
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
    i = raw.find("{")
    j = raw.rfind("}")
    if i != -1 and j != -1:
        try:
            return json.loads(raw[i:j+1])
        except Exception:
            pass
    return None


# ── Stage 3: GROBID parser (optional) ────────────────────────────────────────
def grobid_parse(raw_ref: str):
    """Parse one reference string via a running GROBID server. Returns dict or None."""
    if not GROBID_URL:
        return None
    try:
        import requests
        r = requests.post(
            f"{GROBID_URL}/api/processCitation",
            data={"citations": raw_ref, "consolidateCitations": "0"},
            timeout=20,
        )
        if r.status_code != 200 or not r.text.strip():
            return None
        return _tei_to_fields(r.text)
    except Exception:
        return None


def _tei_to_fields(tei: str) -> dict:
    """Minimal TEI -> fields (avoids a heavy XML dep; GROBID TEI is shallow here)."""
    def grab(tag, attr=None, val=None):
        if attr:
            m = re.search(rf'<{tag}[^>]*{attr}="{val}"[^>]*>(.*?)</{tag}>', tei, re.S)
        else:
            m = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", tei, re.S)
        return re.sub(r"<[^>]+>", "", m.group(1)).strip() if m else ""
    surnames = re.findall(r"<surname>(.*?)</surname>", tei)
    return {
        "author": "; ".join(surnames),
        "title": grab("title", "level", "a") or grab("title"),
        "journal_series": grab("title", "level", "j"),
        "year": grab("date"),
        "volume": grab("biblScope", "unit", "volume"),
        "pages": grab("biblScope", "unit", "page"),
        "publisher": grab("publisher"),
    }


# ── Stage 3/4: LLM parse ─────────────────────────────────────────────────────
PARSE_SYS = """You parse ONE bibliographic reference (possibly OCR-noisy, English/German/French)
into fields. Return ONLY JSON, no prose:
{"author":"","year":"","title":"","journal_series":"","volume":"","pages":"","place":"","publisher":"","language":""}
Capture the COMPLETE reference. Preserve diacritics. Use "" for absent fields. Do not invent."""

def llm_parse(raw_ref: str, hint_type: str = "") -> dict:
    if not _api_key():
        return {}
    try:
        time.sleep(RATE_LIMIT_SLEEP)
        resp = _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": PARSE_SYS},
                      {"role": "user", "content": f"type hint: {hint_type}\nREFERENCE:\n{raw_ref[:1200]}"}],
            max_tokens=400, temperature=0.0,
        )
        return _parse_json(resp.choices[0].message.content) or {}
    except Exception as e:
        return {"_error": str(e)[:80]}


def parse_fields(candidate: dict, prefer_grobid: bool = True) -> Citation:
    """Stage 3 entry point: GROBID first (if configured), else LLM. Sets parsed_by + a base confidence."""
    span = candidate["span"]
    hint = candidate.get("hint_type", "")
    cls  = "document-level" if hint == "document-level" else "inline"

    fields, engine = None, ""
    if hint != "manuscript-ref":      # manuscript refs don't need parsing
        if prefer_grobid:
            fields = grobid_parse(span)
            engine = "crf" if fields else ""
        if not fields:
            fields = llm_parse(span, hint)
            engine = "llm" if fields and "_error" not in fields else engine

    fields = fields or {}
    c = Citation(
        span=span, citation_class=cls, cite_type=hint,
        author=fields.get("author", ""), year=str(fields.get("year", "")),
        title=fields.get("title", ""), journal_series=fields.get("journal_series", ""),
        volume=str(fields.get("volume", "")), pages=str(fields.get("pages", "")),
        place=fields.get("place", ""), publisher=fields.get("publisher", ""),
        language=fields.get("language", ""), parsed_by=engine or "regex",
    )
    if hint == "manuscript-ref":
        m = re.match(r"([A-Z][A-Za-z]{0,3})\s+(\d+)", span)
        if m:
            c.journal_series, c.pages = m.group(1), m.group(2)
    return c


# ── Stage 6: critic ──────────────────────────────────────────────────────────
CRITIC_SYS = """You are a strict reviewer of extracted bibliographic citations.
For the given citation span + parsed fields, decide if it is a REAL bibliographic
citation or noise (OCR garbage, a page/line marker like "Z. 80" or "S. 45", a
fragment, or a stray number). Return ONLY JSON:
{"verdict":"keep|reject","reason":"short","fix":{"author":"","year":"","title":"","journal_series":"","volume":"","pages":""}}
Put corrected values in "fix" ONLY for fields you can confidently improve; leave others "".
Reject anything that is not actually a reference to a work."""

def critic_review(cit: Citation) -> dict:
    if not _api_key():
        return {"verdict": "keep", "reason": "no-llm"}
    payload = (f'span: {cit.span}\nauthor: {cit.author}\nyear: {cit.year}\n'
               f'title: {cit.title}\njournal_series: {cit.journal_series}\n'
               f'volume: {cit.volume}\npages: {cit.pages}\ntype: {cit.cite_type}')
    try:
        time.sleep(RATE_LIMIT_SLEEP)
        resp = _get_client().chat.completions.create(
            model=MODEL,
            messages=[{"role": "system", "content": CRITIC_SYS},
                      {"role": "user", "content": payload}],
            max_tokens=300, temperature=0.0,
        )
        return _parse_json(resp.choices[0].message.content) or {"verdict": "keep", "reason": "parse-fail"}
    except Exception as e:
        return {"verdict": "keep", "reason": f"err:{str(e)[:40]}"}


# ── ambiguous op.cit. resolver (used by resolution.resolve_crossref) ─────────
def make_llm_opcit_resolver():
    """Return a resolver(cit, candidates)->chosen|None that asks the LLM which work op.cit. means."""
    def resolver(cit, candidates):
        if not _api_key():
            return candidates[-1] if candidates else None
        opts = "\n".join(f"{i}: {c.author} {c.year} {c.title or c.journal_series} {c.volume}"
                         for i, c in enumerate(candidates))
        try:
            time.sleep(RATE_LIMIT_SLEEP)
            resp = _get_client().chat.completions.create(
                model=MODEL,
                messages=[{"role": "system", "content":
                           'Pick which earlier work an "op. cit." refers to. Return ONLY {"index": N}.'},
                          {"role": "user", "content": f'op.cit span: {cit.span}\ncandidates:\n{opts}'}],
                max_tokens=40, temperature=0.0,
            )
            d = _parse_json(resp.choices[0].message.content) or {}
            idx = int(d.get("index", -1))
            return candidates[idx] if 0 <= idx < len(candidates) else candidates[-1]
        except Exception:
            return candidates[-1] if candidates else None
    return resolver
