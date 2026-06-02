#!/usr/bin/env python3
"""Exporters for citation extraction v4: CSV, highlighted HTML, Zotero (CSL-JSON + RDF)."""
import os, json, html, re
from dataclasses import asdict
import pandas as pd

# Color per citation class/type for the HTML viewer
TYPE_COLORS = {
    "author-date":       "#E6F1FB",  # blue
    "author-title-full": "#EAF3DE",  # green
    "journal-abbrev":    "#FCEBEB",  # red
    "footnote-ref":      "#FAEEDA",  # amber
    "manuscript-ref":    "#E1F5EE",  # teal
    "series-ref":        "#E1F5EE",
    "ibid":              "#F1EFE8",  # gray
    "vgl-cross-ref":     "#F1EFE8",
    "eq-ref":            "#F1EFE8",
    "jstor-meta":        "#FBEAF0",  # pink
    "title-page":        "#FBEAF0",
    "loc-data":          "#FBEAF0",
}
DEFAULT_COLOR = "#EEEDFE"  # purple


def export_csv(cits, out_dir):
    rows = [asdict(c) for c in cits]
    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "citations.csv")
    df.to_csv(path, index=False)
    return path


def export_csl_json(cits, out_dir):
    """CSL-JSON — directly importable by Zotero (File > Import)."""
    items = []
    for i, c in enumerate(cits):
        if c.citation_class == "document-level" or c.title:
            itype = "book" if c.publisher else "article-journal"
        else:
            itype = "article-journal" if c.journal_series else "manuscript"

        item = {
            "id": f"cite_{i}",
            "type": itype,
            "title": c.title or c.span,
        }
        if c.author:
            # split "Last, First" or "Last und Last"
            authors = []
            for a in re.split(r"\s+und\s+|\s+and\s+|;\s*", c.author):
                a = a.strip()
                if not a:
                    continue
                if "," in a:
                    fam, giv = a.split(",", 1)
                    authors.append({"family": fam.strip(), "given": giv.strip()})
                else:
                    authors.append({"family": a})
            if authors:
                item["author"] = authors
        if c.year:
            yr = re.search(r"\d{4}", c.year)
            if yr:
                item["issued"] = {"date-parts": [[int(yr.group(0))]]}
        if c.journal_series:
            item["container-title"] = c.resolved_abbrev or c.journal_series
        if c.volume:  item["volume"] = c.volume
        if c.pages:   item["page"] = c.pages
        if c.publisher: item["publisher"] = c.publisher
        if c.place:   item["publisher-place"] = c.place
        item["note"] = f"confidence={c.confidence}; type={c.cite_type}; source={c.page_id}"
        items.append(item)

    path = os.path.join(out_dir, "citations_zotero.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    return path


def export_rdf(cits, out_dir):
    """Zotero RDF (legacy import format). CSL-JSON is preferred but RDF kept for compatibility."""
    def esc(s): return html.escape(str(s or ""))
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"',
             '         xmlns:z="http://www.zotero.org/namespaces/export#"',
             '         xmlns:dc="http://purl.org/dc/elements/1.1/"',
             '         xmlns:bib="http://purl.org/net/biblio#"',
             '         xmlns:dcterms="http://purl.org/dc/terms/">']
    for i, c in enumerate(cits):
        is_book = bool(c.publisher)
        tag = "bib:Book" if is_book else "bib:Article"
        lines.append(f'  <{tag} rdf:about="#item_{i}">')
        lines.append(f'    <dc:title>{esc(c.title or c.span)}</dc:title>')
        if c.author:
            lines.append('    <bib:authors><rdf:Seq><rdf:li><foaf:Person '
                         'xmlns:foaf="http://xmlns.com/foaf/0.1/">'
                         f'<foaf:surname>{esc(c.author)}</foaf:surname>'
                         '</foaf:Person></rdf:li></rdf:Seq></bib:authors>')
        if c.year:    lines.append(f'    <dc:date>{esc(c.year)}</dc:date>')
        if c.journal_series:
            lines.append(f'    <dcterms:isPartOf>{esc(c.resolved_abbrev or c.journal_series)}</dcterms:isPartOf>')
        if c.volume:  lines.append(f'    <z:volume>{esc(c.volume)}</z:volume>')
        if c.pages:   lines.append(f'    <bib:pages>{esc(c.pages)}</bib:pages>')
        if c.publisher: lines.append(f'    <dc:publisher>{esc(c.publisher)}</dc:publisher>')
        lines.append(f'    <dc:description>confidence={c.confidence}; {esc(c.cite_type)}</dc:description>')
        lines.append(f'  </{tag}>')
    lines.append('</rdf:RDF>')
    path = os.path.join(out_dir, "citations_zotero.rdf")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path


def export_triples(cits, out_dir):
    """RDF triples (JSON-LD) for citations that carry factual statements."""
    triples = []
    for c in cits:
        subj = getattr(c, "rdf_subject", "")
        pred = getattr(c, "rdf_predicate", "")
        if subj and pred:
            triples.append({
                "@type": "Statement",
                "subject": subj,
                "predicate": pred,
                "object": getattr(c, "rdf_object", ""),
                "citation": c.span,
                "temporal": getattr(c, "rdf_temporal", ""),
                "spatial": getattr(c, "rdf_spatial", ""),
                "confidence": c.confidence,
            })
    path = os.path.join(out_dir, "triples.jsonld")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"@context": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
                   "@graph": triples}, f, ensure_ascii=False, indent=2)
    return path


def export_html(cits, out_dir, source_files):
    """Interactive HTML viewer: re-renders each page with citations highlighted."""
    # group by page
    by_page = {}
    for c in cits:
        by_page.setdefault(c.page_id, []).append(c)

    # rebuild page text from source CSVs and highlight spans
    page_blocks = []
    for src in source_files:
        from pathlib import Path
        pid = Path(src).stem
        try:
            df = pd.read_csv(src)
            tcol = "text" if "text" in df.columns else df.columns[1]
            segs = [str(t) for t in df[tcol].dropna() if len(str(t).strip()) >= 30]
        except Exception:
            segs = []
        page_text = "\n\n".join(segs)
        page_cits = by_page.get(pid, [])

        # highlight: sort spans by length desc to avoid nested replacement.
        # Use a placeholder-token approach so we don't re-highlight inside
        # an already-inserted <mark> tag, and tolerate OCR whitespace variance.
        placeholders = {}
        working = page_text  # work on RAW text, escape at the end
        for idx, c in enumerate(sorted(page_cits, key=lambda x: -len(x.span))):
            span = c.span.strip()
            if len(span) < 2:
                continue
            # build a whitespace-tolerant regex: collapse any run of
            # whitespace in the span to \s+ so OCR spacing variance matches
            parts = re.split(r"\s+", span)
            pat = r"\s+".join(re.escape(p) for p in parts if p)
            if not pat:
                continue
            try:
                rx = re.compile(pat)
            except re.error:
                continue
            m = rx.search(working)
            if not m:
                continue
            token = f"\x00\x01{idx}\x02\x00"
            placeholders[token] = c
            working = working[:m.start()] + token + working[m.end():]

        # now escape the whole thing, then swap placeholders for <mark> tags
        highlighted = html.escape(working)
        for token, c in placeholders.items():
            color = TYPE_COLORS.get(c.cite_type, DEFAULT_COLOR)
            tooltip = html.escape(
                f"{c.cite_type} | conf {c.confidence} | "
                f"{c.author} {c.year} {c.journal_series} {c.volume} {c.pages}".strip()
                + (f" -> {c.resolved_abbrev}" if c.resolved_abbrev else "")
            )
            mark = (f'<mark class="cit cls-{c.citation_class}" '
                    f'style="background:{color}" title="{tooltip}" '
                    f'data-conf="{c.confidence}">{html.escape(c.span.strip())}</mark>')
            highlighted = highlighted.replace(html.escape(token), mark, 1)

        page_blocks.append(
            f'<section class="page"><h2>{html.escape(pid)} '
            f'<span class="count">{len(page_cits)} citations</span></h2>'
            f'<div class="text">{highlighted}</div></section>'
        )

    total = len(cits)
    hi = sum(1 for c in cits if c.confidence >= 0.7)

    doc = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Citation Extraction — Highlighted</title>
<style>
  body {{ font-family: Georgia, serif; max-width: 900px; margin: 0 auto;
         padding: 2rem; line-height: 1.7; color: #1a1a1a; background:#faf9f6; }}
  h1 {{ font-family: -apple-system, sans-serif; }}
  .summary {{ background:#fff; border:1px solid #ddd; border-radius:8px;
              padding:1rem; margin-bottom:1.5rem; font-family:sans-serif; font-size:.9rem; }}
  .controls {{ font-family:sans-serif; margin-bottom:1rem; font-size:.85rem; }}
  .page {{ background:#fff; border:1px solid #e2e0d8; border-radius:8px;
           padding:1.5rem; margin-bottom:1.5rem; }}
  .page h2 {{ font-family:sans-serif; font-size:1rem; border-bottom:1px solid #eee;
              padding-bottom:.5rem; }}
  .count {{ color:#888; font-weight:normal; font-size:.8rem; float:right; }}
  .text {{ white-space:pre-wrap; font-size:.95rem; }}
  mark.cit {{ border-radius:3px; padding:1px 2px; cursor:help; }}
  mark.cls-document-level {{ outline:2px dashed #999; }}
  .legend {{ font-family:sans-serif; font-size:.8rem; display:flex; gap:1rem;
             flex-wrap:wrap; margin-top:.5rem; }}
  .legend span {{ padding:2px 8px; border-radius:3px; }}
  .dim mark[data-conf]:not([data-conf^="0.7"]):not([data-conf="1.0"]):not([data-conf^="0.8"]):not([data-conf^="0.9"]) {{ opacity:.35; }}
</style></head>
<body>
<h1>Citation Extraction — Highlighted Viewer</h1>
<div class="summary">
  <strong>{total}</strong> citations extracted &nbsp;|&nbsp;
  <strong>{hi}</strong> high-confidence (≥0.7)
  <div class="legend">
    <span style="background:{TYPE_COLORS['author-date']}">author-date</span>
    <span style="background:{TYPE_COLORS['author-title-full']}">full citation</span>
    <span style="background:{TYPE_COLORS['journal-abbrev']}">journal/abbrev</span>
    <span style="background:{TYPE_COLORS['footnote-ref']}">footnote</span>
    <span style="background:{TYPE_COLORS['manuscript-ref']}">manuscript</span>
    <span style="background:{TYPE_COLORS['jstor-meta']}">document-level</span>
  </div>
</div>
<div class="controls">
  <label><input type="checkbox" onchange="document.body.classList.toggle('dim',this.checked)">
  Dim low-confidence (&lt;0.7)</label>
  &nbsp;&nbsp;Hover any highlight for extracted fields.
</div>
{''.join(page_blocks)}
</body></html>"""

    path = os.path.join(out_dir, "viewer.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


def export_all(cits, out_dir, source_files):
    print("  Writing exports:")
    for fn in [
        lambda: export_csv(cits, out_dir),
        lambda: export_csl_json(cits, out_dir),
        lambda: export_rdf(cits, out_dir),
        lambda: export_triples(cits, out_dir),
        lambda: export_html(cits, out_dir, source_files),
    ]:
        p = fn()
        print(f"    {p}")
