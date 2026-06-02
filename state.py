#!/usr/bin/env python3
"""Shared state for the citation extraction graph.

The whole point of the graph (vs the old linear script) is that every node
reads and writes ONE state object, so later nodes can look backward — which is
what makes ibid./op.cit. resolution and author de-duplication possible.
"""
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class Citation:
    # raw
    span: str
    citation_class: str = "inline"        # inline | document-level
    cite_type: str = ""                   # author-date | journal-abbrev | manuscript-ref | ...
    # parsed fields
    author: str = ""
    year: str = ""
    title: str = ""
    journal_series: str = ""
    volume: str = ""
    pages: str = ""
    place: str = ""
    publisher: str = ""
    language: str = ""
    # resolution
    resolved_abbrev: str = ""             # 3a: journal_series expanded via the map
    resolves_to_index: int = -1           # 3b: index of the citation an ibid/op.cit points to
    xref_kind: str = ""                   # ibid | op-cit | idem | "" (not a cross-ref)
    author_cluster_id: int = -1           # 5: disambiguated author identity
    bib_matched: bool = False
    # provenance
    page_id: str = ""
    segment_id: int = -1
    seg_type: str = ""
    order: int = -1                       # document order (critical for cross-ref)
    # quality
    confidence: float = 0.0
    parsed_by: str = ""                   # crf | llm | regex  (which engine produced fields)
    critic_verdict: str = ""              # keep | reject | (empty = not reviewed)
    notes: str = ""

    def core_complete(self) -> bool:
        return bool(self.author and (self.journal_series or self.title))


@dataclass
class CitationState:
    page_id: str = ""
    raw_text: str = ""
    segments: list = field(default_factory=list)     # list[dict]: {segment_id, text, seg_type}
    candidates: list = field(default_factory=list)    # list[dict]: candidate spans before parsing
    citations: list = field(default_factory=list)     # list[Citation], in document order
    abbrev_map: dict = field(default_factory=dict)    # merged RlA + harvested
    bib_df = None                                      # pandas DataFrame or None
    # config knobs
    critic_threshold: float = 0.6     # only run the LLM critic below this score
    llm_parse_threshold: float = 0.5  # escalate to LLM parse below this CRF confidence
    retries: int = 0
    max_retries: int = 1

    def add(self, c: Citation):
        c.order = len(self.citations)
        self.citations.append(c)
        return c
