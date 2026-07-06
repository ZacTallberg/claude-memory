"""Markdown -> HTML for note/snippet display."""
from __future__ import annotations

from markdown_it import MarkdownIt

_md = (MarkdownIt("commonmark", {"html": False, "linkify": True, "typographer": True})
       .enable("table").enable("strikethrough"))


def md_to_html(text: str) -> str:
    return _md.render(text or "")
