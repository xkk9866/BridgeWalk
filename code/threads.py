"""Sentence-level evidence units shared by the walker, the packer and the reader.

A passage `Title: text` is split into sentences (character spans into `text`).  A context item may carry
`sents = {str(pid): [sentence indices]}`; such a passage is rendered as `[Pk] Title: <its selected sentences in
document order>`.  Passages without an entry in `sents` are rendered whole, so every passage-level result file
renders exactly as before."""
import re

_ABBR = {"mr", "mrs", "ms", "dr", "st", "jr", "sr", "inc", "ltd", "co", "corp", "vs", "no", "gen", "col", "lt", "mt",
         "ft", "sgt", "capt", "prof", "rev", "gov", "sen", "rep", "est", "approx", "dept", "univ", "jan", "feb", "mar",
         "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec", "e.g", "i.e", "etc", "u.s", "u.k", "a.d", "b.c",
         "fig", "op", "vol", "ca", "c"}
_CAND = re.compile(r"[.!?][\"'”’)\]]*\s+(?=[\"'“‘(\[]?[A-Z0-9])|\n+")
_cache = {}


def split(text):
    """list of (start, end) character spans of the sentences of `text` (cached by identity of the string)"""
    k = id(text), len(text)
    hit = _cache.get(k)
    if hit is not None and hit[0] is text:
        return hit[1]
    spans, a = [], 0
    for m in _CAND.finditer(text):
        if m.group().startswith("\n"):
            b = m.start()
        else:
            b = m.start() + 1
            w = re.search(r"([A-Za-z.]+)$", text[a:m.start()])
            tokw = w.group(1).lower().strip(".") if w else ""
            if tokw in _ABBR or (len(tokw) == 1 and tokw.isalpha()):
                continue                      # abbreviation or a single initial ("J. R. R."): not a boundary
        if text[a:b].strip():
            spans.append((a, b))
        a = m.end()
    if text[a:].strip():
        spans.append((a, len(text)))
    if not spans:
        spans = [(0, len(text))]
    if len(_cache) > 200000:
        _cache.clear()
    _cache[k] = (text, spans)
    return spans


def sentence_of(text, pos):
    """index of the sentence of `text` containing character offset `pos`"""
    for j, (a, b) in enumerate(split(text)):
        if pos < b:
            return j
    return len(split(text)) - 1


def render_passage(C, p, idx=None):
    t = C.texts[p]
    if idx is None:
        return f"{C.titles[p]}: {t}"
    sp = split(t)
    return f"{C.titles[p]}: " + " ".join(t[sp[j][0]:sp[j][1]].strip() for j in sorted(set(idx)) if j < len(sp))


def render(C, pas, facts=(), sents=None):
    """identical to reader.context(); `sents` maps str(pid) -> selected sentence indices"""
    lines, pos = [], {}
    sents = sents or {}
    for i, p in enumerate(pas):
        pos[p] = i + 1
        lines.append(f"[P{i+1}] " + render_passage(C, p, sents.get(str(p))))
    if facts:
        lines.append("Knowledge-graph facts:")
        for f in facts:
            src = [pos[p] for p in C.f_prov[f] if p in pos]
            tag = f"P{src[0]}" if src else C.titles[C.f_prov[f][0]] if C.f_prov[f] else "?"
            lines.append(f"- ({tag}) {C.facts[f]}")
    return "\n".join(lines)
