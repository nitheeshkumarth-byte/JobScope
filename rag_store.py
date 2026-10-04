"""The candidate's own document library, and BM25 retrieval over it.

WHY THIS EXISTS
Resume tailoring was picking which bullets to show with term overlap against a
job description. That is brittle in a specific way: it can only match words the
posting happens to use, and the posting text is a <=900 char scrape that boards
403 on (see resume_generator.desc_snippet). Retrieval here works the other way
round - the JD is the query, the candidate's own documents are the corpus - and
it returns text VERBATIM.

THE ONE RULE THIS MODULE ENFORCES
A generated resume must never assert something the candidate did not write.
resume_generator already refuses to invent a skills section out of CV prose
(see the note above _rank_skill_groups), because "a resume that asserts skills
the candidate never claimed is worse than one missing a section - it is a false
claim about the person, printed on the document they send to employers." That
rule is much easier to keep when the content pipeline cannot emit a word that was
not stored by the user, so retrieval here returns stored substrings and nothing
else. There is no generation step in this module to hallucinate from.

WHY BM25 AND NOT EMBEDDINGS
The corpus is one-to-a-few documents - hundreds of chunks. At that size BM25 is
competitive with a dense retriever, it needs no model download, it adds no
dependency to a Python 3.14 environment where torch wheels are still settling,
and every score is explainable ("this chunk contains all three of the terms the
posting asked for"). Embeddings become worth their cost at library scale; this
is not library scale. See rag_store.bm25_scores for the ranking itself.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

import auth

# --------------------------------------------------------------------------
# tokenizing
# --------------------------------------------------------------------------

# Tech resumes are full of tokens a naive \w+ split destroys: "C++" becomes
# "c", "C#" becomes "c", ".NET" loses its dot. Those are precisely the terms a
# JD names, so they are folded to word-ish forms and kept whole rather than
# being cut into fragments that match everything.
_FOLDED = (
    ("c++", "cpp"),
    ("c#", "csharp"),
    ("f#", "fsharp"),
    (".net", "dotnet"),
    ("node.js", "nodejs"),
)

# The folded forms are real words, not plurals, so they must survive the stemmer
# below. Without this "nodejs" became "nodej" and stopped matching a posting
# that said Node.js - the exact failure the fold exists to prevent.
_PROTECTED = frozenset(dst for _, dst in _FOLDED)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#.]*")

# Deliberately short. An aggressive list strips terms that carry real signal in
# a job description ("work", "team", "role" are all common in postings), and the
# whole point of this module is to notice the JD's vocabulary.
_STOPWORDS = frozenset("""
a an and are as at be but by for from has have in into is it its of on or
that the their them then there these they this to was were will with
you your we our us
""".split())


def _stem(token: str) -> str:
    """Fold a plural so "databases" matches a posting that says "database".

    Only applied to tokens long enough that it cannot mangle a short acronym:
    "aws" stays "aws" (len 3), "databases" becomes "database". Words ending in
    "ss" are left alone so "class"/"process" survive, and so are the folded
    tech tokens, which are already canonical.
    """
    if token in _PROTECTED:
        return token
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lowercase, fold tech tokens, drop stopwords, light-stem plurals."""
    if not text:
        return []
    low = text.lower()
    for src, dst in _FOLDED:
        low = low.replace(src, dst)
    out = []
    for raw in _TOKEN_RE.findall(low):
        token = raw.strip(".")
        if not token or len(token) < 2 or token in _STOPWORDS:
            continue
        out.append(_stem(token))
    return out


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

# A chunk is roughly ONE resume bullet. That is the unit the tailoring decision
# is actually made at: the JD picks which achievements lead, and two bullets
# welded into one 600-char chunk cannot be chosen between. 260 chars holds a
# typical achievement line whole while keeping neighbouring bullets apart, so
# the shortest ones still group rather than scattering.
CHUNK_CHARS = 260
CHUNK_OVERLAP = 0

_BULLET_RE = re.compile(r"^\s*(?:[-*•·‣–—]|\(?\d{1,2}[.)])\s+")


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit, with enough provenance to trace it to its source."""
    doc_id: str
    doc_name: str
    kind: str
    index: int
    text: str

    @property
    def source_ref(self) -> str:
        return f"{self.doc_name}#{self.index}"


def split_units(text: str) -> list[str]:
    """Split a document into bullet/paragraph units, order preserved.

    Bullets are kept whole: a half-bullet in a generated resume reads as
    truncated, and the sentence fragment that results is usually the part that
    carried the achievement.
    """
    units: list[str] = []
    buf: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            if buf:
                units.append(" ".join(buf))
                buf = []
            continue
        if _BULLET_RE.match(raw):
            if buf:
                units.append(" ".join(buf))
            buf = [line]
        elif buf:
            # A continuation line belongs to the bullet above it.
            buf.append(line)
        else:
            buf = [line]
    if buf:
        units.append(" ".join(buf))
    return [u for u in (x.strip() for x in units) if u]


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS) -> list[str]:
    """One chunk per unit. Units longer than `chunk_chars` are split on word
    boundaries rather than truncated.

    Chunks are deliberately NOT greedily packed up to the budget. Packing merges
    neighbouring units, so a 260-char budget silently welded the summary, the
    skills block and the first achievement into a single chunk - and BM25 then
    scored that blob as one unit, unable to choose the achievement the posting
    actually asked for. Independence is the whole point: the JD selects between
    bullets, so a bullet has to be selectable on its own.

    The cost is that a bare heading ("PROJECTS") becomes its own chunk. That is
    harmless - it carries almost no terms, so it scores near zero and is never
    selected - and it is better than the alternative of fusing it onto an
    achievement and making that achievement unselectable.
    """
    chunks: list[str] = []
    for unit in split_units(text):
        if len(unit) <= chunk_chars:
            chunks.append(unit)
            continue
        words, cur = unit.split(), ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > chunk_chars:
                chunks.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

K1 = 1.5
B = 0.75


@dataclass
class _Doc:
    chunk: Chunk
    tokens: list[str] = field(default_factory=list)
    length: int = 0


def bm25_scores(query: str, docs: list[_Doc]) -> list[float]:
    """Okapi BM25. Returns one score per doc, in the order given.

    Standard formulation: term frequency saturates (k1) so a chunk that repeats
    "python" ten times does not outrank one that mentions it once alongside four
    other matched terms, and length normalisation (b) stops a long chunk from
    winning purely by being long.
    """
    q_terms = tokenize(query)
    if not q_terms or not docs:
        return [0.0] * len(docs)

    avgdl = sum(d.length for d in docs) / len(docs) or 1.0
    n = len(docs)
    df: dict[str, int] = {}
    for d in docs:
        for term in set(d.tokens):
            df[term] = df.get(term, 0) + 1

    # Query terms are de-duplicated before scoring. A posting that says "python
    # python developer" would otherwise score a python chunk twice for one
    # mention of the word, while a chunk containing it once is penalised.
    q_unique = list(dict.fromkeys(q_terms))

    scores = []
    for d in docs:
        freqs: dict[str, int] = {}
        for t in d.tokens:
            freqs[t] = freqs.get(t, 0) + 1
        score = 0.0
        for term in q_unique:
            f = freqs.get(term)
            if not f:
                continue
            idf = _idf(df.get(term, 0), n)
            denom = f + K1 * (1 - B + B * (d.length / avgdl))
            score += idf * (f * (K1 + 1)) / denom
        scores.append(score)
    return scores


def _idf(df: int, n: int) -> float:
    """Smoothed inverse document frequency, floored at zero.

    A term in EVERY chunk carries no discriminating power (it matches every
    candidate equally), and the log here goes negative once df > n/2. Flooring
    at 0 means such terms add nothing instead of actively penalising.
    """
    return max(0.0, math.log(1 + (n - df + 0.5) / (df + 0.5)))


def rank(query: str, chunks: list[Chunk], k: int = 8) -> list[tuple[Chunk, float]]:
    """Top-k chunks by BM25, best first, ties broken by document order.

    Ties keep the order the user uploaded, so a resume does not reshuffle
    equivalent bullets run to run - an unstable order makes two builds of the
    same resume look like different documents.
    """
    if not chunks:
        return []
    docs = [_Doc(chunk=c, tokens=tokenize(c.text), length=0) for c in chunks]
    for d in docs:
        d.length = len(d.tokens)
    scores = bm25_scores(query, docs)
    scored = [(d.chunk, s) for d, s in zip(docs, scores) if s > 0]
    scored.sort(key=lambda pair: -pair[1])
    return scored[:k]


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

MAX_DOC_CHARS = 400_000


def _now() -> str:
    return auth._now()


def add_document(uid: int, doc_id: str, name: str, text: str,
                 kind: str = "cv") -> dict:
    """Store one document for a user. Re-uploading the same id replaces it."""
    body = (text or "").strip()
    if not body:
        raise ValueError("document text is empty")
    if len(body) > MAX_DOC_CHARS:
        raise ValueError(f"document too large: {len(body)} chars "
                         f"(limit {MAX_DOC_CHARS})")
    row = {"id": doc_id, "user_id": uid, "name": name or doc_id,
           "kind": kind or "cv", "text": body, "created_at": _now()}
    with auth.connect() as conn:
        conn.execute(
            "INSERT INTO rag_documents (id, user_id, name, kind, text, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(user_id, id) DO UPDATE SET "
            "name = excluded.name, kind = excluded.kind, text = excluded.text",
            (row["id"], uid, row["name"], row["kind"], row["text"],
             row["created_at"]))
    return {k: v for k, v in row.items() if k != "text"} | {"chars": len(body)}


def list_documents(uid: int) -> list[dict]:
    with auth.connect() as conn:
        rows = conn.execute(
            "SELECT id, name, kind, created_at, LENGTH(text) AS chars "
            "FROM rag_documents WHERE user_id = ? ORDER BY created_at, id",
            (uid,)).fetchall()
    return [dict(r) for r in rows]


def delete_document(uid: int, doc_id: str) -> bool:
    with auth.connect() as conn:
        cur = conn.execute(
            "DELETE FROM rag_documents WHERE user_id = ? AND id = ?",
            (uid, doc_id))
        return cur.rowcount > 0


def all_chunks(uid: int) -> list[Chunk]:
    """Every chunk of every document belonging to this user, in upload order."""
    with auth.connect() as conn:
        rows = conn.execute(
            "SELECT id, name, kind, text FROM rag_documents "
            "WHERE user_id = ? ORDER BY created_at, id", (uid,)).fetchall()
    out: list[Chunk] = []
    for r in rows:
        for i, piece in enumerate(chunk_text(r["text"])):
            out.append(Chunk(doc_id=r["id"], doc_name=r["name"],
                             kind=r["kind"], index=i, text=piece))
    return out


def document_text(uid: int, doc_id: str) -> str:
    """The full stored text of one document, or "" if this user has no such
    document.

    Used to build the resume body when the open profile has no CV of its own:
    the candidate's own uploaded file becomes the content source instead of
    resume_data's placeholder person.
    """
    with auth.connect() as conn:
        row = conn.execute(
            "SELECT text FROM rag_documents WHERE user_id = ? AND id = ?",
            (uid, doc_id)).fetchone()
    return row["text"] if row else ""


def retrieve(uid: int, query: str, k: int = 8) -> list[tuple[Chunk, float]]:
    """Search this user's documents. Returns [] when there is nothing to search."""
    return rank(query, all_chunks(uid), k=k)
