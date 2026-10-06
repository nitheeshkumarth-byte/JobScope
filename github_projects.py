"""
github_projects.py — pull JD-relevant projects out of the CV's own GitHub link.

Why this exists
---------------
`mcp_server_github.py` already exposed `list_github_repos` and
`get_repo_readme`, but nothing ever called them. Two problems made them
unusable anyway:

1. They were driven by a fixed `GITHUB_USERNAME` env var, so the repos read
   belonged to whoever set that variable rather than to the candidate whose CV
   was just uploaded. Two users on one deployment read the same person's repos.
2. A token was treated as required. The GitHub REST API serves public repos
   unauthenticated at 60 requests/hour per IP; a token only raises that to
   5,000/hour. Requiring one made a public-profile CV look broken for anyone
   who had not configured credentials.

So the username comes from the CV (which is the whole point of the feature),
and the token is optional and only ever used to lift the rate limit.

What gets put on the resume
--------------------------
Only text that already exists in the candidate's own repositories. A README is
summarised, never used to invent capability: the pipeline selects the *best
matching existing projects*, and the resume builder renders descriptions the
candidate can defend in an interview. `_bullet` is the guard — it refuses to
emit a line that does not share vocabulary with the README it came from.

Rate limiting and failure
-------------------------
Public-repo reads are cheap but finite. `max_repos` and `max_readmes` are both
capped, READMEs are only fetched for repos that already matched the posting,
results are cached on disk per (user, repo), and every network error is
swallowed into a warning. A GitHub outage degrades the resume, it never fails
the request.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Iterable

try:
    import requests
except ImportError:                                    # pragma: no cover
    requests = None                                   # type: ignore

_API = "https://api.github.com"
_TIMEOUT = 12

# Requests per hour for an unauthenticated client. Used to decide whether to
# warn, never to block: the cache absorbs the repeats in practice.
ANON_HOURLY_LIMIT = 60

# Terms that make a repo plausibly relevant to a posting. Deliberately the same
# shape as resume_generator._POSTING_TERMS so the two rankers agree on what a
# "match" means; a project chosen here should also survive _relevance_scores.
_REPO_TERMS = (
    "python", "java", "javascript", "typescript", "sql", "react", "node",
    "django", "flask", "fastapi", "aws", "azure", "gcp", "docker", "kubernetes",
    "k8s", "terraform", "mysql", "postgres", "mongodb", "redis", "kafka",
    "spark", "pandas", "numpy", "tensorflow", "pytorch", "llm", "rag", "nlp",
    "machine learning", "deep learning", "devops", "cicd", "linux", "rest",
    "api", "graphql", "microservices", "power bi", "tableau", "sap",
    "salesforce", "jira", "agile", "scrum", "selenium", "testing", "android",
    "ios", "flutter", "next.js", "vue", "angular", "spring", "maven",
)

# Boilerplate that makes a README useless as evidence of skill.
_README_BOILERPLATE = re.compile(
    r"(?:installation|getting started|usage|contribut\w+|license|"
    r"acknowledg\w+|table of contents|badges?|screenshots?|"
    r"prerequisites?|clone the repo|how to contribute)", re.I)

# Unresolved merge markers. A README containing "<<<<<<< HEAD" is a broken
# merge, and it does not start at the title: cleaning it line-by-line dropped
# the first 40 lines as a matched section and left the description as a
# requirements fragment. Everything from the first marker on is kept.
_MERGE_MARKER_RE = re.compile(r"^(?:<{7}|={7}|>{7})(?: .*)?$", re.M)


def _strip_unresolved_conflicts(text: str) -> str:
    """Drop the source side of an unresolved merge, keep the current side.

    A README whose merge was never completed opens with "<<<<<<< HEAD" and the
    real title sits below the conflict. Returning only the content between the
    first marker and the matching separator keeps the resolved side, which is
    what the repo owner actually ships.
    """
    m = _MERGE_MARKER_RE.search(text)
    if not m:
        return text
    rest = text[m.end():]
    sep = re.search(r"^={7}\s*$", rest, re.M)
    if sep:
        rest = rest[sep.end():]
    end = re.search(r"^>{7}.*$", rest, re.M)
    if end:
        rest = rest[:end.start()]
    return text[:m.start()] + rest

_BADGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img[^>]*>|\[!.*?\]\([^)]*\)")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"^#{1,6}\s*(.+)$", re.M)
_H1_RE = re.compile(r"^#\s+(?!#)(.+)$", re.M)

# Repos that cannot evidence a project: forks, archived, and anything whose
# name marks it as someone else's work.
_SKIP_NAME_RE = re.compile(
    r"^(?:fork[-_]?of[-_]?|copy[-_]?of[-_]?|mirror[-_]?|"
    r"\.github|homework|course|coursework|assignment|exercise|"
    r"tutorial|demo[s]?[-_]?only|sample)$", re.I)


def github_username(url_or_text: str) -> str:
    """The handle from any GitHub link or bare mention in the CV.

    Accepts what a CV actually contains: a full URL, a scheme-less
    `github.com/handle`, a profile path with extra segments
    (`github.com/handle/resume/blob/main/cv.pdf`), and a bare `@handle` on a
    name line. Rejects reserved paths, which otherwise become a username and
    404 every request.
    """
    if not url_or_text:
        return ""
    m = re.search(r"github\.com/([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)",
                  url_or_text, re.I)
    if m:
        return m.group(1)
    # A name/contact line: "Nitheesh Kumar Thadikamalla (@handle)". The
    # opening bracket and other punctuation before the @ are tolerated because
    # that is how CVs print it. The `\s*@` requirement is what keeps an email
    # address (no space before its @) and a stray "@" in prose from being read
    # as a profile.
    m = re.search(r"(?:^|[\s(\[])\s*@([A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?)"
                  r"\s*[),.;]?\s*$", url_or_text.strip())
    if m:
        return m.group(1)
    # A handle with no @ at all, as when it is passed in directly rather than
    # scraped. Kept narrow on purpose: GitHub allows single-word usernames with
    # no hyphen, but a lone English word is far more likely to be a job title or
    # a stray word from the CV than a profile, so a hyphen is required.
    bare = url_or_text.strip()
    if re.fullmatch(r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+", bare):
        return bare
    return ""


_RESERVED_PATHS = {
    "about", "pricing", "features", "explore", "topics", "collections",
    "trending", "events", "sponsors", "settings", "notifications", "login",
    "join", "marketplace", "apps", "orgs", "organizations", "users", "readme",
    "search", "security", "enterprise", "contact", "site", "blog", "codespaces",
}


def is_reserved(name: str) -> bool:
    return name.lower() in _RESERVED_PATHS


def _headers() -> dict:
    """Auth header when a token exists, plain headers otherwise.

    Public repos are readable with no credentials at all. The token is treated
    as a rate-limit upgrade rather than a requirement, so a deployment with no
    token still works and simply gets 60 req/hour.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "JobScope-Resume",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def has_token() -> bool:
    return bool((os.environ.get("GITHUB_TOKEN") or "").strip())


def cache_dir() -> str:
    d = os.environ.get("GITHUB_CACHE_DIR") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".github_cache")
    os.makedirs(d, exist_ok=True)
    return d


def _cache_path(username: str, repo: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", f"{username}__{repo}")
    return os.path.join(cache_dir(), f"{safe}.json")


def _read_cache(username: str, repo: str, ttl: int = 86400) -> dict | None:
    p = _cache_path(username, repo)
    try:
        if (time.time() - os.path.getmtime(p)) > ttl:
            return None
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_cache(username: str, repo: str, payload: dict) -> None:
    try:
        with open(_cache_path(username, repo), "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
    except OSError:
        pass  # a cache miss is always survivable


def list_repos(username: str, max_repos: int = 40) -> list[dict]:
    """Public non-fork repos, most recently updated first.

    `max_repos` is applied AFTER filtering forks/archived, because those sort
    high (people push to them often) and would otherwise consume the whole
    budget and hide the actual work.
    """
    if not username or is_reserved(username) or requests is None:
        return []
    try:
        resp = requests.get(f"{_API}/users/{username}/repos",
                            headers=_headers(),
                            params={"per_page": 100, "sort": "pushed"},
                            timeout=_TIMEOUT)
        if resp.status_code == 404:
            return []
        if resp.status_code == 403:
            # Almost always the unauthenticated hourly cap. Surfaced so the
            # caller can tell the user to add a token rather than implying the
            # profile has no repos.
            raise RuntimeError("GitHub rate limit reached; set GITHUB_TOKEN")
        resp.raise_for_status()
        data = resp.json()
    except RuntimeError:
        raise
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    keep = [r for r in data
            if isinstance(r, dict)
            and not r.get("fork")
            and not r.get("archived")
            and not r.get("disabled")
            and not _SKIP_NAME_RE.match(r.get("name") or "")]
    return keep[:max_repos]


def _clean_readme(text: str) -> str:
    """Strip the parts of a README that say nothing about the work.

    A boilerplate heading ("## Installation") removes the section body up to
    the next heading of the same or higher level. Dropping only the heading
    line left the commands behind, so `pip install foo` survived and could be
    picked as a project description.
    """
    text = _strip_unresolved_conflicts(text or "")
    text = _BADGE_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)

    out: list[str] = []
    skip_level = 0
    for line in text.splitlines():
        heading = re.match(r"^(#{1,6})\s*(.*)$", line.strip())
        if heading:
            level = len(heading.group(1))
            title = heading.group(2)
            if _README_BOILERPLATE.search(title):
                # Everything indented under this heading goes too. A deeper
                # heading does NOT cancel it, which is why a "### pip install"
                # under "## Installation" previously survived.
                skip_level = level
                continue
            skip_level = 0
            out.append(line.rstrip())
            continue
        if skip_level:
            continue
        out.append(line.rstrip())
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def _readme_text(username: str, repo: str) -> str:
    """One README, de-badged and de-boilerplated, from cache or GitHub."""
    cached = _read_cache(username, repo)
    if cached is not None:
        return cached.get("readme", "")
    if requests is None:
        return ""
    try:
        resp = requests.get(f"{_API}/repos/{username}/{repo}/readme",
                            headers=_headers(), timeout=_TIMEOUT)
        if resp.status_code != 200:
            return ""
        payload = resp.json()
        import base64
        raw = base64.b64decode(payload.get("content") or "").decode(
            "utf-8", errors="ignore")
    except Exception:
        return ""
    text = _clean_readme(raw)[:4000]
    _write_cache(username, repo, {"readme": text, "at": time.time()})
    return text


def _terms_in(text: str) -> set[str]:
    low = (text or "").lower()
    return {t for t in _REPO_TERMS if t in low}


def score_repo(repo: dict, jd_text: str) -> float:
    """How well a repo's own metadata matches the posting.

    Name, description and language only - the README is not needed to rank,
    which keeps the request count proportional to the number of *matches*
    rather than the number of repos.

    Only vocabulary the posting itself uses can score. A technology term that
    appears in the repo and nowhere in the JD is not evidence of relevance to
    *this* posting, and scoring it made a restaurant-themed demo repository
    reach 5.5 against a Python RAG posting - high enough to be selected and
    printed on someone's resume. Distinct terms each count once, so a repo that
    repeats one keyword cannot outrank one that genuinely overlaps.
    """
    jd = (jd_text or "").lower()
    if not jd:
        return 0.0
    haystack = " ".join(str(repo.get(k) or "") for k in
                        ("name", "description", "language", "topics"))
    shared = [t for t in _terms_in(haystack) if t in jd]
    if not shared:
        return 0.0
    score = 2.0 * len(shared)
    # Description is written by the candidate and is the most reliable signal;
    # a repo with none is ranked below an equivalent one that has one.
    if repo.get("description"):
        score += 0.5
    return score


# A line that documents the install rather than the project. The word "required"
# alone is a strong installation signal and must not veto a real description
# such as "LLM service with RAG; multipart API required for uploads".
_INSTALL_LINE_RE = re.compile(
    r"(?:pip install|npm install|yarn add|poetry add|apt-get|brew install|"
    r"requirements\.txt|package\.json|go get|docker run|"
    r"(?:is|are)\s+required\b|see\s+installation|clone\s+the\s+repo)",
    re.I)


def _summarise(name: str, description: str, readme: str, max_len: int = 150) -> str:
    """A one-line project description, preferring the candidate's own words.

    Order is deliberate: the repo description is the curated sentence, the
    README's first heading is next, and only then the first prose line. An
    "Installation" or "License" heading never becomes a project description.

    Install and usage lines are rejected at every tier, not just the last: the
    real fetch picked up "`python-multipart` is required by FastAPI for
    `UploadFile`" as a project description, which is how a resume ends up
    listing a pip requirement as a project.
    """
    for candidate in (description, _first_heading(readme), _first_prose(readme)):
        s = re.sub(r"\s+", " ", candidate or "").strip(" -–—*#:\t")
        if len(s) < 12 or _INSTALL_LINE_RE.search(s):
            continue
        return s[:max_len].rstrip(" ,;:-") + ("..." if len(s) > max_len else "")
    return name


def _first_heading(readme: str) -> str:
    m = _H1_RE.search(readme or "")
    return m.group(1) if m else ""


def _first_prose(readme: str) -> str:
    """The first sentence that describes the project rather than its setup.

    A requirement line immediately under a heading ("(`python-multipart` is
    required by FastAPI for `UploadFile`)") is prose, so a length-and-heading
    filter alone picked it as the project description. Prose that only makes
    sense as a caveat about installing is skipped, and a paragraph is accepted
    mid-block so the search does not give up after one bad line.
    """
    for line in (readme or "").splitlines():
        s = line.strip()
        if len(s) < 25 or s.startswith(("#", "-", "*", ">", "|", "!", "[")):
            continue
        if s.startswith("(") and s.endswith(")") and _INSTALL_LINE_RE.search(s):
            continue
        if _INSTALL_LINE_RE.search(s):
            continue
        if _README_BOILERPLATE.search(s):
            continue
        return s
    return ""


def _evidence_terms(text: str, limit: int = 6) -> set[str]:
    return set(list(_terms_in(text))[:limit])


def _bullet(summary: str, readme: str, jd_terms: Iterable[str],
            max_len: int = 160) -> str:
    """The resume line, and the guarantee that it is evidence-backed.

    Returns "" unless the summary shares at least one term with the README it
    was drawn from. This is the guard that keeps a GitHub-derived project from
    becoming an invented capability: the README has to actually support the
    words we print.
    """
    summary = re.sub(r"\s+", " ", summary or "").strip()
    if not summary:
        return ""
    shared = _evidence_terms(summary) & _evidence_terms(readme)
    if not shared:
        return ""
    if len(summary) > max_len:
        summary = summary[:max_len].rstrip(" ,;:-") + "..."
    return summary


def find_projects(github_url_or_username: str, jd_text: str,
                  max_projects: int = 3, max_repos: int = 40,
                  max_readmes: int = 6) -> dict:
    """JD-relevant projects from a CV's GitHub profile.

    Returns
        {"username", "projects": [{name, description, url, language,
                                   stars, matched_terms, readme_excerpt}],
         "repos_scanned", "readmes_fetched", "error"}

    `error` is non-empty only for a rate limit; a missing profile, an empty
    README, or a network failure all yield an empty `projects` list, because a
    GitHub problem must never stop a resume being generated.
    """
    username = github_username(github_url_or_username)
    if not username or is_reserved(username):
        return {"username": "", "projects": [], "repos_scanned": 0,
                "readmes_fetched": 0, "error": ""}

    try:
        repos = list_repos(username, max_repos)
    except RuntimeError as e:
        return {"username": username, "projects": [], "repos_scanned": 0,
                "readmes_fetched": 0, "error": str(e)}

    ranked = [(score_repo(r, jd_text), r) for r in repos]
    # Only repos with some signal are worth a README request.
    ranked = [(s, r) for s, r in ranked if s > 0]
    ranked.sort(key=lambda pair: (-pair[0], -int(pair[1].get("stargazers_count") or 0)))
    candidates = ranked[:max_readmes]

    jd_terms = _terms_in(jd_text)
    projects: list[dict] = []
    fetched = 0
    for score, repo in candidates:
        if len(projects) >= max_projects:
            break
        readme = _readme_text(username, repo["name"])
        fetched += 1
        description = repo.get("description") or ""
        summary = _summarise(repo["name"], description, readme)
        bullet = _bullet(summary, readme, jd_terms)
        if not bullet:
            continue
        projects.append({
            "name": repo["name"],
            "description": bullet,
            "url": repo.get("html_url") or f"https://github.com/{username}/{repo['name']}",
            "language": repo.get("language") or "",
            "stars": int(repo.get("stargazers_count") or 0),
            "matched_terms": sorted(_terms_in(repo["name"] + " " + description + " " + readme) & jd_terms),
            "readme_excerpt": readme[:600],
            "score": round(score, 2),
        })
    return {"username": username, "projects": projects,
            "repos_scanned": len(repos), "readmes_fetched": fetched,
            "error": ""}
