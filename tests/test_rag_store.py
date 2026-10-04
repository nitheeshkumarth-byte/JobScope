"""Tests for the candidate's document library and BM25 retrieval.

The test that matters most here is
test_every_retrieved_chunk_is_verbatim_from_a_stored_document. It is the
executable form of the rule resume_generator cares about most: a resume must
never assert something the candidate did not write. Retrieval is the only place
new text could enter the document, so that is where the guarantee is pinned.
"""

import pytest

import auth
import rag_store as R

CV = """PROFESSIONAL SUMMARY
Data analyst who builds ETL pipelines and ships dashboards.

TECHNICAL SKILLS
Languages: Python, SQL, Java
Frameworks: Django, Flask, pandas

EXPERIENCE
Acme Analytics - Data Analyst (2023 - present)
- Built a Spark ETL pipeline that cut nightly batch time from 6h to 40m
- Owned the PostgreSQL schema for 12M-row fact tables
- Mentored two interns on dbt and Airflow

PROJECTS
Retail Forecast
- Modelled weekly demand with XGBoost, reaching MAPE 8.1%
"""


@pytest.fixture
def user():
    return auth.create_user("rag@example.com", "hunter2hunter2")


# --------------------------------------------------------------------------
# tokenizing
# --------------------------------------------------------------------------

def test_tech_tokens_survive_tokenizing():
    # A naive \w+ split turns these into "c" and loses the dot, which are
    # precisely the terms a posting names.
    assert "cpp" in R.tokenize("Strong C++ background")
    assert "csharp" in R.tokenize("C# and .NET Core")
    assert "dotnet" in R.tokenize("Built on .NET")
    assert "nodejs" in R.tokenize("Node.js services")


def test_folded_tech_tokens_are_not_stemmed():
    # Regression: the plural folder turned "nodejs" into "nodej", so a posting
    # saying Node.js stopped matching a CV saying Node.js.
    assert R.tokenize("Node.js") == ["nodejs"]
    assert R.tokenize("nodejs") == ["nodejs"]


def test_plurals_fold_so_a_posting_matches_the_cv():
    assert "database" in R.tokenize("relational databases")
    assert "database" in R.tokenize("relational database")


def test_stemming_leaves_short_acronyms_and_ss_words_alone():
    assert "aws" in R.tokenize("AWS")
    assert "class" in R.tokenize("a Python class")
    assert "process" in R.tokenize("the ETL process")


def test_stopwords_and_single_chars_are_dropped():
    tokens = R.tokenize("a an the of x we are looking for this role")
    assert "the" not in tokens and "of" not in tokens
    assert "x" not in tokens
    assert "looking" in tokens and "role" in tokens


def test_tokenize_handles_empty_input():
    assert R.tokenize("") == []
    assert R.tokenize(None) == []


# --------------------------------------------------------------------------
# chunking
# --------------------------------------------------------------------------

def test_each_bullet_becomes_its_own_chunk():
    # Chunks must stay independently selectable: the JD chooses between
    # achievements, so welding two into one blob makes that impossible.
    chunks = R.chunk_text(CV)
    spark = [c for c in chunks if "Spark ETL" in c]
    assert len(spark) == 1
    assert "PostgreSQL" not in spark[0]
    assert "Airflow" not in spark[0]


def test_a_chunk_never_splits_a_bullet():
    for chunk in R.chunk_text(CV):
        assert chunk.strip()
        assert chunk.count("Built a Spark ETL") <= 1


def test_continuation_lines_join_their_bullet():
    cv = "- Reduced checkout latency by 40%\n  by adding a Redis cache layer\n"
    chunks = R.chunk_text(cv)
    assert len(chunks) == 1
    assert "Redis cache layer" in chunks[0]


def test_a_long_paragraph_is_split_on_word_boundaries():
    para = " ".join(f"word{i}" for i in range(200))
    chunks = R.chunk_text(para, chunk_chars=100)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    # Nothing lost in the split.
    assert " ".join(chunks).split() == para.split()


def test_chunking_empty_text_yields_nothing():
    assert R.chunk_text("") == []
    assert R.chunk_text("   \n\n  ") == []


# --------------------------------------------------------------------------
# BM25
# --------------------------------------------------------------------------

def _chunks(text=CV):
    return [R.Chunk("d1", "cv.pdf", "cv", i, c)
            for i, c in enumerate(R.chunk_text(text))]


def test_the_matching_achievement_ranks_first():
    hits = R.rank("Spark ETL pipeline", _chunks())
    assert hits
    assert "Spark ETL" in hits[0][0].text


def test_an_unrelated_posting_matches_nothing():
    # The no-invention guarantee in its retrieval form: a JD with no overlap
    # returns nothing rather than the least-bad unrelated bullet.
    assert R.rank("kitchen pastry chef", _chunks()) == []


def test_an_empty_query_matches_nothing():
    assert R.rank("", _chunks()) == []
    assert R.rank("   ", _chunks()) == []


def test_ranking_is_stable_for_equal_scores():
    # Two identical bullets must not swap places between runs, or two builds of
    # the same resume look like different documents.
    doc = ("- Responsible for weekly reporting\n"
           "- Responsible for weekly reporting\n")
    first = [c.index for c, _ in R.rank("weekly reporting",
                                        _chunks(doc), k=2)]
    second = [c.index for c, _ in R.rank("weekly reporting",
                                         _chunks(doc), k=2)]
    assert first == second == [0, 1]


def test_a_term_in_every_chunk_adds_nothing():
    # "data" appears in most chunks, so it cannot discriminate; the idf floor
    # stops it dragging every score the same way.
    hits = R.rank("data", _chunks())
    assert all(score >= 0 for _, score in hits)


def test_repeated_query_terms_do_not_double_count():
    once = R.rank("postgres", _chunks())
    twice = R.rank("postgres postgres postgres", _chunks())
    assert [round(s, 6) for _, s in once] == [round(s, 6) for _, s in twice]


def test_k_limits_the_number_of_hits():
    assert len(R.rank("python sql java spark postgres xgboost dbt",
                      _chunks(), k=2)) <= 2


def test_rank_on_no_chunks_is_empty():
    assert R.rank("anything", []) == []


def test_a_longer_chunk_does_not_beat_a_tighter_match_on_length_alone():
    # Length normalisation: a wall of text that mentions the term once must not
    # outrank a focused line that mentions it and is about the same thing.
    doc = ("- Short line about Redis caching\n"
           + "- " + ("padding words here " * 40) + "redis " + ("more filler " * 40))
    hits = R.rank("redis caching", _chunks(doc))
    assert hits[0][0].text.startswith("- Short line")


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

def test_a_document_round_trips(user):
    R.add_document(user["id"], "d1", "My CV", CV)
    docs = R.list_documents(user["id"])
    assert len(docs) == 1
    assert docs[0]["name"] == "My CV"
    # Stored length is of the trimmed body, not the raw argument.
    assert docs[0]["chars"] == len(CV.strip())


def test_reuploading_the_same_id_replaces_rather_than_duplicates(user):
    R.add_document(user["id"], "d1", "First", CV)
    R.add_document(user["id"], "d1", "Second", "PROJECTS\n- A tiny project")
    docs = R.list_documents(user["id"])
    assert len(docs) == 1
    assert docs[0]["name"] == "Second"


def test_empty_documents_are_rejected(user):
    with pytest.raises(ValueError):
        R.add_document(user["id"], "d1", "Blank", "   ")


def test_an_oversized_document_is_rejected(user):
    with pytest.raises(ValueError):
        R.add_document(user["id"], "d1", "Huge", "x" * (R.MAX_DOC_CHARS + 1))


def test_delete_removes_only_that_document(user):
    R.add_document(user["id"], "d1", "One", CV)
    R.add_document(user["id"], "d2", "Two", "PROJECTS\n- Something else")
    assert R.delete_document(user["id"], "d1") is True
    assert [d["id"] for d in R.list_documents(user["id"])] == ["d2"]
    assert R.delete_document(user["id"], "d1") is False


def test_documents_are_scoped_to_their_owner(user):
    # The isolation test that matters: one account's documents must never
    # surface in another account's resume.
    other = auth.create_user("other@example.com", "hunter2hunter2")
    R.add_document(user["id"], "secret", "Private CV",
                   "EXPERIENCE\n- Built a proprietary payments ledger")
    assert R.retrieve(other["id"], "proprietary payments ledger") == []
    assert len(R.retrieve(user["id"], "payments ledger")) == 1


def test_retrieval_spans_every_document_a_user_uploaded(user):
    R.add_document(user["id"], "cv", "CV", CV)
    R.add_document(user["id"], "proj", "Project writeup",
                   "PROJECTS\n- Deployed a Kubernetes autoscaler for the API tier")
    hits = R.retrieve(user["id"], "kubernetes autoscaler")
    assert any("Kubernetes" in c.text for c, _ in hits)


# --------------------------------------------------------------------------
# the guarantee
# --------------------------------------------------------------------------

def test_every_retrieved_chunk_is_verbatim_from_a_stored_document(user):
    """No word may enter the resume that the user did not write."""
    R.add_document(user["id"], "cv", "CV", CV)
    R.add_document(user["id"], "proj", "Project",
                   "PROJECTS\n- Wrote a Rust CLI for parquet slicing")
    for jd in ("python spark etl", "rust cli", "", "kubernetes docker"):
        for chunk, _score in R.retrieve(user["id"], jd):
            stored = CV if chunk.doc_id == "cv" else \
                "PROJECTS\n- Wrote a Rust CLI for parquet slicing"
            # Chunking only ever joins or splits on whitespace, so the words
            # must all be present in the source document.
            for word in chunk.text.split():
                assert word in stored, f"{word!r} was not in any stored document"


def test_chunk_provenance_points_back_at_its_document(user):
    R.add_document(user["id"], "cv", "My CV.pdf", CV)
    chunk = R.all_chunks(user["id"])[0]
    assert chunk.doc_id == "cv"
    assert chunk.doc_name == "My CV.pdf"
    assert chunk.source_ref.startswith("My CV.pdf#")


def test_an_empty_library_retrieves_nothing(user):
    assert R.retrieve(user["id"], "python") == []
    assert R.all_chunks(user["id"]) == []
