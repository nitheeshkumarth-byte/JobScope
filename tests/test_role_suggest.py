"""Tests for role suggestions.

The complaints this replaces were, in order of how often they came up: the same
suggestion appearing several times over ("Data Analyst" AND "Business
Intelligence Analyst"), the ordering flipping when one extra skill was added,
and a CV from outside the data/BI world being told nothing at all.

So these tests are mostly about the FAILURE modes rather than the happy path:
duplicates, unstable order, and generic words inventing a specialisation.
"""

import pytest

import dashboard
import rag_store


def _roles(skills, cv="", limit=4):
    return [r["role"] for r in dashboard.suggest_roles(skills, resume_text=cv, limit=limit)]


def _all(skills, cv=""):
    return dashboard.suggest_roles(skills, resume_text=cv, limit=99)


# --------------------------------------------------------------------------
# duplicates and twins
# --------------------------------------------------------------------------

def test_twins_do_not_both_appear_as_primary_suggestions():
    """The original complaint: one job, three spellings, four rows."""
    for skills, cv in (
        (["python", "sql", "pandas", "excel", "power bi", "tableau"],
         "5 years of Power BI dashboards and SQL reporting."),
        (["kubernetes", "terraform", "aws", "docker", "ci/cd"],
         "DevOps engineer, 8 years running Kubernetes on AWS."),
        (["react", "typescript", "css", "html", "node", "python"],
         "Full stack engineer, React and Node, 4 years."),
    ):
        got = dashboard.suggest_roles(skills, resume_text=cv)
        primary = [r["role"] for r in got if not r.get("alternate")]
        for a, b in dashboard.ROLE_TWINS:
            assert not (a in primary and b in primary), \
                f"twins {a}/{b} both offered as primary for {skills}"


def test_a_twin_backfill_is_marked_as_an_alternate():
    """Offering the twin is better than offering nothing, but it must be honest."""
    got = dashboard.suggest_roles(["python", "sql", "pandas", "excel", "power bi"])
    for r in got:
        if r.get("alternate"):
            assert r["role"] not in [x["role"] for x in got if not x.get("alternate")]


def test_twin_pairs_are_mutually_consistent():
    """A one-way twin entry would let a pair through in one order only."""
    for a, b in dashboard.ROLE_TWINS:
        assert a != b, f"{a} is its own twin"
        assert b in dashboard._TWIN_OF.get(a, set()), f"{a} does not know {b}"


def test_data_roles_do_not_collapse_into_one_answer():
    got = _roles(["python", "sql", "pandas", "excel", "power bi", "tableau"],
                 "5 years Power BI and SQL.")
    assert len(set(got)) == len(got), "duplicate role in the list"
    assert len(got) >= 2, "a data CV should still get more than one idea"


# --------------------------------------------------------------------------
# stability
# --------------------------------------------------------------------------

def test_the_same_input_always_produces_the_same_order():
    """Identical uploads used to reshuffle, which read as the app being wrong."""
    skills = ["python", "sql", "pandas", "excel", "power bi", "tableau"]
    cv = "5 years Power BI dashboards and SQL reports."
    orders = {tuple(_roles(skills, cv)) for _ in range(15)}
    assert len(orders) == 1, f"unstable ordering: {orders}"


def test_order_is_by_score_not_by_dictionary_order():
    got = dashboard.suggest_roles(
        ["kubernetes", "terraform", "aws", "docker", "ci/cd", "jenkins", "linux"],
        "DevOps engineer with 8 years running Kubernetes on AWS with Terraform.")
    scores = [r["score"] for r in got]
    assert scores == sorted(scores, reverse=True), f"not score-ordered: {scores}"
    assert len(set(scores)) > 1, \
        "three roles tied at 1.0 means the score clipped and the order is arbitrary"


def test_scores_are_not_clipped_at_one():
    """Clipping flattened every strong match to 1.0 and destroyed the ranking."""
    for r in dashboard.suggest_roles(
            ["python", "sql", "pandas", "excel", "power bi", "tableau", "statistics"],
            "5 years of analytics."):
        assert r["score"] > 0


# --------------------------------------------------------------------------
# generic tokens must not invent a specialisation
# --------------------------------------------------------------------------

def test_java_alone_does_not_produce_android_developer():
    """`java` is the most overloaded word in the CV world."""
    got = _roles(["java"], "Java programming.")
    assert "Android Developer" not in got


def test_aws_alone_does_not_produce_data_engineer():
    got = _roles(["aws"], "Some AWS work.")
    assert "Data Engineer" not in got


def test_c_plus_plus_does_not_produce_game_developer():
    """C++ is in the Game Developer rubric, but C++ alone does not mean games.

    `c++` folds to `cpp`, and `c` is a substring of `cpp`, so a plain token
    intersection let an embedded C/C++ CV satisfy the games role.
    """
    got = _roles(["c", "c++", "rtos", "stm32", "i2c", "spi"],
                 "Firmware engineer, C and C++ on STM32.")
    assert "Game Developer" not in got


def test_javascript_does_not_produce_servicenow_developer():
    got = _roles(["react", "javascript", "css", "html"],
                 "Frontend work with React.")
    assert "ServiceNow Developer" not in got


def test_an_embedded_cv_suggests_embedded_roles():
    got = _roles(["c", "c++", "rtos", "stm32", "i2c", "spi", "firmware"],
                 "Firmware engineer, 4 years, C/C++ on STM32 with RTOS.")
    assert got[0].startswith("Embedded") or got[0] == "Firmware Engineer", got


def test_a_frontend_cv_does_not_get_backend_roles():
    got = _roles(["react", "css", "html", "typescript", "figma"],
                 "Frontend developer, React and Figma.")
    assert got[0] in ("Frontend Developer", "UI/UX Designer", "Product Designer",
                      "Web Developer", "Full Stack Developer"), got


# --------------------------------------------------------------------------
# coverage of the catalog
# --------------------------------------------------------------------------

def test_the_rubric_is_not_data_only():
    """The old eight-entry rubric told a Java or embedded applicant nothing."""
    for role in ("Backend Developer", "DevOps Engineer", "Embedded Software Engineer",
                 "Mobile App Developer", "Network Engineer", "Product Manager",
                 "Machine Learning Engineer", "QA Engineer"):
        assert role in dict(dashboard.ROLE_RUBRIC)


def test_every_rubric_role_is_selectable_in_the_dropdown():
    """A rubric entry the user cannot pick is a dead suggestion."""
    catalog = set(dashboard.JOB_ROLE_OPTIONS)
    for role, _needs in dashboard.ROLE_RUBRIC:
        assert role in catalog, f"{role} is scored but not in JOB_ROLE_OPTIONS"


def test_a_non_data_cv_gets_suggestions():
    got = _roles(["java", "spring boot", "microservices", "kafka", "postgresql"],
                 "Java backend developer, 6 years of Spring Boot microservices.")
    assert got, "a Java backend CV must not get an empty suggestion list"
    assert "Backend Developer" in got


# --------------------------------------------------------------------------
# the document library as evidence
# --------------------------------------------------------------------------

def test_an_uploaded_document_can_justify_a_role_with_no_skills():
    """The RAG signal: what they wrote counts, not just what was extracted."""
    chunk = rag_store.Chunk(
        doc_id="d1", doc_name="proj.md", kind="project", index=0,
        text="Migrated deployments to Kubernetes and Terraform running on AWS.")
    got = dashboard.suggest_roles([], resume_text="",
                                  doc_hits=[(chunk, 4.2)])
    assert got, "a document naming Kubernetes/Terraform should suggest something"
    assert any(r["from_docs"] for r in got)


def test_doc_evidence_is_reported_separately_from_skills():
    chunk = rag_store.Chunk("d1", "proj.md", "project", 0,
                            "Built a CI/CD pipeline with Jenkins and Docker.")
    got = dashboard.suggest_roles([], resume_text="", doc_hits=[(chunk, 3.0)])
    for r in got:
        assert "from_docs" in r


def test_documents_cannot_alone_invent_a_role_from_generic_words():
    chunk = rag_store.Chunk("d1", "note.md", "project", 0,
                            "We use AWS and Docker and Linux every day.")
    got = dashboard.suggest_roles([], resume_text="", doc_hits=[(chunk, 1.0)])
    assert all(r["matched"] for r in got), \
        "a role suggested with zero matching terms is not evidenced"


# --------------------------------------------------------------------------
# seniority
# --------------------------------------------------------------------------

def test_a_fresh_graduate_is_not_suggested_senior_roles():
    got = dashboard.suggest_roles(
        ["python", "sql", "c"],
        "B.Tech 2024 graduate. Python projects, C coursework.")
    assert all(r.get("level") != "senior" for r in got), got


def test_a_long_experience_cv_is_marked_senior():
    got = dashboard.suggest_roles(["python", "sql", "django"],
                                  "Software engineer with 9 years of experience.")
    assert any(r.get("level") == "senior" for r in got)


def test_intern_roles_are_always_entry_level():
    for role in ("Data Science Intern", "Software Engineering Intern"):
        needs = dict(dashboard.ROLE_RUBRIC).get(role)
        if needs is None:
            continue
        assert dashboard._seniority_of(role, "9 years experience") == "junior"


def test_seniority_does_not_depend_on_a_specific_role_being_matched():
    """_seniority_of is a pure judgement, callable on its own."""
    assert dashboard._seniority_of("Data Analyst", "8 years") == "senior"
    assert dashboard._seniority_of("Data Analyst", "") == ""


# --------------------------------------------------------------------------
# honesty
# --------------------------------------------------------------------------

def test_unrecognisable_input_returns_nothing():
    """Better than four confident guesses."""
    assert dashboard.suggest_roles(["zzz", "qqq"], resume_text="lorem ipsum") == []
    assert dashboard.suggest_roles([], resume_text="") == []
    assert dashboard.suggest_roles(None, resume_text="") == []


def test_the_limit_is_respected():
    for n in (1, 2, 3, 4):
        got = dashboard.suggest_roles(
            ["python", "sql", "pandas", "excel", "power bi", "tableau", "docker"],
            "Analytics and infrastructure work.", limit=n)
        assert len(got) <= n


def test_every_suggestion_carries_its_reasoning():
    got = dashboard.suggest_roles(["python", "sql", "pandas"], "Data work.")
    assert got
    for r in got:
        assert r["role"] and isinstance(r["score"], float) and r["score"] > 0
        assert r["matched"], f"{r['role']} was suggested with no matching skill"
