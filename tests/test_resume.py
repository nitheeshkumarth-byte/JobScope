"""The generated .tex contact block.

These cover the one rule that is not "fall back to the canonical data": the
GitHub link. resume_data.LINKS holds the repo owner's real profile, so
inheriting its github entry put a hardcoded account into every generated
resume - including for the people this multi-user app is actually for.
"""
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import resume_data
import resume_generator
import resume_generator as rg
from agent import AgentConfig


def _tex(**cfg_kwargs) -> str:
    cfg = AgentConfig(**cfg_kwargs)
    return rg.build_resume(cfg, {"title": "Backend Engineer",
                                 "company": "Acme", "link": "https://x/1"})[0]


def _links_line(tex: str) -> str:
    """The links row out of the rendered header."""
    for line in tex.splitlines():
        if "\\href{http" in line and "github.com" in line:
            return line
    return ""


def _contact_row(tex: str) -> str:
    """The header contact line: the one carrying the mailto/phone separators."""
    for line in tex.splitlines():
        if "mailto" in line or ("+" in line and "vert" in line):
            return line
    return ""


# --------------------------------------------------------------------------
# LaTeX escaping of text that comes from the candidate's own CV
#
# The Objective paragraph and the skill bullets were interpolated raw. Those are
# exactly the fields where the special characters live: "R&D", "100% cheaper",
# "C#" and "F_sh" are ordinary English to whoever wrote the CV. "%" comments out
# the rest of the line and "&" fails with "Misplaced alignment tab", so the
# document did not compile at all.
# --------------------------------------------------------------------------

_HAZY_CV = (
    "Jane Doe\njane@example.com\nAustin, TX\n\n"
    "Summary\n"
    "Built CI/CD for R&D teams: cost 100% lower & used C# plus F_sh tooling.\n"
    "Delivered 20% faster for 3 teams.\n\n"
    "Technical Skills\nLanguages:C++, Python, React\n\n"
    "Experience\nAcme Corp\nBackend Engineer\nShipped 100% uptime\n"
)


def test_the_objective_escapes_latex_specials_from_the_cv():
    tex = _tex(resume_text=_HAZY_CV)
    objective = tex.split("\\section*{Objective}")[1].split("\\section*{")[0]
    # Every one of these is legal English in a CV and illegal in a .tex body.
    for raw in ("R&D", "100%", "C#", "F_sh"):
        assert raw not in objective, f"{raw!r} reached the .tex unescaped"
    assert "R\\&D" in objective
    assert "100\\%" in objective
    assert "C\\#" in objective
    assert "F\\_sh" in objective


def test_an_escaped_objective_still_does_not_eat_its_own_markup():
    """The reason the Objective was left raw in the first place: running the
    appended \\textit{} markup through the escaper turned it into visible
    backslashes. The fix has to escape the summary only, so this must still
    come out as real markup."""
    tex = _tex(resume_text=_HAZY_CV)
    assert "\\textit{Targeting the Backend Engineer role at Acme.}" in tex
    assert "textbackslash" not in tex, "the escaper ate its own markup"


def test_a_cv_with_typographic_characters_in_skills_still_compiles():
    """En dashes, curly quotes and accents survive a PDF text layer intact and
    have no pdflatex mapping."""
    cv = ("Jane Doe\njane@example.com\nAustin, TX\n\n"
          "Technical Skills\nLanguages:Caf\u00e9, Node\u2013js, \u201cReact\u201d\n\n"
          "Experience\nAcme Corp\nEngineer\nShipped things\n")
    tex = _tex(resume_text=cv)
    skills = tex.split("\\section*{Technical Skills}")[1].split("end{itemize}")[0]
    assert not [ch for ch in skills if ord(ch) > 126], skills
    assert "Caf" in skills          # accent folded, not dropped mid-word
    assert "Node-js" in skills      # en dash folded to a hyphen
    assert "React" in skills        # curly quotes folded away entirely


def test_the_whole_document_is_plain_ascii_even_for_a_hazy_cv():
    """The existing ASCII test used a CV with no summary and ASCII-only skills,
    which is why it passed while both gaps were open."""
    tex = _tex(resume_text=_HAZY_CV)
    offenders = sorted({ch for ch in tex if ord(ch) > 126})
    assert not offenders, offenders


def test_a_link_display_string_is_escaped_too():
    """The href argument was escaped but the display text was not, and a query
    string keeps its & and # right across into the visible label."""
    tex = _tex(resume_text="Jane Doe\njane@example.com\nAustin, TX\n"
                           "https://github.com/a_b?x=1&y=2#z\nBuilt APIs.\n")
    line = _links_line(tex)
    # The label sits between the closing braces of \href{...}{label}.
    label = line.rsplit("}{", 1)[-1].rstrip("}")
    assert "&" not in label, label
    assert "#" not in label, label
    assert "a\\_b" in label, label


def test_a_board_description_with_specials_still_escapes():
    """The desc argument, already covered, must keep working alongside the CV."""
    cv = ("Jane Doe\njane@example.com\nAustin, TX\n\nSummary\nBackend developer.\n\n"
          "Experience\nAcme Corp\nEngineer\nShipped things\n")
    tex = rg.build_resume(AgentConfig(resume_text=cv),
                          {"title": "Data Scientist", "company": "A&B",
                           "link": "https://x/1"},
                          desc="Own 50% of the pipeline & keep costs at $10")[0]
    objective = tex.split("\\section*{Objective}")[1].split("\\section*{")[0]
    assert "A\\&B" in objective
    assert "\\$10" in objective
    # No unescaped specials: lookbehind rather than a plain substring test,
    # because "$10" occurs inside the correct "\$10".
    assert not re.search(r"(?<!\\)[&%#$]", objective), objective


# --------------------------------------------------------------------------
# Location extraction
#
# The city vocabulary was used as a yes/no test and then the WHOLE line was
# kept, so an Experience line became the location. This is not hypothetical: the
# stored CVs for this project contain "Quiddity Infotech LLC Hyderabad".
# --------------------------------------------------------------------------


def test_an_employer_name_in_the_same_line_is_not_part_of_the_location():
    assert rg._location_from_line("Quiddity Infotech LLC Hyderabad") == "Hyderabad"
    assert rg._location_from_line("J-Spiders Hyderabad") == "Hyderabad"


def test_a_qualifier_in_parentheses_is_dropped():
    assert rg._location_from_line("Corizo Bengaluru (Remote)") == "Bengaluru"


def test_a_skills_line_is_never_mistaken_for_a_location():
    """Comma-separated capitalised words are the shape of both a "City, ST"
    pair and a skills line, so the second component has to vouch for it."""
    assert rg._location_from_line("Git, GitHub") == ""
    assert rg._location_from_line("Google Gemini API, TensorFlow, Keras") == ""
    assert rg._location_from_line("Machine Learning, TensorFlow") == ""


def test_an_org_named_after_a_country_is_not_a_location():
    """"Indian Railways" contains "India" as a substring."""
    assert rg._location_from_line("Coach Management System - Indian Railways") == ""


def test_city_state_and_country_forms_are_still_read():
    assert rg._location_from_line("Austin, TX") == "Austin, TX"
    assert rg._location_from_line("Bengaluru, Karnataka") == "Bengaluru, Karnataka"
    assert rg._location_from_line("New Delhi, India") == "Delhi, India"
    # A region may sit between the city and the country.
    assert rg._location_from_line("Toulouse, Occitanie, France") == "Toulouse, Occitanie, France"


def test_an_employer_line_does_not_become_the_candidate_location():
    cv = ("Nitheesh Kumar Thadikamalla\nnitheeshkumar.th@gmail.com\n"
          "+91 7997457091\n\nExperience\nQuiddity Infotech LLC\n"
          "Quiddity Infotech LLC Hyderabad\nBackend Engineer\n")
    contact = rg.extract_contact(cv)
    assert contact["location"] == "Hyderabad"
    assert contact["name"] == "Nitheesh Kumar Thadikamalla"
    tex = _tex(resume_text=cv)
    row = _contact_row(tex)
    # The row ends with the line separator, so match the field, not the suffix.
    assert re.search(r"Hyderabad(?!\w)", row), row
    assert "Quiddity Infotech LLC Hyderabad" not in row


def test_the_cv_location_fallback_never_returns_the_candidates_name():
    """Its regex had an optional comma, so a bare two-word name matched and the
    header printed the person's own name as their location."""
    assert rg._cv_location(["Jane Doe", "jane@example.com"]) == ""
    assert rg._cv_location(["Jane Doe | Hyderabad, India"]) == "Hyderabad, India"
    tex = _tex(resume_text="Jane Smith\njane.smith@example.com\n"
                           "Python developer.\n\nExperience\nAcme\nEngineer\n")
    row = _contact_row(tex)
    assert not re.search(r"Jane Smith(?!\w)", row), row


# --------------------------------------------------------------------------
# Skills are never invented
#
# A CV with no skills heading used to fall back to scanning its prose for any
# word from _POSTING_TERMS, so "I taught java classes and mentored a kubernetes
# club" became a claimed skill set on the document sent to employers.
# --------------------------------------------------------------------------


def test_a_cv_without_a_skills_section_gets_no_skills_section():
    cv = ("Jane Doe\njane@example.com\nAustin, TX\n\nSummary\nBackend engineer.\n\n"
          "Experience\nAcme Corp\nBackend Engineer\n"
          "I taught java classes and mentored a kubernetes club; used docker daily.\n")
    tex = _tex(resume_text=cv)
    assert "\\section*{Technical Skills}" not in tex
    # "docker" may legitimately appear in the Experience prose. What must not
    # happen is it being promoted into a skills bullet.
    assert not re.search(r"\\item\s+(?:docker|kubernetes|java)\b", tex)


def test_a_cv_that_does_declare_skills_still_gets_the_section():
    tex = _tex(resume_text=_HAZY_CV)
    assert "\\section*{Technical Skills}" in tex
    assert "C++" in tex or "C\\+\\+" in tex


# --------------------------------------------------------------------------
# The docstring documents the precedence the code implements
# --------------------------------------------------------------------------


def test_the_build_resume_docstring_matches_the_code():
    """The docstring used to claim "agent config > the CV's own links", which is
    the exact regression the adjacent code comment says was fixed. The tests
    above pin the code's order; this pins the prose to it."""
    doc = " ".join(rg.build_resume.__doc__.split())
    assert "explicit argument > the CV's own links > agent config" in doc
    assert "agent config > the CV's own links" not in doc


def test_a_github_link_that_was_never_supplied_is_not_invented():
    """No explicit link, nothing in the config, and no link in the CV text.
    Falling back to resume_data.LINKS would put the canonical owner's
    profile on someone else's resume."""
    tex = _tex(resume_text="Jane Doe\nPython backend developer\n"
                           "jane.doe@example.com\n+1 555 0100\nAustin, TX\n"
                           "Built APIs with Django and FastAPI.")
    assert "github.com" not in tex
    assert resume_data.LINKS["github"] not in tex
    # The rest of the contact block is untouched by that.
    assert "jane.doe@example.com" in tex
    assert "Jane Doe" in tex


def test_an_empty_links_row_does_not_leave_a_stray_header_line(monkeypatch):
    """Dropping the last link must not leave a blank line inside
    \\begin{center} with a [3pt] of spacing against nothing."""
    # The canonical LinkedIn/portfolio entries normally keep the row alive, so
    # empty it out to reach the branch.
    monkeypatch.setattr(rg.resume_data, "LINKS", {})
    tex = _tex(resume_text="Jane Doe\njane.doe@example.com\nAustin, TX\n"
                           "Built APIs with Django and FastAPI.")
    assert "github.com" not in tex
    header = tex.split(r"\begin{document}")[1].split(r"\begin{center}")[1]
    header = header.split(r"\end{center}")[0]
    assert "\n\n" not in header, f"the header has a blank line in it: {header!r}"
    assert "@LINKS@" not in tex


def test_a_github_url_that_is_only_a_github_com_word_is_not_detected():
    """The skill list says "Git" and "GitHub"; that must not be read as a
    profile link and turned into a github.com address."""
    tex = _tex(resume_text="Jane Doe\nSkills: Python, Django, Docker, Git, GitHub\n"
                           "jane.doe@example.com")
    assert "github.com" not in tex


def test_an_explicit_github_url_is_used():
    tex = _tex(github_url="https://github.com/janedoe")
    assert "github.com/janedoe" in tex


def test_a_github_url_in_the_config_is_used():
    tex = _tex(github_url="", resume_text="", **{"target_role": "Backend"})
    cfg = AgentConfig(github_url="https://github.com/from-config")
    tex2 = rg.build_resume(cfg, {"title": "Backend Engineer",
                                  "company": "Acme", "link": "https://x/1"})[0]
    assert "github.com/from-config" in tex2
    assert tex  # the no-link case still built


def test_a_github_url_detected_in_the_cv_text_is_used():
    tex = _tex(resume_text="Jane Doe\nhttps://github.com/janedoe\n"
                           "Python backend developer\njane.doe@example.com")
    assert "github.com/janedoe" in tex


def test_a_per_request_url_beats_the_one_in_the_cv():
    """The `github_url` argument is a deliberate override from the caller."""
    tex = rg.build_resume(
        AgentConfig(resume_text="Jane Doe\nhttps://github.com/fromcv"),
        {"title": "Backend Engineer", "company": "Acme", "link": "https://x/1"},
        "", "https://github.com/explicit")[0]
    assert "github.com/explicit" in tex
    assert "github.com/fromcv" not in tex


def test_the_cv_beats_the_saved_profile_field():
    """A saved github_url is a setting, not an override.

    It used to outrank the CV, which mattered because /api/resume/detect used
    to copy a link out of a *different* saved CV into this field. The uploaded
    CV is the content, so its own link wins and the setting only fills a gap.
    """
    cfg = AgentConfig(github_url="https://github.com/from-config",
                      resume_text="Jane Doe\nhttps://github.com/fromcv")
    tex, _ = rg.build_resume(cfg, {"title": "Backend Engineer",
                                   "company": "Acme", "link": "https://x/1"})
    assert "github.com/fromcv" in tex
    assert "github.com/from-config" not in tex


def test_the_saved_profile_field_still_fills_a_gap_the_cv_leaves():
    """Reordering precedence must not stop a link the user typed into Settings
    from appearing when the CV simply has none."""
    cfg = AgentConfig(github_url="https://github.com/from-config",
                      resume_text="Jane Doe\njane@example.com\nPython developer")
    tex, _ = rg.build_resume(cfg, {"title": "Backend Engineer",
                                   "company": "Acme", "link": "https://x/1"})
    assert "github.com/from-config" in tex


def test_a_scheme_less_github_in_the_cv_is_still_picked_up():
    tex = _tex(resume_text="Jane Doe\ngithub.com/janedoe\nPython developer")
    assert "https://github.com/janedoe" in tex


def test_a_mention_of_github_that_is_not_a_url_is_not_made_into_one():
    """Covered by the detect test above; kept as the plain-English statement
    of the rule."""
    assert rg._find_github("Skills: Python, Django, Git, GitHub") == ""


def test_no_canonical_links_leak_into_a_cv_driven_resume():
    """An uploaded CV is the whole content source, links included.

    This test used to assert the opposite - that LinkedIn and the portfolio
    stayed put while only GitHub lost its fallback. That was true when the CV
    only moved the header while the canonical body was still emitted, and it
    is exactly the leak that was reported: upload a CV and the finished resume
    still advertised somebody else's LinkedIn and portfolio. Removing the
    fallback for one link but not the other three was never a defensible rule.
    """
    tex = _tex(resume_text="Jane Doe\njane.doe@example.com\nAustin, TX\n"
                           "Built APIs with Django and FastAPI.")
    for url in resume_data.LINKS.values():
        assert url not in tex, f"canonical {url} leaked into a CV resume"


def test_canonical_content_is_still_used_when_no_cv_was_uploaded():
    """The no-upload path keeps working, so a brand-new account with nothing
    uploaded still gets a complete resume instead of an empty page."""
    cfg = AgentConfig(resume_text="")
    tex, _ = rg.build_resume(
        cfg, {"title": "Data Analyst", "company": "Acme"}, "")
    assert resume_data.NAME in tex
    assert resume_data.EXPERIENCE[0][0] in tex
    assert resume_data.PROJECTS[0][0] in tex
    # ...but still never invents a GitHub.
    assert "github.com" not in tex.lower()


def test_the_cv_supplies_the_body_when_one_is_uploaded():
    """The actual bug: the canonical experience/projects/education were
    emitted verbatim, so uploading a CV changed only the header."""
    cv = ("Jane Doe\njane.doe@example.com\nAustin, TX\n"
          "Experience\nBackend Engineer Jan 2024 - Present\nAcme\n"
          "- Owned the payments API\n"
          "Projects\nLedger Tool - Flask CLI\n- Wrote a CSV importer\n"
          "Education\nBSc Computer Science 2020 - 2024\n"
          "Technical Skills\nLanguages:Python, Go\nBackend / Web:Django\n")
    tex = _tex(resume_text=cv)
    assert "Backend Engineer Jan 2024 - Present" in tex
    assert "Owned the payments API" in tex
    assert "Ledger Tool" in tex
    assert "Wrote a CSV importer" in tex
    assert "BSc Computer Science" in tex
    assert "Languages:" in tex and "Python, Go" in tex
    for canonical_only in (resume_data.EXPERIENCE[0][0],
                           resume_data.PROJECTS[0][0],
                           resume_data.EDUCATION[0][0]):
        assert canonical_only not in tex, f"canonical {canonical_only!r} leaked"


def test_sections_absent_from_the_cv_are_omitted_entirely():
    """No CV section means no heading either. Falling back to resume_data for a
    missing section would make a half-parsed CV sprout the canonical
    person's internships."""
    tex = _tex(resume_text="Jane Doe\njane.doe@example.com\nAustin, TX\n"
                           "Technical Skills\nLanguages:Python\n")
    for heading in ("Experience", "Projects", "Education", "Certifications"):
        assert f"\\section*{{{heading}}}" not in tex


def test_a_cv_that_cannot_be_split_still_says_so():
    tex = _tex(resume_text="jane.doe@example.com")
    assert "No parsed content" in tex


def test_an_email_address_is_not_mistaken_for_a_portfolio_link():
    """gmail.com matches the bare-domain pattern, so without a boundary the
    tail of the CV's own email address came back as their portfolio."""
    assert rg._find_links("reach me at jane.doe@gmail.com") == {}
    # A real hosted portfolio is still found, subdomain and all.
    assert rg._find_links("jane.dev")["portfolio"] == "https://jane.dev"
    assert rg._find_links("jane.vercel.app")["portfolio"] == "https://jane.vercel.app"


def test_generated_cv_text_is_plain_ascii():
    """pdflatex on Overleaf errors on characters it cannot map, and a CV is
    full of en/em dashes."""
    tex = _tex(resume_text="Jane Doe – jane.doe@example.com\n"
                           "Experience\nEngineer — Jan 2024 – Present\n"
                           "- Shipped 100% uptime\n")
    assert all(ord(ch) < 128 for ch in tex), "non-ascii left in the .tex"


def _section(tex: str, name: str) -> str:
    """The body of one \\section*, so assertions about Experience cannot be
    confused by the skills bullets above it."""
    marker = f"\\section*{{{name}}}"
    if marker not in tex:
        return ""
    rest = tex.split(marker, 1)[1]
    nxt = re.search(r"\\section\*\{", rest)
    return rest[:nxt.start()] if nxt else rest


def test_a_wrapped_bullet_is_one_item_not_two():
    """A PDF text layer wraps bullet text onto its own line. The first
    continuation starts lowercase; the second is split on a line-break hyphen,
    which is dropped rather than kept with a space after it."""
    cv = ("Jane Doe\njane.doe@example.com\nAustin, TX\n"
          "Technical Skills\nLanguages:Python\n"
          "Experience\nEngineer Jan 2024 - Present\nAcme\n"
          "- Built a payments API; kept\non call and reviewed code\n"
          "- Shipped it container-\nized on AWS\n"
          "Second Job Jan 2022 - Jan 2024\nGlobex\n- Ran the nightly batch\n")
    exp = _section(_tex(resume_text=cv), "Experience")
    items = [ln.strip() for ln in exp.splitlines() if ln.strip().startswith("\\item")]
    assert len(items) == 3, items
    assert "kept on call and reviewed code" in items[0]
    assert "containerized" in items[1]
    assert "container- ized" not in exp
    # The next entry's heading stayed a heading rather than the bullet above it.
    assert "Second Job Jan 2022 - Jan 2024" in exp


def test_skill_groups_keep_their_labels_out_of_the_skill_list():
    """The CV prints '•Backend / Web:Django, Flask'. Splitting on commas alone
    made the label itself a 'skill', so 'Programming Languages Python' was
    searched for and printed as a bullet."""
    groups = rg._skill_groups([
        "•Programming Languages:Python, Java",
        "•Databases:MySQL, MongoDB, Redis,",
        "Git, GitHub",
    ])
    assert groups == [("Programming Languages", "Python, Java"),
                      ("Databases", "MySQL, MongoDB, Redis, Git, GitHub")]


def test_an_unlabelled_line_after_a_complete_group_is_its_own_group():
    """Only a wrapped continuation (previous value still ends in a comma)
    belongs to the group above it. Anything else is a separate item, so this
    does not silently absorb skills the CV listed on their own."""
    groups = rg._skill_groups(["•Languages:Python, Java", "Docker, Kubernetes"])
    assert groups == [("Languages", "Python, Java"), ("", "Docker, Kubernetes")]


def _skills_tex(posting: str) -> str:
    cfg = AgentConfig(resume_text="Jane Doe\nTechnical Skills\n"
                                  "Languages:Python, Go\nDatabases:PostgreSQL\n")
    return rg.build_resume(cfg, {"title": "Engineer", "company": "Acme"},
                           posting)[0]


def test_skill_order_follows_the_posting_but_nothing_is_dropped():
    assert _skills_tex("We need strong Python skills.").index("Languages:") \
        < _skills_tex("We need strong Python skills.").index("Databases:")
    assert _skills_tex("Deep PostgreSQL experience required.").index("Databases:") \
        < _skills_tex("Deep PostgreSQL experience required.").index("Languages:")
    for still_there in ("Python", "Go", "PostgreSQL"):
        assert still_there in _skills_tex("We need strong Python skills.")


def test_ranking_ignores_the_cv_itself():
    """The CV names every skill the candidate has, so feeding it into the
    ranking made each group match itself. A posting that mentions nothing
    must leave the CV's own order alone."""
    tex = _skills_tex("We need a backend engineer.")
    assert tex.index("Languages:") < tex.index("Databases:")
