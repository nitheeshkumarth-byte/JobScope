"""Tests for the shared listing filters (filters.py).

Every case here is a regression test for a bug that actually shipped:
- EXPERIENCE_FILTER only matched a literal "5+", so "5 years" / "5-8 yrs"
  (the far more common phrasings) sailed through the seniority gate.
- The seniority regex was matched against the COMPANY name, so every opening
  at "Staffwise" / "Lead Generation" / "Senior Living Partners" was deleted.
- "remote-india" was both a DROP token and a KEEP exemption, so an India-tied
  posting cancelled itself out and survived.
"""

import pytest

from filters import (BOARD_ALIASES, CITY_OPTIONS, COUNTRIES, DEFAULT_REGIONS,
                     DEFAULT_WORK_MODES,
                     REGIONS, REGION_TOKENS, WORK_MODES, WORLDWIDE,
                     boards_location, canonical_board, city_region,
                     country_region,
                     detect_board, identify_countries, identify_region,
                     india_rule_applies,
                     is_india_tied,
                     is_senior_or_experienced,
                     keep_job, known_cities, listing_work_modes, matches_country,
                     matches_region,
                     normalize_cities, normalize_countries,
                     matches_work_mode, normalize_regions,
                     normalize_work_modes, outside_india, region_label)


# --------------------------------------------------------------------------
# City catalog
#
# Cities are treated the opposite way to countries on purpose. A country is
# matched against a token table, so an unknown one is a typo and gets dropped.
# A city is handed to the boards as a search STRING, so an unrecognised place is
# still a real place — dropping it would empty an on-site search while the
# filter looked like it had worked.
# --------------------------------------------------------------------------

def test_normalize_cities_keeps_unknown_places_but_drops_duplicates():
    assert normalize_cities(["Berlin", "  ", "berlin", "Smalltown"]) == \
        ["Berlin", "Smalltown"]
    # Both spellings of the same tech hub survive: they are two different
    # searches, not a duplicate pair.
    assert normalize_cities(["Bengaluru", "Bangalore"]) == ["Bengaluru", "Bangalore"]
    assert normalize_cities(None) == []


def test_normalize_cities_preserves_order_because_the_first_city_is_the_search():
    # boards_location() reads cities[0] for on-site/hybrid searches, so
    # re-sorting here would silently redirect the search somewhere else.
    assert normalize_cities(["Pune", "Austin"]) == ["Pune", "Austin"]
    # normalize_countries DOES sort - the two must not be conflated.
    assert normalize_countries(["Japan", "Germany"]) == ["Germany", "Japan"]


def test_boards_location_ignores_blank_leading_cities():
    """cities[0] used to be read raw, so a leading empty string became the
    boards' search location."""
    assert boards_location(["onsite"], ["", "  ", "Pune"], False) == "Pune"
    assert boards_location(["onsite"], ["", "  "], False) == "Remote"


def test_city_region_and_known_cities_agree():
    for city in known_cities():
        assert city_region(city), f"{city} has no region"
    assert city_region("Nowhere-at-all") == ""
    assert city_region("  Berlin  ") == "europe"


def test_the_catalog_lists_alphabetically_and_is_deduplicated():
    names = known_cities()
    assert names == sorted(names, key=str.lower)
    assert len(set(names)) == len(names)


# --------------------------------------------------------------------------
# Experience filter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "5+ years",
    "5 years",
    "5yrs",
    "5 + years",
    "5-8 yrs",
    "5 to 8 years",
    "7-10 years",
    "10-12 years",
    "8 years experience",
    "minimum 5 years",
    "10 years",
    "10+ years",
    "12+ yrs of exp",
])
def test_experience_filter_catches_senior_requirements(text):
    assert is_senior_or_experienced("Analyst", text)


@pytest.mark.parametrize("text", [
    "0-2 years",
    "2 years",
    "1 year",
    "3-4 years",
    "4 yrs",
    "3+ years",            # under the documented 5+ bar
    "less than 5 years preferred",
    "under 5 years",
    "up to 5 years",
    "fewer than 5 years",
    "0 to 5 years",
    "2-5 years",
    "10 years ago",         # a date, not a requirement
])
def test_experience_filter_spares_entry_level(text):
    assert not is_senior_or_experienced("Data Analyst", text)


# --------------------------------------------------------------------------
# Seniority filter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Senior Data Analyst",
    "Team Lead",
    "Engineering Manager",
    "Head of Data",
    "VP Analytics",
    "Vice President, Data",
    "Principal Engineer",
    "Chief Data Officer",
    "Staff Software Engineer",
])
def test_seniority_filter_catches_senior_titles(text):
    assert is_senior_or_experienced(text, "Remote")


@pytest.mark.parametrize("text", [
    "Data Analyst",
    "Junior Analyst",
    "Graduate Intern",
    "Management Trainee",   # unbounded "manager" used to kill this one
    "Architecture Plus",    # unbounded "architect" used to kill this one
    "Sales Associate",
])
def test_seniority_filter_spares_entry_level(text):
    assert not is_senior_or_experienced(text, "Remote")


def test_seniority_filter_ignores_company_name():
    """The regression: these were company names, not titles.

    keep_job() has no company input at all, so a listing at any of these
    companies now survives.
    """
    for company in ("Staffwise", "Lead Generation Ltd", "Senior Living Partners"):
        assert not is_senior_or_experienced("Data Analyst", "Remote"), company
        job = {"title": "Data Analyst", "company": company,
               "location": "Remote", "link": "https://example.com/j/1"}
        assert keep_job(job, remote_only=True), company


# --------------------------------------------------------------------------
# India / remote filter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("loc", [
    "Bangalore", "Mumbai, India", "Hyderabad", "Remote-India",
    "Work From Home India",
])
def test_india_tied_locations_are_dropped(loc):
    assert is_india_tied(loc)
    assert not outside_india(loc)


@pytest.mark.parametrize("loc", [
    "Remote (Global)", "Anywhere in the World", "Berlin, Germany",
    "Work From Home (India)",   # remote wherever it is advertised from
    "WFH",
])
def test_genuinely_remote_locations_are_kept(loc):
    assert outside_india(loc)


def test_remote_india_is_not_a_work_from_home_exemption():
    """'remote-india' used to be listed as BOTH a drop token and a wfh
    exemption, so the two cancelled and the posting survived."""
    assert not outside_india("Remote-India")
    assert not outside_india("Remote India")


# --------------------------------------------------------------------------
# keep_job
# --------------------------------------------------------------------------

def _job(**over):
    base = {"title": "Data Analyst", "company": "Acme",
            "location": "Remote (Global)", "link": "https://example.com/j/1"}
    base.update(over)
    return base


def test_keep_job_accepts_a_good_listing():
    assert keep_job(_job(), remote_only=True)


@pytest.mark.parametrize("over", [
    {"title": "Senior Data Analyst"},
    {"location": "Bangalore"},
    {"link": ""},
    {"link": "javascript:alert(1)"},
    {"link": "ftp://example.com/job"},
])
def test_keep_job_rejects_bad_listings(over):
    assert not keep_job(_job(**over), remote_only=True)


def test_keep_job_respects_remote_only_off():
    assert keep_job(_job(location="Bangalore"), remote_only=False)


def test_keep_job_rejects_non_dict():
    assert not keep_job(None, remote_only=True)
    assert not keep_job("not a job", remote_only=True)


# --------------------------------------------------------------------------
# Board alias detection
# --------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("only linkedin please", "LinkedIn"),
    ("just search indeed", "Indeed"),
    ("search naukri", "Naukri"),
    ("look on weworkremotely", "WeWorkRemotely"),
    ("wwr roles", "WeWorkRemotely"),
    ("check remotive and arbeitsnow", "Remotive"),
    ("find me jobs", None),
])
def test_detect_board(text, expected):
    assert detect_board(text) == expected


def test_detect_board_handles_empty_input():
    assert detect_board("") is None
    assert detect_board(None) is None


@pytest.mark.parametrize("text,expected", [
    # The board named FIRST in the sentence wins, even when another board is
    # declared earlier in BOARD_ALIASES. Iterating the alias table made the
    # answer depend on dict order, so "naukri and linkedin" resolved to
    # LinkedIn - the opposite of what the user typed.
    ("naukri and linkedin please", "Naukri"),
    ("linkedin and naukri please", "LinkedIn"),
    ("indeed, then glassdoor", "Indeed"),
    ("glassdoor then indeed", "Glassdoor"),
    ("internshala before naukri", "Internshala"),
    # A longer alias starting at the same offset beats a shorter one, so
    # "we work remotely" is not truncated to a bare "wework".
    ("we work remotely please", "WeWorkRemotely"),
    ("search naukri for remote python roles", "Naukri"),
])
def test_detect_board_follows_word_order_not_alias_table_order(text, expected):
    assert detect_board(text) == expected


def test_detect_board_order_does_not_depend_on_the_alias_table():
    """Reversing BOARD_ALIASES must not change which board is detected."""
    from filters import BOARD_ALIASES

    text = "naukri and linkedin please"
    baseline = detect_board(text)
    original = dict(BOARD_ALIASES)
    try:
        BOARD_ALIASES.clear()
        BOARD_ALIASES.update(reversed(list(original.items())))
        assert detect_board(text) == baseline
    finally:
        BOARD_ALIASES.clear()
        BOARD_ALIASES.update(original)


def test_canonical_board_respects_the_known_set():
    known = ("Indeed", "LinkedIn")
    assert canonical_board("just indeed", known) == "Indeed"
    assert canonical_board("INDEED", known) == "Indeed"
    # Naukri is a real alias but not in this scraper's board set -> None, so
    # the caller degrades to a normal multi-board run instead of KeyError.
    assert canonical_board("naukri", known) is None
    assert canonical_board("typo-board", known) is None
    assert canonical_board("", known) is None
    assert canonical_board(None, known) is None


def test_board_aliases_are_all_distinct():
    assert len(set(BOARD_ALIASES.values())) == len(BOARD_ALIASES)


# --------------------------------------------------------------------------
# Work type: remote / WFH / hybrid / on-site
# --------------------------------------------------------------------------

def _job(**kw):
    base = {"title": "Data Analyst Intern", "location": "Remote, US",
            "link": "https://example.com/j/1"}
    base.update(kw)
    return base


@pytest.mark.parametrize("location,expected", [
    # 'remote' and 'wfh' are one family: "Work From Home" IS working remotely.
    # Separating them made matches_work_mode compute an empty intersection and
    # silently delete every WFH listing from a remote search - and
    # DEFAULT_WORK_MODES is ["remote"], so that was the default configuration.
    ("Remote, US", {"remote", "wfh"}),
    ("Remote", {"remote", "wfh"}),
    ("Anywhere", {"remote", "wfh"}),
    ("Worldwide", {"remote", "wfh"}),
    ("Virtual - Europe", {"remote", "wfh"}),
    ("Work From Home (India)", {"remote", "wfh"}),
    ("Hybrid - London", {"hybrid"}),
    # A bare city states WHERE, not HOW you work: a remote listing is tagged
    # "Paris, France". Reading it as on-site deleted every remote role outside
    # the user's own country and made Europe+Remote return nothing.
    ("Bangalore, India", set()),
    ("London, UK", set()),
    ("Paris, ile-de-France, France", set()),
    ("", set()),
    (None, set()),
])
def test_listing_work_modes(location, expected):
    assert listing_work_modes(location) == expected


def test_hybrid_is_not_also_remote_or_onsite():
    """Otherwise selecting both 'remote' and 'hybrid' would double every
    hybrid listing, and 'hybrid' would be indistinguishable from 'remote'."""
    assert listing_work_modes("Hybrid - London") == {"hybrid"}


def test_a_bare_city_is_never_read_as_onsite():
    """The regression that emptied a Europe + Remote search entirely: the
    boards return 'Paris, France' for a fully remote role."""
    assert matches_work_mode("Paris, France", ["remote"]) is True


@pytest.mark.parametrize("location", [
    "Work From Home (India)", "Remote", "Remote, US", "Anywhere",
    "Work From Home", "Virtual - Europe",
])
def test_a_remote_search_keeps_work_from_home_listings(location):
    """The two labels are the SAME arrangement to a search for remote work.

    They were disjoint tokens, so a remote-only search computed
    stated & wanted == set() and dropped every WFH posting - while boards label
    the same role either way depending on who wrote the listing.
    """
    assert matches_work_mode(location, ["remote"]) is True
    assert matches_work_mode(location, ["wfh"]) is True


def test_hybrid_is_still_excluded_from_a_remote_only_search():
    """Widening remote/wfh into one family must not make hybrid match too."""
    assert matches_work_mode("Hybrid - London", ["remote"]) is False
    assert matches_work_mode("Bengaluru, Karnataka, India", ["remote"]) is True
    # and it is likewise not claimed as on-site
    assert matches_work_mode("Paris, France", ["onsite"]) is True


@pytest.mark.parametrize("location,want,keep", [
    ("Remote, US", ["remote"], True),
    ("Remote, US", ["onsite"], False),
    ("Bangalore, India", ["onsite"], True),
    ("Work From Home (India)", ["wfh"], True),
    # Was False. WFH and remote are one arrangement, so a remote-only search
    # must keep a Work-From-Home posting; boards label the same role either way
    # and the old disjoint tokens deleted half of every remote hunt.
    ("Work From Home (India)", ["remote"], True),
    ("Hybrid - London", ["hybrid"], True),
    ("Hybrid - London", ["remote"], False),
    ("Hybrid - London", ["onsite"], False),
    # multi-select is an OR, not an AND
    ("Hybrid - London", ["remote", "hybrid"], True),
    ("Remote, US", ["remote", "onsite"], True),
    # no selection = no filtering (backward compatible)
    ("Anything At All", [], True),
    # a location that states no arrangement is kept for any selection
    ("Paris, France", ["onsite"], True),
    ("Paris, France", ["remote"], True),
    ("", ["onsite"], True),
])
def test_matches_work_mode(location, want, keep):
    assert matches_work_mode(location, want) is keep


def test_normalize_work_modes_orders_dedupes_and_drops_junk():
    assert normalize_work_modes(["onsite", "REMOTE", "remote", "nonsense", " wfh "]) == \
        ["remote", "wfh", "onsite"]
    assert normalize_work_modes([]) == []
    assert normalize_work_modes(None) == []


def test_keep_job_accepts_work_modes_and_stays_backward_compatible():
    onsite = _job(location="Bangalore, India")
    # Old call signature: no work_modes -> unchanged behaviour, India dropped.
    assert keep_job(onsite, remote_only=True) is False
    assert keep_job(_job(location="Remote, US"), remote_only=True) is True
    # New argument lets an on-site hunt through the same listing.
    assert keep_job(onsite, remote_only=True, work_modes=["onsite"]) is True
    assert keep_job(_job(location="Remote, US"), work_modes=["onsite"]) is False
    # work_modes=[] and work_modes=None both mean "don't filter".
    assert keep_job(_job(location="Bangalore, India"), remote_only=False,
                    work_modes=[]) is True


# --------------------------------------------------------------------------
# Region
# --------------------------------------------------------------------------

@pytest.mark.parametrize("location,want,keep", [
    ("London, UK", ["uk-ireland"], True),
    ("London, UK", ["europe"], True),         # UK is also counted as Europe
    ("London, UK", ["asia-pacific"], False),
    ("San Francisco, CA", ["north-america"], True),
    ("San Francisco, CA", ["europe"], False),
    ("Bengaluru, India", ["asia-pacific"], True),
    ("Bengaluru, India", ["north-america"], False),
    ("Berlin, Germany", ["europe"], True),
    ("São Paulo, Brazil", ["latam"], True),
    ("São Paulo, Brazil", ["north-america"], False),
    ("Dubai, UAE", ["africa-middle-east"], True),
    ("Toronto, Canada", ["north-america"], True),
    # worldwide / nothing selected matches everything (pre-feature behaviour)
    ("Bengaluru, India", ["worldwide"], True),
    ("Bengaluru, India", [], True),
    # multi-select is an OR
    ("Berlin, Germany", ["north-america", "europe"], True),
    # A blank location is DROPPED for a specific region: the boards return a
    # large share of listings with no location, and keeping them would make
    # the filter look effective while every result was unattributed.
    ("", ["europe"], False),
    ("   ", ["europe"], False),
    ("", ["worldwide"], True),
    # worldwide still keeps them, because no attribution is being claimed
    ("", ["worldwide", "europe"], True),
    ("Atlantis", ["europe"], False),
])
def test_matches_region(location, want, keep):
    assert matches_region(location, want) is keep


def test_region_tokens_do_not_match_inside_longer_words():
    """'la' for Los Angeles and 'eu' for Europe are the dangerous ones: a
    naive substring match would put "Malay" in Europe or "LA" everywhere."""
    assert matches_region("Malaysia", ["north-america"]) is False
    assert matches_region("Kuala Lumpur, Malaysia", ["europe"]) is False
    # the bare-abbreviation pass must respect word boundaries too
    assert matches_region("Trust Guzzle", ["north-america"]) is False
    assert matches_region("Brussels", ["north-america"]) is False


@pytest.mark.parametrize("location", [
    "Remote (US)", "Remote, US", "Remote - U.S.A.", "Remote (USA)",
])
def test_common_us_remote_locations_resolve_to_north_america(location):
    """"Remote (US)" is the single most common location string the boards
    return; failing to resolve it would empty a North-America search."""
    assert matches_region(location, ["north-america"]) is True


@pytest.mark.parametrize("location,region", [
    ("Remote (UK)", "uk-ireland"),
    ("Remote (EU)", "europe"),
])
def test_other_bare_abbreviations_resolve_to_their_own_region(location, region):
    assert matches_region(location, [region]) is True


def test_a_region_selection_does_not_bleed_into_others():
    for loc, region in [("Remote (US)", "uk-ireland"), ("Remote (UK)", "north-america"),
                        ("Remote (EU)", "asia-pacific")]:
        assert matches_region(loc, [region]) is False


# --------------------------------------------------------------------------
# An explicitly named country outranks a city name
# --------------------------------------------------------------------------

@pytest.mark.parametrize("location,expected", [
    # the collisions the boards actually produce
    ("Sydney Olympic Park, New South Wales, Australia", "asia-pacific"),
    ("Paris, Ontario, Canada", "north-america"),
    ("New Mexico, United States", "north-america"),
    ("Valencia, Spain", "europe"),            # "Valencia" is also a US city
    ("Bogota, Colombia", "latam"),
    # plain cases
    ("Paris, France", "europe"),
    ("London, UK", "uk-ireland"),
    ("South Wales, UK", "uk-ireland"),
    ("Bengaluru, Karnataka, India", "asia-pacific"),
    ("Remote (US)", "north-america"),
    ("Melbourne, Victoria, Australia", "asia-pacific"),
    ("Cape Town, South Africa", "africa-middle-east"),
    ("Remote, Worldwide", ""),                # no single region
    ("", ""),
])
def test_identify_region(location, expected):
    assert identify_region(location) == expected


def test_a_country_beats_a_conflicting_city_name():
    """Without country-first resolution these three were misfiled, and a
    Europe search returned an Australian and a Canadian listing."""
    assert matches_region("Paris, Ontario, Canada", ["europe"]) is False
    assert matches_region("New Mexico, United States", ["latam"]) is False
    assert matches_region("Sydney Olympic Park, New South Wales, Australia",
                          ["europe"]) is False
    # ... and each lands in its own region
    assert matches_region("Paris, Ontario, Canada", ["north-america"]) is True
    assert matches_region("Sydney Olympic Park, New South Wales, Australia",
                          ["asia-pacific"]) is True


def test_a_worldwide_listing_belongs_to_every_region():
    """"Remote, Worldwide" is the easiest role to apply to; dropping it from a
    regional search hides exactly the listings you most want."""
    for region in REGIONS:
        if region != WORLDWIDE:
            assert matches_region("Remote, Worldwide", [region]) is True
    assert matches_region("Anywhere", ["europe"]) is True
    assert matches_region("Multiple locations", ["latam"]) is True


def test_uk_places_still_satisfy_a_europe_selection():
    """The boards file UK roles under Europe, so Europe must keep them; the
    uk-ireland chip is for wanting the UK exclusively."""
    assert matches_region("London, UK", ["europe"]) is True
    assert matches_region("Dublin, Ireland", ["europe"]) is True
    # ... but Europe-only places must not leak into a UK-only search
    assert matches_region("Berlin, Germany", ["uk-ireland"]) is False


def test_normalize_regions_orders_dedupes_and_drops_junk():
    assert normalize_regions(["EUROPE", "europe", "north-america", "mars"]) == \
        ["north-america", "europe"]
    assert normalize_regions(None) == []


def test_keep_job_filters_by_region():
    eu = _job(location="Berlin, Germany")
    assert keep_job(eu, remote_only=False) is True
    assert keep_job(eu, remote_only=False, regions=["europe"]) is True
    assert keep_job(eu, remote_only=False, regions=["asia-pacific"]) is False
    assert keep_job(eu, remote_only=False, regions=["worldwide"]) is True


def test_both_filters_compose():
    """The whole point: a hybrid role in Europe, with the India rule still
    filtering out the same-shaped listing from India."""
    hybrid_berlin = _job(location="Hybrid - Berlin, Germany")
    hybrid_bangalore = _job(location="Hybrid - Bangalore, India")
    wm, rg = ["hybrid"], ["europe"]
    assert keep_job(hybrid_berlin, remote_only=True, work_modes=wm, regions=rg) is True
    assert keep_job(hybrid_bangalore, remote_only=True, work_modes=wm, regions=rg) is False


# --------------------------------------------------------------------------
# Where the boards are pointed
# --------------------------------------------------------------------------

@pytest.mark.parametrize("modes,cities,remote_only,expected", [
    # remote hunts search "Remote", exactly as before
    (["remote"], ["Bangalore"], True, "Remote"),
    (["remote", "wfh"], [], True, "Remote"),
    ([], ["Bangalore"], True, "Remote"),
    # on-site / hybrid need a real place -> first city
    (["onsite"], ["Bangalore", "Pune"], False, "Bangalore"),
    (["hybrid"], ["  London  ", ""], False, "London"),
    (["remote", "onsite"], ["Pune"], False, "Pune"),
    # ... but with no city there is nothing to search, so stay remote
    (["onsite"], [], False, "Remote"),
    # legacy remote_only alone still searches "Remote"
    (None, ["Pune"], True, "Remote"),
    (None, ["Pune"], False, "Pune"),
])
def test_boards_location(modes, cities, remote_only, expected):
    assert boards_location(modes, cities, remote_only) == expected


@pytest.mark.parametrize("remote_only,modes,applies", [
    # no chips -> the legacy flag decides, exactly as before
    (True, None, True),
    (True, [], True),
    (False, None, False),
    (False, [], False),
    # a remote hunt still excludes India-tied postings
    (True, ["remote"], True),
    (True, ["remote", "onsite"], True),
    (True, ["wfh"], True),
    # ... but asking for on-site/hybrid is a deliberate request for physical
    # roles, so the India rule must not delete the listings being asked for
    (True, ["onsite"], False),
    (True, ["hybrid"], False),
    (True, ["hybrid", "remote"], True),   # mixed -> remote side keeps the rule
    (False, ["onsite"], False),
])
def test_india_rule_applies(remote_only, modes, applies):
    assert india_rule_applies(remote_only, modes) is applies


def test_onsite_hunt_survives_the_legacy_remote_only_flag():
    """The old flag is still True by default in saved profiles, so an on-site
    search must not be emptied by it."""
    onsite = _job(location="Bangalore, India")
    hybrid = _job(location="Hybrid - Bangalore, India")
    assert keep_job(onsite, remote_only=True, work_modes=["onsite"]) is True
    assert keep_job(hybrid, remote_only=True, work_modes=["hybrid"]) is True
    # but a pure remote hunt still drops both
    assert keep_job(onsite, remote_only=True, work_modes=["remote"]) is False
    assert keep_job(hybrid, remote_only=True, work_modes=["remote"]) is False


def test_region_label_known_and_unknown():
    assert region_label("europe") == "Europe"
    assert region_label("EUREPE") == ""
    assert region_label("nonsense") == ""
    assert region_label(None) == ""


def test_default_work_modes_and_regions_are_known_keys():
    assert set(DEFAULT_WORK_MODES) <= set(WORK_MODES)
    assert set(DEFAULT_REGIONS) <= set(REGIONS)
    assert WORLDWIDE in REGIONS


def test_every_region_except_worldwide_has_tokens():
    for key in REGIONS:
        if key != WORLDWIDE:
            assert REGION_TOKENS.get(key), f"{key} has no city tokens"


# --------------------------------------------------------------------------
# Country filter
# --------------------------------------------------------------------------

@pytest.mark.parametrize("location,expected", [
    # an explicit country decides, same precedence rule as regions
    ("Paris, Ontario, Canada", ["Canada"]),
    ("Dülmen, Germany", ["Germany"]),
    ("Tokyo, Japan", ["Japan"]),
    ("London, United Kingdom", ["United Kingdom"]),
    ("Bengaluru, India", ["India"]),
    # the real collisions, guarded
    ("New Mexico, USA", ["United States"]),
    ("Mexico City, Mexico", ["Mexico"]),
    ("Northern Ireland, UK", ["United Kingdom"]),
    ("Berlin, Germany / Paris, France", ["France", "Germany"]),
    # nothing to attribute
    ("Remote, Worldwide", []),
    ("Anywhere in the World", []),
    ("", []),
    ("   ", []),
])
def test_identify_countries(location, expected):
    assert identify_countries(location) == expected


def test_normalize_countries_resolves_aliases_and_drops_junk():
    assert normalize_countries(["uk"]) == ["United Kingdom"]
    assert normalize_countries(["U.S.A.", "usa"]) == ["United States"]
    assert normalize_countries([" Germany ", "GERMANY", "germany"]) == ["Germany"]
    assert normalize_countries(["holland"]) == ["Netherlands"]
    # unknown values are dropped, never passed through to the boards
    assert normalize_countries(["atlantis", "", None, "  "]) == []
    assert normalize_countries(None) == []
    assert normalize_countries([]) == []


def test_matches_country_keeps_only_the_named_countries():
    assert matches_country("Dülmen, Germany", ["Germany"]) is True
    assert matches_country("Tokyo, Japan", ["Germany"]) is False
    # several allowed, either matches
    assert matches_country("Tokyo, Japan", ["Germany", "Japan"]) is True


def test_matches_country_is_off_when_nothing_is_selected():
    # no country chosen -> the filter has no opinion, even about a blank location
    assert matches_country("Tokyo, Japan", []) is True
    assert matches_country("Tokyo, Japan", None) is True
    assert matches_country("", []) is True


def test_a_worldwide_listing_satisfies_any_country():
    # dropping these would hide exactly the roles that are easiest to apply to
    assert matches_country("Remote, Worldwide", ["Germany"]) is True
    assert matches_country("Anywhere in the World", ["Japan"]) is True


def test_an_unlocated_listing_is_dropped_by_a_country_filter():
    # the boards return many listings with no location; letting them through
    # would make the filter look like it worked while nothing was attributed
    assert matches_country("", ["Germany"]) is False
    assert matches_country("   ", ["Germany"]) is False


def test_country_is_narrower_than_region():
    # same listing, two different questions
    listing = "Warsaw, Poland"
    assert matches_region(listing, ["europe"]) is True
    assert matches_country(listing, ["Germany"]) is False
    assert matches_country(listing, ["Poland"]) is True


def test_keep_job_filters_by_country():
    job = {"title": "Junior Data Analyst", "location": "Dülmen, Germany",
           "link": "https://example.com/j/1"}
    assert keep_job(job, remote_only=False, countries=["Germany"]) is True
    assert keep_job(job, remote_only=False, countries=["Japan"]) is False


def test_country_region_maps_a_country_to_its_region():
    assert country_region("Canada") == "north-america"
    assert country_region("germany") == "europe"
    assert country_region("uk") == "uk-ireland"
    assert country_region("atlantis") == ""


def test_every_country_maps_to_a_known_region():
    for name, (region, pattern) in COUNTRIES.items():
        assert region in REGIONS, f"{name} -> unknown region {region}"
        assert pattern, f"{name} has no alias pattern"


def test_every_country_is_reachable_from_its_own_name():
    for name in COUNTRIES:
        assert name in identify_countries(name)


@pytest.mark.parametrize("modes,cities,remote_only,countries,expected", [
    # a country is only used when there is no city — a city is more specific
    (["onsite"], ["Bangalore"], False, ["Germany"], "Bangalore"),
    (["onsite"], [], False, ["Germany"], "Germany"),
    (["onsite"], [], False, ["Germany", "Japan"], "Germany"),   # alphabetical
    # remote hunts still search "Remote" and filter afterwards
    (["remote"], [], True, ["Germany"], "Remote"),
    (["remote"], [], False, ["Germany"], "Remote"),
    # nothing to work with -> stay remote, as before
    (["onsite"], [], False, [], "Remote"),
    (None, [], True, ["Germany"], "Remote"),
])
def test_boards_location_falls_back_to_a_country(modes, cities, remote_only,
                                                 countries, expected):
    assert boards_location(modes, cities, remote_only, countries) == expected
