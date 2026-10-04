"""Test-wide safety net.

Two jobs:

1. Make the project modules importable from tests/ regardless of the cwd
   pytest is launched from.
2. Guarantee no test can touch the real accounts database or the real
   profiles.json. Both hold personal data, and a test that registers an account
   against the live DB would migrate somebody's actual CV sessions into it. Every
   test therefore starts pointed at a throwaway path; the ones that need their
   own file override these with monkeypatch, which happens after this fixture
   runs and therefore wins.
"""

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


@pytest.fixture(autouse=True)
def isolate_personal_data(tmp_path_factory, monkeypatch):
    """Redirect users.db and profiles.json to a scratch directory."""
    import auth
    import dashboard

    scratch = tmp_path_factory.mktemp("personal-data")
    monkeypatch.setattr(auth, "DB_FILE", str(scratch / "users.db"))
    monkeypatch.setattr(dashboard, "PROFILES_FILE",
                        str(scratch / "profiles.json"))
    monkeypatch.setattr(dashboard.auth, "DB_FILE", str(scratch / "users.db"))
    auth.init_db()
    yield
