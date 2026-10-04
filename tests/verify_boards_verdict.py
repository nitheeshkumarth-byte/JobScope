"""Replay the dashboard's success/failure decision against real tool output.

The bug being guarded: index.html decided whether the boards step worked by
grepping the tool's prose for the word "blocked". One board that answered
perfectly and simply had no matching posting was reported with that word, so a
250-listing hunt displayed as "all boards blocked".

This extracts the real boardsVerdict() out of static/index.html (rather than a
copy, so the check cannot drift) and runs it under node against the cases that
matter.
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
html = (ROOT / "static" / "index.html").read_text(encoding="utf-8")

m = re.search(r"function boardsVerdict\(output\) \{.*?\n\}", html, re.S)
if not m:
    sys.exit("boardsVerdict() not found in static/index.html")

REAL_OK = (
    "Job search across boards (remote; outside India):\n"
    "[Indeed] 8\n[LinkedIn] 30\n"
    "[WeWorkRemotely] 0 (nothing matched this query)\n"
    "[Remotive] 9\n[Arbeitnow] 160\n"
    "[skipped] Naukri, Glassdoor, Foundit\n"
    "[TOTAL] 20 listing(s)\n\n###JOBS_JSON###\n"
    + json.dumps({"sources": {"Indeed": 8, "LinkedIn": 30,
                              "WeWorkRemotely": "blocked", "Remotive": 9},
                  "blocked": [], "no_matches": ["WeWorkRemotely"],
                  "boards_worked": True, "total": 20})
)

ALL_WALLED = (
    "Job search across boards:\n"
    "[Indeed] blocked (HTTP 403)\n[LinkedIn] blocked (HTTP 403)\n"
    "[TOTAL] 0 listing(s)\n\n###JOBS_JSON###\n"
    + json.dumps({"sources": {"Indeed": "blocked", "LinkedIn": "blocked"},
                  "blocked": ["Indeed", "LinkedIn"], "no_matches": [],
                  "boards_worked": False, "total": 0})
)

ONE_QUIET_BOARD = (
    "[TOTAL] 0 listing(s)\n\n###JOBS_JSON###\n"
    + json.dumps({"sources": {"Remotive": 0}, "blocked": [],
                  "no_matches": [], "boards_worked": False, "total": 0})
)

# Every board answered, none had a match. The "sources" map used to label these
# "blocked", which pushed this case down the failure branch.
ALL_QUIET = (
    "[TOTAL] 0 listing(s)\n\n###JOBS_JSON###\n"
    + json.dumps({"sources": {"Indeed": 0, "LinkedIn": 0, "Remotive": 0},
                  "blocked": [], "no_matches": ["Indeed", "LinkedIn", "Remotive"],
                  "boards_worked": True, "total": 0})
)

# No payload at all: the pre-Scrapling shape, or a truncated response.
LEGACY_OK = ("Job search across boards:\n[Indeed] 12\n[Remotive] 9\n"
             "[TOTAL] 20 listing(s)")
LEGACY_WALLED = ("Job search across boards:\n[Indeed] blocked (HTTP 403)\n"
                 "[TOTAL] 0 listing(s)")

CASES = [
    ("real hunt, one quiet board", REAL_OK, True),
    ("every board walled", ALL_WALLED, False),
    ("reachable, nothing matched", ONE_QUIET_BOARD, True),
    ("all boards quiet, none matched", ALL_QUIET, True),
    ("no payload, boards worked", LEGACY_OK, True),
    ("no payload, all walled", LEGACY_WALLED, False),
]

harness = [m.group(0), "const CASES = " + json.dumps(
    [[n, o, want] for n, o, want in CASES]) + ";",
    r"""
let bad = 0;
for (const [name, output, wantOk] of CASES) {
  const v = boardsVerdict(output);
  const pass = v.ok === wantOk;
  if (!pass) bad++;
  console.log((pass ? "PASS" : "FAIL") + "  " + name +
              "  ->  ok=" + v.ok + "  \"" + v.text + "\"");
}
process.exit(bad ? 1 : 0);
"""]

with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                 encoding="utf-8") as fh:
    fh.write("\n".join(harness))
    path = fh.name

r = subprocess.run(["node", path], capture_output=True, text=True)
print(r.stdout.rstrip())
if r.stderr.strip():
    print("stderr:", r.stderr.strip())
Path(path).unlink(missing_ok=True)
sys.exit(r.returncode)