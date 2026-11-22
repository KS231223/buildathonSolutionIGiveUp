"""
Decide from a unified diff alone whether a commit cannot change any test result.

    is_skippable(patch_text) -> reason string, or None if the commit must be tested

Validate against ground truth (any flagged commit with failing tests = unsafe rule):

    python skip_rules.py sample/commits sample/answers.json

ASSUMPTION: each entry in the commits dir is a file or directory holding a
unified diff. Adjust read_patch() if your layout differs.
"""

import json
import re
import sys
from pathlib import Path

HUNK_RE = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")

# Never affect a test run. Everything else (pyproject.toml, setup.py,
# sqlglotrs/, conftest.py, requirements, ...) is treated as relevant.
NON_CODE = re.compile(
    r"^(docs?/|\.github/|\.gitignore$|\.pre-commit|LICENSE|CHANGELOG|README|CONTRIBUTING|posts/|.*\.(md|rst|png|jpg|jpeg|svg|gif)$)",
    re.IGNORECASE,
)


def parse_files(text):
    """-> {path: [changed lines without the +/- marker]} (context lines ignored)."""
    files = {}
    path = None
    old_left = new_left = 0
    for row in text.splitlines():
        if old_left <= 0 and new_left <= 0:
            if row.startswith("diff --git "):
                m = re.match(r"diff --git a/(.*) b/(.*)$", row)
                path = m.group(2) if m else None
                if path:
                    files.setdefault(path, [])
                continue
            m = HUNK_RE.match(row)
            if m:
                old_left = 1 if m.group(1) is None else int(m.group(1))
                new_left = 1 if m.group(2) is None else int(m.group(2))
            continue
        tag = row[:1]
        if tag == " ":
            old_left -= 1
            new_left -= 1
        elif tag == "-":
            old_left -= 1
            files[path].append(row[1:])
        elif tag == "+":
            new_left -= 1
            files[path].append(row[1:])
    return files


def only_blank_or_comment(lines):
    return all(not l.strip() or l.lstrip().startswith("#") for l in lines)


def is_skippable(text):
    files = parse_files(text)
    if not files:
        return None  # nothing recognisable: never treat as skippable

    reasons = set()
    for path, lines in files.items():
        if NON_CODE.match(path):
            reasons.add("non-code files")
        elif path.endswith(".py") and only_blank_or_comment(lines):
            reasons.add("comment/blank-only python change")
        else:
            return None  # something that can change behaviour
    return " + ".join(sorted(reasons))


# ------------------------------------------------------------------ checking

def read_patch(entry):
    if entry.is_file():
        return entry.read_text(encoding="utf-8", errors="replace")
    parts = [p for p in sorted(entry.rglob("*")) if p.is_file()
             and p.suffix in (".diff", ".patch", "")]
    return "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in parts)


def main(commits_dir, answers_file):
    answers = json.loads(Path(answers_file).read_text())["commits"]
    flagged = unsafe = empty_total = 0

    for entry in sorted(Path(commits_dir).iterdir()):
        cid = entry.name if entry.is_dir() else entry.stem
        truth = answers.get(cid)
        if truth is None:
            continue
        empty_total += not truth["failing"]
        reason = is_skippable(read_patch(entry))
        if reason:
            flagged += 1
            if truth["failing"]:
                unsafe += 1
                print(f"UNSAFE  {cid}: flagged '{reason}' but {len(truth['failing'])} tests fail")

    total = len(answers)
    print(f"{total} commits, {empty_total} have no failing tests (free to skip)")
    print(f"rules flagged {flagged}; unsafe {unsafe}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
