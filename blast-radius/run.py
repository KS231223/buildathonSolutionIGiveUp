"""
Entry point: for every commit in /data/commits print {"commit": id, "workers": [[...]]}.

Pipeline per commit
    1. read the patch
    2. skip_rules.is_skippable(patch)       docs / comments only        -> [] (cost 0)
    3. Selector.select(parse(patch))        line-level coverage lookup  -> test numbers
    4. split the numbers over 1..8 workers (cost = busiest worker + 2s per worker)

When a commit is UNCERTAIN (unreadable patch, selector error, file with no index,
non-python change, nothing parsable):
    UNCERTAIN=all   (default) run every test               -> safe, slow
    UNCERTAIN=cut             run nothing for that commit  -> fast, risks 300s per miss

Env vars
    WIDEN=0..4          widening ladder, see select_tests.WIDEN_LEVELS (default 2)
                        0 exact statements ... 4 wide. Pick with tune.py.
    IMPORT_TIME=module|touching|walk|all   policy for changes outside any function (default module)
    EMPTY_FALLBACK=none|module|touching    a change that finds no tests tests the module (default none)
    Rule "function edits -> function tests, everything else -> the module":
        WIDEN=1 IMPORT_TIME=touching EMPTY_FALLBACK=touching
    WORKERS=auto|1..8   (default auto)
    UNMAPPED=1|0        run the tests that have no coverage data on every code change (default 1)
    VERBOSE=1           per-commit reasons on stderr
    COMMITS_DIR / TESTS_FILE   override /data/commits and /data/tests.json (local runs)
"""

import heapq
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from select_tests import WIDEN_LEVELS, Selector, parse_unified_diff  # noqa: E402
from skip_rules import is_skippable  # noqa: E402

COMMITS_DIR = Path(os.environ.get("COMMITS_DIR", "/data/commits"))
TESTS_FILE = Path(os.environ.get("TESTS_FILE", "/data/tests.json"))
MAX_WORKERS = 8

UNCERTAIN = os.environ.get("UNCERTAIN", "all")
IMPORT_TIME = os.environ.get("IMPORT_TIME", "module")
WIDEN = int(os.environ.get("WIDEN", "2"))
UNMAPPED = os.environ.get("UNMAPPED", "1") != "0"
EMPTY_FALLBACK = os.environ.get("EMPTY_FALLBACK", "none")
if EMPTY_FALLBACK == "none":
    EMPTY_FALLBACK = None
VERBOSE = bool(os.environ.get("VERBOSE"))
WALK_ALL = os.environ.get("WALK_ALL", "0") == "1"

# ------------------------------------------------------------------ input

def commit_ids():
    # Assumption: one entry per commit. Directories are used as-is; files
    # lose their extension (abcd.diff -> abcd). Check against `ls sample/commits`.
    return [(e.name if e.is_dir() else e.stem, e) for e in sorted(COMMITS_DIR.iterdir())]


def read_patch(entry):
    """All text under the entry that looks like a unified diff, or None."""
    files = [entry] if entry.is_file() else sorted(p for p in entry.rglob("*") if p.is_file())
    chunks = []
    for p in files:
        try:
            text = p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if "diff --git" in text or "\n@@ " in text or text.startswith("@@ "):
            chunks.append(text)
    return "\n".join(chunks) if chunks else None


# ------------------------------------------------------------------ workers

def split(tests, k):
    """Longest-processing-time-first balancing -> list of k lists of test numbers."""
    bins = [(0.0, i) for i in range(k)]
    heapq.heapify(bins)
    workers = [[] for _ in range(k)]
    for number, seconds in sorted(tests, key=lambda t: -t[1]):
        load, i = heapq.heappop(bins)
        workers[i].append(number)
        heapq.heappush(bins, (load + seconds, i))
    return workers


def cost(tests, workers):
    """Estimated parallel execution time plus 2s per additional worker."""
    seconds = dict(tests)

    active_workers = [w for w in workers if w]
    if not active_workers:
        return 0.0

    busiest = max(
        sum(seconds[n] for n in worker)
        for worker in active_workers
    )

    worker_overhead = 2.0 * (len(active_workers) - 1)

    return busiest + worker_overhead

def plan(tests):
    if not tests:
        return []
    mode = os.environ.get("WORKERS", "auto")
    if mode == "auto":
        options = [split(tests, k) for k in range(1, min(MAX_WORKERS, len(tests)) + 1)]
        best = min(options, key=lambda w: cost(tests, w))
    else:
        best = split(tests, max(1, min(int(mode), MAX_WORKERS, len(tests))))
    return [w for w in best if w]


# ------------------------------------------------------------------ decision

def decide(entry, selector, patch=None):
    """-> (list of test numbers or None if uncertain, reason)"""
    patch = patch if patch is not None else read_patch(entry)
    if patch is None:
        return None, "unreadable patch"

    reason = is_skippable(patch)
    if reason:
        return [], f"skip: {reason}"

    try:
        result = selector.select(parse_unified_diff(patch))
    except Exception as exc:  # any surprise is "uncertain", never a crash
        return None, f"selector error: {type(exc).__name__}: {exc}"

    if result["uncertain"]:
        return None, "uncertain: " + "; ".join(result["uncertain"][:3])
    kinds = sorted({v["kind"] for v in result["selected"].values()})
    return selector.numbers(result["selected"]), f"selected via {kinds}"


def load_tests(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    seconds = {int(t["number"]): float(t.get("seconds", 0.0)) for t in data["tests"]}
    tests_meta = {t["id"]: {"number": int(t["number"]), "file": t.get("file")} for t in data["tests"]}
    return seconds, tests_meta


def main():
    seconds, tests_meta = load_tests(TESTS_FILE)
    all_numbers = sorted(seconds)

    selector = Selector(
    tests_meta,
    import_time=IMPORT_TIME,
    walk_all=True,
    expand_modules=True,
    empty_fallback=EMPTY_FALLBACK,
    include_unmapped=UNMAPPED,
    **WIDEN_LEVELS[WIDEN],
)

    counts = {"skipped": 0, "selected": 0, "uncertain": 0}
    out = sys.stdout

    for commit, entry in commit_ids():
        numbers, reason = decide(entry, selector)

        if numbers is None:
            counts["uncertain"] += 1
            numbers = all_numbers if UNCERTAIN == "all" else []
        elif not numbers:
            counts["skipped"] += 1
        else:
            counts["selected"] += 1

        pairs = [(n, seconds[n]) for n in numbers if n in seconds]
        out.write(json.dumps({"commit": commit, "workers": plan(pairs)}) + "\n")

        if VERBOSE:
            print(f"{commit}: {len(pairs)} tests ({reason})", file=sys.stderr)

    print(f"commits: {counts}  uncertain->{UNCERTAIN}  widen={WIDEN} import_time={IMPORT_TIME} "
          f"empty_fallback={EMPTY_FALLBACK}", file=sys.stderr)


if __name__ == "__main__":
    main()
