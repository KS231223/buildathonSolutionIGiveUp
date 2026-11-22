"""
Select the tests affected by a change, using the most precise evidence available.

Each changed OLD-side line is classified:

  statement   line is inside a statement in a function body
              -> tests that executed THAT statement (exact, from coverage)
  function    line is in a function but not in a statement (else:, try:, def line...)
              -> tests that executed any line of that function
  insertion   new lines between old lines
              -> same block as a neighbouring statement: that statement's tests
                 inside the function, other block: the whole function's tests
  snapped     decorator / gap line outside any function -> the CLOSEST function's tests
  import-time module / class level (constants, class bodies, new methods...)
              -> coverage cannot see these; policy set by import_time
                 (module = the module's own tests, not every test that touches it)
  no-op       blank / comment-only change -> nothing

Coverage already contains every test that reached a line, so the reverse graph is
NOT walked for statement/function hits. It is only walked for import-time changes
(import_time="walk") or everywhere (walk_all=True).

Library use (what run.py does; maps are loaded once):
    sel = Selector(tests)                     # tests = {id: {"number": n, "file": f}}
    result = sel.select(parse_unified_diff(patch_text))
    result["selected"], result["uncertain"], ...

CLI:
    python select_tests.py --patch change.diff
    git diff | python select_tests.py --patch -
    python select_tests.py optimizer/simplify.py:120 dialects/athena.py:30-45
    python select_tests.py --git-diff HEAD --repo ../sqlglot
    python select_tests.py --patch change.diff --numbers      # test numbers, one per line
"""

import argparse
import json
import re
import subprocess
import sys
from collections import deque
from pathlib import Path

from scope_lookup import (
    GRAPH_DIR,
    MODULE_TESTS_FILE,
    MODULE_TOUCH_TESTS_FILE,
    STATEMENT_TESTS_FILE,
    SYMBOL_TESTS_FILE,
    UNMAPPED_TESTS_FILE,
    LineIndex,
    load_json,
    normalise_file,
)
from skip_rules import NON_CODE

TEST_DEPENDENCIES_FILE = GRAPH_DIR / "test_dependencies.json"
REVERSE_GRAPH_FILE = GRAPH_DIR / "reverse_graph.json"
TEST_MAP_FILE = GRAPH_DIR / "test_map.json"
DIALECT_TESTS_FILE = GRAPH_DIR / "dialect_to_tests.json"

FIXTURE_TESTS_FILE = GRAPH_DIR / "fixture_to_tests.json"

# Cumulative widening ladder: each step adds cheap tests to buy safety (a miss costs 300s).
#   0  exact statement-level coverage
#   1  any change in a function -> every test that ran that function
#   2  + the changed module's own tests (conventional test file)
#   3  + every test in the same test files as anything already selected
#   4  + every test touching the changed module (wide)
WIDEN_LEVELS = {
    0: {},
    1: {"granularity": "function"},
    2: {"granularity": "function", "module_own": True},
    3: {"granularity": "function", "module_own": True, "expand_test_files": True},
    4: {"granularity": "function", "module_touching": True, "expand_test_files": True},
}

HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


# ============================================================
# INPUT: changes
# ============================================================

def parse_unified_diff(text):
    """
    Parse ANY unified diff (-U0, -U3, ...) into
      {old_path: {"deleted": {old_line: text},
                  "inserts": [{"before": n, "after": n+1, "texts": [...]}]}}
    Only '-' lines are changed old lines. A block of '+' lines with no '-' lines
    is a pure insertion between old lines `before` and `after`.
    """
    files = {}
    path = None
    old_ln = 0
    old_left = new_left = 0
    block_deleted = False
    insert = None

    def entry():
        if not path:
            return None
        return files.setdefault(path, {"deleted": {}, "inserts": []})

    for row in text.splitlines():
        if old_left <= 0 and new_left <= 0:  # outside a hunk
            if row.startswith("--- "):
                p = row[4:].split("\t")[0]
                if p == "/dev/null":
                    path = None
                else:
                    path = p[2:] if p.startswith("a/") else p
                continue
            m = HUNK_RE.match(row)
            if m:
                old_left = 1 if m.group(2) is None else int(m.group(2))
                new_left = 1 if m.group(4) is None else int(m.group(4))
                start = int(m.group(1))
                old_ln = start + 1 if old_left == 0 else start
                block_deleted = False
                insert = None
            continue

        tag = row[:1]
        if tag == " ":
            old_ln += 1
            old_left -= 1
            new_left -= 1
            block_deleted = False
            insert = None
        elif tag == "-":
            e = entry()
            if e is not None:
                e["deleted"][old_ln] = row[1:]
            old_ln += 1
            old_left -= 1
            block_deleted = True
            insert = None
        elif tag == "+":
            if not block_deleted:
                e = entry()
                if e is not None:
                    if insert is None:
                        insert = {"before": old_ln - 1, "after": old_ln, "texts": []}
                        e["inserts"].append(insert)
                    insert["texts"].append(row[1:])
            new_left -= 1

    return files


def changes_from_git(ref, repo):
    """ref can be 'HEAD', 'main...HEAD', or 'parent..commit'."""
    out = subprocess.run(
        ["git", "-C", repo, "diff", "-U0", "--no-color", ref],
        capture_output=True, text=True, check=True,
    ).stdout
    return parse_unified_diff(out)


def parse_spec(spec):
    """'file:12' or 'file:12-30' -> (file, [lines])"""
    file, _, rng = spec.rpartition(":")
    if "-" in rng:
        start, end = rng.split("-", 1)
        return file, list(range(int(start), int(end) + 1))
    return file, [int(rng)]


def is_noop(text):
    return text is not None and (not text.strip() or text.lstrip().startswith("#"))


def indent_of(text):
    return len(text) - len(text.lstrip(" \t"))


# ============================================================
# GRAPH WALK (only for import-time changes unless walk_all)
# ============================================================


def walk_up(
    seeds,
    reverse_graph,
    definitions,
    max_depth,
    expand_modules,
    module_definitions=None,
):
    """
    BFS over dependents.

    When enabled, module nodes expand to definitions contained in
    that module, in addition to following reverse-graph edges.
    """
    depth = {}
    parent = {}
    queue = deque()

    module_definitions = module_definitions or {}

    for seed in seeds:
        if seed not in depth:
            depth[seed] = 0
            parent[seed] = None
            queue.append(seed)

    while queue:
        node = queue.popleft()

        is_definition = node in definitions
        is_module = node in module_definitions

        if not is_definition and not (expand_modules and is_module):
            continue

        if max_depth is not None and depth[node] >= max_depth:
            continue

        neighbours = []

        # Expand module into its definitions.
        if expand_modules and is_module:
            neighbours.extend(
                (name, "contains")
                for name in module_definitions[node]
            )

        # Follow reverse dependency edges.
        neighbours.extend(
            (edge["dependent"], edge["relationship"])
            for edge in reverse_graph.get(node, [])
        )

        for dependent, relationship in neighbours:
            if dependent in depth:
                continue

            depth[dependent] = depth[node] + 1
            parent[dependent] = (node, relationship)
            queue.append(dependent)

    return depth, parent


def chain(node, parent):
    steps = [node]
    while parent[node] is not None:
        node, rel = parent[node]
        steps.append(f"<-{rel}- {node}")
    return " ".join(steps)


# ============================================================
# SELECTOR
# ============================================================

class Selector:
    def __init__(self, tests, import_time="module", walk_all=False, max_depth=None,
                 no_ancestors=False, expand_modules=False, ignore_module_dependents=False,
                 granularity="statement", module_own=False, module_touching=False,
                 expand_test_files=False, empty_fallback=None, include_unmapped=True):
        """tests: {test_id: {"number": int, "file": str}}  (e.g. from tests.json)"""
        self.idx = LineIndex()
        self.definitions = self.idx.definitions
        self.symbol_tests = load_json(SYMBOL_TESTS_FILE)
        self.dialect_tests = (
            load_json(DIALECT_TESTS_FILE)
            if DIALECT_TESTS_FILE.exists()
            else {}
        )
        self.module_tests = load_json(MODULE_TESTS_FILE)  # module's own tests (narrow)
        self._module_touch = None                         # every touching test (wide), lazy
        self.stmt_tests = load_json(STATEMENT_TESTS_FILE) if STATEMENT_TESTS_FILE.exists() else {}
        self.test_dependencies = (
            load_json(TEST_DEPENDENCIES_FILE)
            if TEST_DEPENDENCIES_FILE.exists()
            else {}
        )
        self._reverse = None

        self.tests = tests
        self.id_to_num = {tid: m.get("number") for tid, m in tests.items()}
        self.num_to_id = {n: tid for tid, n in self.id_to_num.items() if n is not None}
        self.all_ids = list(tests)

        self.tests_by_file = {}
        for tid, m in tests.items():
            self.tests_by_file.setdefault(m.get("file"), []).append(tid)
        self._fixtures = None
        self.tests_by_name = {}  # "test_foo" -> ids (parametrisation stripped)
        for tid in tests:
            name = tid.split("::")[-1].split("[")[0]
            self.tests_by_name.setdefault(name, []).append(tid)

        # None | "module" | "touching": a changed line whose lookup finds NO tests
        # (function never ran in coverage, ...) falls back to the module's tests
        self.empty_fallback = empty_fallback
        # tests with NO coverage can never be selected by a rule, so they run on every code change
        self.include_unmapped = include_unmapped
        self._unmapped = None
        self.granularity = granularity
        self.module_own = module_own
        self.module_touching = module_touching
        self.expand_test_files = expand_test_files

        self.import_time_policy = import_time
        self.walk_all = walk_all
        self.max_depth = max_depth
        self.no_ancestors = no_ancestors
        self.expand_modules = expand_modules
        self.ignore_module_dependents = ignore_module_dependents

    @property
    def module_touch_tests(self):
        if self._module_touch is None:
            self._module_touch = load_json(MODULE_TOUCH_TESTS_FILE)
        return self._module_touch

    @property
    def unmapped_numbers(self):
        if self._unmapped is None:
            self._unmapped = load_json(UNMAPPED_TESTS_FILE) if UNMAPPED_TESTS_FILE.exists() else []
        return self._unmapped

    @property
    def fixture_tests(self):
        if self._fixtures is None:
            self._fixtures = load_json(FIXTURE_TESTS_FILE) if FIXTURE_TESTS_FILE.exists() else {}
        return self._fixtures

    @property
    def reverse_graph(self):
        if self._reverse is None:
            self._reverse = load_json(REVERSE_GRAPH_FILE) if REVERSE_GRAPH_FILE.exists() else {}
        return self._reverse

    def select(self, raw):
        """
        raw: {file: {"deleted": {line: text}, "inserts": [...]}}
        Returns {"selected": {id: {via, kind}}, "changed": [...], "warnings": [...],
                 "uncertain": [...], "depth": {...}}
        `uncertain` lists reasons the result may be incomplete (callers decide what to do).
        """
        idx, definitions = self.idx, self.definitions
        symbol_tests, module_tests, stmt_tests = self.symbol_tests, self.module_tests, self.stmt_tests

        selected = {}
        changed = []
        warnings = []
        uncertain = []
        walk_seeds = []
        direct_tests = {}

        state = {"n": 0, "hits": 0, "h": 0}

        def add_ids(ids, via, kind):
            state["hits"] += len(ids)
            for t in ids:
                selected.setdefault(t, {"via": via, "kind": kind})

        def add_numbers(nums, via, kind):
            add_ids([self.num_to_id[n] for n in nums if n in self.num_to_id], via, kind)

        def record(file, line, scope, resolution):
            if (self.empty_fallback and resolution != "no-op" and state["hits"] == state["h"]
                    and file.endswith(".py") and idx.has_file(file)):
                mod = idx.module(file)
                src = self.module_touch_tests if self.empty_fallback == "touching" else module_tests
                add_ids(src.get(mod, []), file, "fallback")
                resolution += "+fallback"
            n = len(selected)
            changed.append({"file": file, "line": line, "scope": scope,
                            "resolution": resolution, "tests_added": n - state["n"]})
            state["n"] = n
            state["h"] = state["hits"]

        def import_time(scope, kind, module, where):
            if self.import_time_policy == "all":
                add_ids(self.all_ids, where, "import-time")
            elif self.import_time_policy == "walk":
                walk_seeds.append(scope)
            elif kind == "class":
                add_ids(symbol_tests.get(scope, []), where, "import-time")
            elif self.import_time_policy == "touching":
                add_ids(self.module_touch_tests.get(module, []), where, "import-time")
            else:  # "module": the module's own tests only
                add_ids(module_tests.get(module, []), where, "import-time")

        def container(name, module):
            """Innermost enclosing class (or the module) of a definition name."""
            if name:
                for n in [name] + idx.ancestors(name):
                    if definitions[n]["kind"] == "class":
                        return n, "class"
            return module, "module"

        def target_function(rel, line, indent):
            """Innermost function enclosing `line` whose def column is left of `indent`."""
            name = idx.scope(rel, line)
            if not name:
                return None
            for n in [name] + idx.ancestors(name):
                d = definitions[n]
                if d["kind"] in ("function", "method"):
                    col = idx.func_col(rel, d["line"])
                    if col is not None and col < indent:
                        return n
            return None

        def statement_hit(rel, slot, where):
            if self.granularity == "function":  # widen: whole function instead of the statement
                fname = idx.scope(rel, idx.statements[rel]["starts"][slot])
                if fname:
                    add_ids(symbol_tests.get(fname, []), where, "statement")
                    return
            add_numbers(stmt_tests.get(rel, {}).get(str(slot), []), where, "statement")

        if not raw:
            uncertain.append("no parsable changes in patch")

        for file, entry in raw.items():
            if file.startswith("tests/"):
                if file.endswith(".py"):
                    hit = [tid for tid, m in self.tests.items() if m.get("file") == file]
                    for tid in hit:
                        direct_tests[tid] = f"test file changed: {file}"
                else:
                    key = file[file.find("tests/"):]
                    nums = self.fixture_tests.get(key)
                    if nums is not None:
                        add_numbers(nums, file, "fixture")
                        record(file, 0, file, "fixture")
                    else:
                        # naming convention: fixtures/optimizer/simplify.sql -> test_simplify
                        stem = key.rsplit("/", 1)[-1].rsplit(".", 1)[0].replace("-", "_")
                        named = self.tests_by_name.get(f"test_{stem}", [])
                        if named:
                            add_ids(named, file, "fixture")
                            record(file, 0, file, "fixture-name")
                        else:
                            msg = f"{file}: test data/fixture changed; not in fixture->test map"
                            warnings.append(msg)
                            uncertain.append(msg)
                continue

            rel = normalise_file(file)
            if not rel.endswith(".py"):
                if not NON_CODE.match(file):
                    uncertain.append(f"{file}: non-python file that can affect behaviour")
                continue
            if not idx.has_file(rel) or rel not in idx.statements:
                msg = f"no index for {file} (new file or not in baseline)"
                warnings.append(msg)
                uncertain.append(msg)
                continue
            module = idx.module(rel)
            # -- dialect-specific test mapping ------------------------------------
            # -- dialect-specific test mapping ------------------------------------
            if rel.startswith(("dialects/", "parsers/")):
                dialect_name = Path(rel).stem.lower()

                if dialect_name in self.dialect_tests:
                    dialect_ids = self.dialect_tests[dialect_name]

                    add_ids(
                        dialect_ids,
                        f"dialect:{dialect_name}",
                        "dialect",
                    )

            # -- deleted / modified old lines
            for line, text in sorted(entry["deleted"].items()):
                where = f"{rel}:{line}"
                slot = idx.statement_slot(rel, line)
                # decorators / gap lines belong to the closest function, not the module
                snap = text is None or text.lstrip().startswith("@")
                info = idx.resolve(rel, line, snap=snap)

                if slot is not None and idx.slot_in_function(rel, slot):
                    statement_hit(rel, slot, where)
                    record(rel, line, info["scope"], "statement")
                    if self.walk_all:
                        walk_seeds.append(info["scope"])
                elif slot is None and is_noop(text):
                    record(rel, line, info["scope"], "no-op")
                elif info["kind"] in ("function", "method"):
                    add_ids(symbol_tests.get(info["scope"], []), where, "function")
                    record(rel, line, info["scope"], "snapped" if info["snapped"] else "function")
                    if self.walk_all:
                        walk_seeds.append(info["scope"])
                else:
                    import_time(info["scope"], info["kind"], module, where)
                    record(rel, line, info["scope"], "import-time")

            # -- pure insertions
            for ins in entry["inserts"]:
                before, after = ins["before"], ins["after"]
                where = f"{rel}:{before}+"
                code = [t for t in ins["texts"] if t.strip() and not t.lstrip().startswith("#")]
                sb = idx.statement_slot(rel, before)
                sa = idx.statement_slot(rel, after)

                if sb is not None and sb == sa:  # inside one multi-line statement
                    if idx.slot_in_function(rel, sb):
                        statement_hit(rel, sb, where)
                        record(rel, before, idx.scope(rel, before) or module, "statement")
                    else:
                        info = idx.resolve(rel, before)
                        import_time(info["scope"], info["kind"], module, where)
                        record(rel, before, info["scope"], "import-time")
                    continue

                if not code:
                    record(rel, before, idx.scope(rel, before) or module, "no-op")
                    continue

                indent = indent_of(code[0])
                func = target_function(rel, before, indent) or target_function(rel, after, indent)

                if func is None:
                    name = idx.scope(rel, before) or idx.scope(rel, after)
                    scope, kind = container(name, module)
                    import_time(scope, kind, module, where)
                    record(rel, before, scope, "import-time")
                    continue

                fline = definitions[func]["line"]
                e = idx.statements[rel]
                if sb is not None and e["flines"][sb] == fline and e["cols"][sb] == indent:
                    statement_hit(rel, sb, where)  # runs only after the previous statement
                    resolution = "insertion/statement"
                elif sa is not None and e["flines"][sa] == fline and e["cols"][sa] == indent:
                    statement_hit(rel, sa, where)  # runs only before the next statement
                    resolution = "insertion/statement"
                else:
                    add_ids(symbol_tests.get(func, []), where, "function")
                    resolution = "insertion/function"
                record(rel, before, func, resolution)
                if self.walk_all:
                    walk_seeds.append(func)

        # ---- reverse-graph walk (import-time seeds, or everything with walk_all)
        depth = {}
        seeds = []
        for s in walk_seeds:
            if s not in seeds:
                seeds.append(s)
            if not self.no_ancestors and s in definitions:
                for anc in idx.ancestors(s):
                    if anc not in seeds:
                        seeds.append(anc)

        if seeds:
            module_definitions = {}

            for name, definition in definitions.items():
                module = idx.module(definition["file"])
                module_definitions.setdefault(module, []).append(name)
            depth, parent = walk_up(
                seeds,
                self.reverse_graph,
                definitions,
                self.max_depth,
                self.expand_modules,
                module_definitions,
            )
            seed_set = set(seeds)
            for node in depth:
                via = chain(node, parent)
                if node in definitions:
                    tests = symbol_tests.get(node, [])
                    if not tests and node in seed_set:
                        tests = module_tests.get(idx.module(definitions[node]["file"]), [])
                    add_ids(tests, via, "walk")
                elif node in seed_set or not self.ignore_module_dependents:
                    add_ids(module_tests.get(node, []), via, "walk")

        for tid, why in direct_tests.items():
            selected.setdefault(tid, {"via": why, "kind": "test-file"})

        # ---- widening (see WIDEN_LEVELS)
        touched = {idx.module(c["file"]) for c in changed
                   if c["resolution"] != "no-op" and c["file"].endswith(".py") and idx.has_file(c["file"])}
        for m in touched:
            if self.module_own:
                add_ids(module_tests.get(m, []), m, "widen-module")
            if self.module_touching:
                add_ids(self.module_touch_tests.get(m, []), m, "widen-touching")
        if self.expand_test_files and selected:
            files = {self.tests[t].get("file") for t in selected if t in self.tests}
            for f in files:
                add_ids(self.tests_by_file.get(f, []), f, "widen-file")

        if self.include_unmapped and (
            direct_tests or any(c["resolution"] != "no-op" for c in changed)
        ):
            add_numbers(self.unmapped_numbers, "no coverage data", "unmapped")

                # ---- final static dependency safety net
        # Add tests that statically depend on changed functions but were
        # missed by coverage, without changing tests already selected.

        changed_symbols = {
            change["scope"]
            for change in changed
            if change["resolution"] != "no-op"
            and change["scope"] in definitions
            and definitions[change["scope"]]["kind"] in ("function", "method")
        }

        if changed_symbols:
            for test_id, dependencies in self.test_dependencies.items():
                if test_id not in self.tests or test_id in selected:
                    continue

                matched = changed_symbols.intersection(dependencies)

                if matched:
                    selected[test_id] = {
                        "via": "static dependency: " + ", ".join(sorted(matched)),
                        "kind": "static-dependency",
                    }

        return {
            "selected": selected,
            "changed": changed,
            "warnings": warnings,
            "uncertain": uncertain,
            "depth": depth,
        }

    def numbers(self, selected):
        return sorted(n for n in (self.id_to_num[t] for t in selected) if n is not None)


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("changes", nargs="*", help="file:line or file:start-end")
    ap.add_argument("--git-diff", metavar="REF")
    ap.add_argument("--patch", metavar="FILE", help="unified diff ('-' for stdin); old-side lines are used")
    ap.add_argument("--repo", default="../sqlglot", help="git repo for --git-diff")
    ap.add_argument("--import-time", choices=["module", "touching", "walk", "all"], default="module",
                    help="policy for module/class-level changes coverage cannot see: "
                         "module = the module's OWN tests (default), "
                         "touching = every test that touches the module (wide), "
                         "walk = walk the reverse graph, all = run every test")
    ap.add_argument("--walk-all", action="store_true",
                    help="also walk the reverse graph from every changed symbol (old, wide behaviour)")
    ap.add_argument("--max-depth", type=int, default=None)
    ap.add_argument("--no-ancestors", action="store_true",
                    help="when walking, don't treat parent classes of a changed method as changed")
    ap.add_argument("--expand-modules", action="store_true")
    ap.add_argument("--ignore-module-dependents", action="store_true")
    ap.add_argument("--empty-fallback", choices=["module", "touching"], default=None,
                    help="when a change finds no tests, test the module (own / touching)")
    ap.add_argument("--no-unmapped", action="store_true",
                    help="don't add the tests that have no coverage data")
    ap.add_argument("--widen", type=int, choices=sorted(WIDEN_LEVELS), default=0,
                    help="widening level 0 (exact) .. 4 (wide), see WIDEN_LEVELS")
    ap.add_argument("--ids", action="store_true", help="print only test ids")
    ap.add_argument("--numbers", action="store_true", help="print only test numbers")
    ap.add_argument("--out", help="write full JSON report to this file")
    args = ap.parse_args()

    if not args.changes and not args.git_diff and not args.patch:
        ap.error("give file:line arguments, --git-diff REF or --patch FILE")

    test_map = load_json(TEST_MAP_FILE)["tests"]
    tests = {tid: {"number": m.get("number"), "file": m.get("file")} for tid, m in test_map.items()}

    sel = Selector(
        tests,
        import_time=args.import_time,
        walk_all=args.walk_all,
        max_depth=args.max_depth,
        no_ancestors=args.no_ancestors,
        expand_modules=args.expand_modules,
        ignore_module_dependents=args.ignore_module_dependents,
        empty_fallback=args.empty_fallback,
        include_unmapped=not args.no_unmapped,
        **WIDEN_LEVELS[args.widen],
    )

    raw = {}

    def merge(files):
        for f, e in files.items():
            cur = raw.setdefault(f, {"deleted": {}, "inserts": []})
            cur["deleted"].update(e["deleted"])
            cur["inserts"].extend(e["inserts"])

    for spec in args.changes:
        file, lines = parse_spec(spec)
        merge({file: {"deleted": {n: None for n in lines}, "inserts": []}})
    if args.git_diff:
        merge(changes_from_git(args.git_diff, args.repo))
    if args.patch:
        text = sys.stdin.read() if args.patch == "-" else open(args.patch, encoding="utf-8").read()
        merge(parse_unified_diff(text))

    result = sel.select(raw)
    
    selected = result["selected"]

    order = {
        "test-file": 0,
        "statement": 1,
        "function": 2,
        "import-time": 3,
        "walk": 4,
        "static-dependency": 5,
    }
    ordered = sorted(selected, key=lambda t: (order.get(selected[t]["kind"], 9), t))
    numbers = sel.numbers(ordered)

    if args.numbers or args.ids:
        print("\n".join(str(n) for n in numbers) if args.numbers else "\n".join(ordered))
        for w in result["warnings"]:
            print(w, file=sys.stderr)
        return

    by_kind = {}
    for t in ordered:
        by_kind[selected[t]["kind"]] = by_kind.get(selected[t]["kind"], 0) + 1

    report = {
        "changed": result["changed"],
        "affected_nodes": len(result["depth"]),
        "tests": {t: selected[t] for t in ordered},
        "numbers": numbers,
        "by_kind": by_kind,
        "total_tests": len(test_map),
        "summary": f"{len(ordered)} of {len(test_map)} tests selected",
        "warnings": result["warnings"],
        "uncertain": result["uncertain"],
    }

    text = json.dumps(report, indent=2)
    if args.out:
        open(args.out, "w", encoding="utf-8").write(text)
        print(report["summary"], "->", args.out)
    else:
        print(text)


if __name__ == "__main__":
    main()
