"""
Line -> scope lookup, plus a statement-level index.

Build (once, on the BASELINE code, after the code graph exists):
    python scope_lookup.py build --source-root ../sqlglot/sqlglot

Query:
    python scope_lookup.py query optimizer/simplify.py 120
    python scope_lookup.py query sqlglot/optimizer/simplify.py:120

Resolution rule (scope):
    innermost definition whose [line, end_line] contains the line,
    otherwise the module itself.

Statement index (statement_index.json), one slot per statement:
    simple statement      -> its full line span
    compound statement    -> its header only (if/for/while/with/try/except/case)
    def / class           -> not a statement (their lines run at import time)
    per slot: first/last line, column, and the def-line of the enclosing
    function (0 = module/class level, i.e. executed at import time)
"""

import argparse
import ast
import bisect
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
GRAPH_DIR = BASE_DIR / "code_graph"

DEFINITIONS_FILE = GRAPH_DIR / "definitions_by_name.json"
INDEX_FILE = GRAPH_DIR / "line_index.json"
STATEMENT_INDEX_FILE = GRAPH_DIR / "statement_index.json"
SYMBOL_TESTS_FILE = GRAPH_DIR / "symbol_to_tests.json"
MODULE_TESTS_FILE = GRAPH_DIR / "module_to_tests.json"            # module's OWN tests (narrow)
MODULE_TOUCH_TESTS_FILE = GRAPH_DIR / "module_to_all_tests.json"  # every test touching the module (wide)
STATEMENT_TESTS_FILE = GRAPH_DIR / "statement_to_tests.json"
UNMAPPED_TESTS_FILE = GRAPH_DIR / "unmapped_tests.json"  # tests with NO coverage (numbers)

DEFAULT_SOURCE_ROOT = BASE_DIR.parent / "sqlglot" / "sqlglot"


# ============================================================
# HELPERS
# ============================================================

def module_from_file(file):
    parts = list(Path(file).with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    return ".".join(["sqlglot"] + parts)


def normalise_file(path):
    """
    Accepts 'optimizer/simplify.py', 'sqlglot/optimizer/simplify.py'
    or 'sqlglot/sqlglot/optimizer/simplify.py' and returns the form used
    in definitions ('optimizer/simplify.py').
    """
    path = str(path).replace("\\", "/")
    marker = "sqlglot/sqlglot/"
    if marker in path:
        return path.split(marker, 1)[1]
    if path.startswith("sqlglot/"):
        return path[len("sqlglot/"):]
    return path


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# BUILD 1: flatten nested definitions into non-overlapping segments
# ============================================================

def flatten_file(defs):
    defs = sorted(defs, key=lambda d: (d["line"], -d["end_line"]))
    segments = []
    stack = []
    cursor = 0

    def emit_until(limit):
        nonlocal cursor
        while stack and stack[-1]["end_line"] < limit:
            top = stack.pop()
            start = max(cursor, top["line"])
            if start <= top["end_line"]:
                segments.append((start, top["end_line"], top["name"]))
            cursor = top["end_line"] + 1

    for d in defs:
        emit_until(d["line"])
        if stack and cursor < d["line"]:
            segments.append((cursor, d["line"] - 1, stack[-1]["name"]))
        cursor = d["line"]
        stack.append(d)

    emit_until(float("inf"))
    return segments


def build_line_index():
    definitions = load_json(DEFINITIONS_FILE)
    if isinstance(definitions, list):
        definitions = {d["name"]: d for d in definitions}

    by_file = {}
    for d in definitions.values():
        by_file.setdefault(d["file"], []).append(d)

    index = {}
    for file, defs in by_file.items():
        segs = flatten_file(defs)
        index[file] = {
            "module": module_from_file(file),
            "starts": [s[0] for s in segs],
            "ends": [s[1] for s in segs],
            "names": [s[2] for s in segs],
        }

    GRAPH_DIR.mkdir(exist_ok=True)
    INDEX_FILE.write_text(json.dumps(index), encoding="utf-8")
    print(f"Line index: {len(index)} files -> {INDEX_FILE}")


# ============================================================
# BUILD 2: statement index
# ============================================================

FUNC_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
NOT_STATEMENTS = FUNC_TYPES + (ast.ClassDef,)
MATCH_CASE = getattr(ast, "match_case", None)


def statement_span(node):
    """Line span that coverage attributes to this node, or None."""
    start = getattr(node, "lineno", None)
    if start is None:  # match_case has no lineno
        start = node.pattern.lineno

    body = getattr(node, "body", None)
    if isinstance(body, list) and body:  # compound: header only
        first = body[0].lineno
        if first <= start:  # one-liner `if x: y` -> the body statement covers it
            return None
        return start, first - 1

    cases = getattr(node, "cases", None)
    if isinstance(cases, list) and cases:  # match statement
        first = cases[0].pattern.lineno
        return (start, first - 1) if first > start else None

    return start, node.end_lineno


def collect_statements(node, func, spans, func_cols):
    """func = (def_line, col) of the innermost enclosing function, or None."""
    for child in ast.iter_child_nodes(node):
        child_func = func

        if isinstance(child, FUNC_TYPES):
            child_func = (child.lineno, child.col_offset)
            func_cols[child.lineno] = child.col_offset
        elif isinstance(child, ast.stmt) or isinstance(child, ast.ExceptHandler) or (
            MATCH_CASE and isinstance(child, MATCH_CASE)
        ):
            if not isinstance(child, NOT_STATEMENTS):
                span = statement_span(child)
                if span:
                    col = getattr(child, "col_offset", None)
                    if col is None:
                        col = child.pattern.col_offset
                    spans.append((span[0], span[1], col, func[0] if func else 0))

        collect_statements(child, child_func, spans, func_cols)


def build_statement_index(source_root):
    source_root = Path(source_root)
    index = {}

    for path in sorted(source_root.rglob("*.py")):
        rel = path.relative_to(source_root).as_posix()
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except (SyntaxError, UnicodeDecodeError):
            continue

        spans, func_cols = [], {}
        collect_statements(tree, None, spans, func_cols)
        spans.sort()

        index[rel] = {
            "starts": [s[0] for s in spans],
            "ends": [s[1] for s in spans],
            "cols": [s[2] for s in spans],
            "flines": [s[3] for s in spans],
            "func_cols": {str(k): v for k, v in func_cols.items()},
        }

    GRAPH_DIR.mkdir(exist_ok=True)
    STATEMENT_INDEX_FILE.write_text(json.dumps(index), encoding="utf-8")
    print(f"Statement index: {len(index)} files -> {STATEMENT_INDEX_FILE}")


# ============================================================
# QUERY
# ============================================================

class LineIndex:
    def __init__(self):
        self.definitions = load_json(DEFINITIONS_FILE)
        if isinstance(self.definitions, list):
            self.definitions = {d["name"]: d for d in self.definitions}
        self.index = load_json(INDEX_FILE)
        self.statements = (
            load_json(STATEMENT_INDEX_FILE) if STATEMENT_INDEX_FILE.exists() else {}
        )

        # functions/methods per file, sorted by def line (for nearest_function)
        self._funcs = {}
        for d in self.definitions.values():
            if d["kind"] in ("function", "method"):
                self._funcs.setdefault(d["file"], []).append(d)
        for lst in self._funcs.values():
            lst.sort(key=lambda d: d["line"])
        self._func_starts = {f: [d["line"] for d in lst] for f, lst in self._funcs.items()}

    def has_file(self, file):
        return file in self.index

    def module(self, file):
        entry = self.index.get(file)
        return entry["module"] if entry else module_from_file(file)

    def scope(self, file, line):
        """Innermost definition name containing the line, or None (module level)."""
        entry = self.index.get(file)
        if not entry:
            return None
        i = bisect.bisect_right(entry["starts"], line) - 1
        if i >= 0 and line <= entry["ends"][i]:
            return entry["names"][i]
        return None

    def statement_slot(self, file, line):
        """Index of the statement slot containing the line, or None."""
        entry = self.statements.get(file)
        if not entry:
            return None
        i = bisect.bisect_right(entry["starts"], line) - 1
        if i >= 0 and line <= entry["ends"][i]:
            return i
        return None

    def slot_in_function(self, file, slot):
        return bool(self.statements[file]["flines"][slot])

    def func_col(self, file, def_line):
        return self.statements.get(file, {}).get("func_cols", {}).get(str(def_line))

    def ancestors(self, name):
        """Enclosing definitions, e.g. mod.Cls.method -> [mod.Cls]."""
        out = []
        parts = name.split(".")
        for n in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:n])
            if candidate in self.definitions:
                out.append(candidate)
        return out

    def nearest_function(self, file, line):
        """
        Closest function/method to a line that sits outside every function body
        (decorators, gaps between functions...). Ties go to the function below.
        """
        funcs = self._funcs.get(file)
        if not funcs:
            return None
        i = bisect.bisect_right(self._func_starts[file], line)
        nxt = funcs[i] if i < len(funcs) else None
        j = i - 1
        while j >= 0 and funcs[j]["end_line"] >= line:  # skip functions that contain the line
            j -= 1
        prv = funcs[j] if j >= 0 else None
        if nxt and prv:
            return nxt["name"] if nxt["line"] - line <= line - prv["end_line"] else prv["name"]
        best = nxt or prv
        return best["name"] if best else None

    def resolve(self, file, line, snap=False):
        """
        snap=True: a line that is not inside any function and not part of a
        statement (decorator, gap) resolves to the CLOSEST function instead of
        the module/class. A class header line is never snapped.
        """
        file = normalise_file(file)
        name = self.scope(file, line)
        module = self.module(file)
        snapped = False

        if snap and self.statement_slot(file, line) is None:
            is_class_header = bool(name) and line == self.definitions[name]["line"]
            if not is_class_header and (name is None or self.definitions[name]["kind"] == "class"):
                near = self.nearest_function(file, line)
                if near:
                    name, snapped = near, True

        return {
            "file": file,
            "line": line,
            "module": module,
            "scope": name or module,
            "kind": self.definitions[name]["kind"] if name else "module",
            "ancestors": self.ancestors(name) if name else [],
            "snapped": snapped,
        }


def query(file, line):
    idx = LineIndex()
    result = idx.resolve(file, line, snap=True)

    slot = idx.statement_slot(result["file"], line)
    if slot is not None:
        e = idx.statements[result["file"]]
        result["statement"] = {
            "lines": [e["starts"][slot], e["ends"][slot]],
            "in_function": idx.slot_in_function(result["file"], slot),
        }

    tests = None
    if SYMBOL_TESTS_FILE.exists():
        if result["kind"] == "module":
            if MODULE_TESTS_FILE.exists():
                tests = load_json(MODULE_TESTS_FILE).get(result["module"], [])
        else:
            tests = load_json(SYMBOL_TESTS_FILE).get(result["scope"], [])
    if tests is not None:
        result["tests_at_scope_level"] = tests
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT),
                   help="folder holding the baseline sqlglot/*.py sources")
    q = sub.add_parser("query")
    q.add_argument("file", help="path, or path:line")
    q.add_argument("line", nargs="?", type=int)
    args = parser.parse_args()

    if args.cmd == "build":
        build_line_index()
        build_statement_index(args.source_root)
    else:
        file, line = args.file, args.line
        if line is None:
            file, _, line = file.rpartition(":")
            line = int(line)
        query(file, line)


if __name__ == "__main__":
    main()
