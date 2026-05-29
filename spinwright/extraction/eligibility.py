from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


# Categories of risky top-level module names. Matched against the dotted module
# path's first component (so "urllib.request" matches "urllib").
_NETWORK_TOP_LEVELS = {"socket", "urllib", "urllib3", "http", "requests", "httpx", "aiohttp", "ftplib", "smtplib"}
_SUBPROCESS_TOP_LEVELS = {"subprocess"}
_HYPOTHESIS_TOP_LEVELS = {"hypothesis"}

_FIXTURE_WHITELIST = {"tmp_path", "self", "cls"}

# pytest marker attributes that change how the test is invoked.
_INVOCATION_MARKERS = {"parametrize", "skip", "skipif", "skipunless", "xfail"}

# Decorator names that signal hypothesis usage at the decorator level.
_HYPOTHESIS_DECORATOR_NAMES = {"given"}


@dataclass(frozen=True)
class Reason:
    code: str
    message: str
    lineno: int | None = None


@dataclass(frozen=True)
class EligibilityResult:
    eligible: bool
    reasons: tuple[Reason, ...]


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------


def check(
    test_source_path: str | Path,
    nodeid: str,
    *,
    allow_pure_conftest_imports: bool = False,
) -> EligibilityResult:
    """AST-based eligibility check for a single test inside the given module.

    ``nodeid`` is a pytest nodeid; only the part after ``::`` (one or two
    components) is used. Anything earlier (file path) is ignored — the caller
    already supplied ``test_source_path``.
    """
    source_path = Path(test_source_path)
    source = source_path.read_text()
    tree = ast.parse(source, filename=str(source_path))

    reasons: list[Reason] = []

    # 1. Module-level imports
    aliases = _collect_aliases(tree)
    reasons.extend(_check_module_imports(tree, allow_pure_conftest_imports))

    # 2. Resolve the test node
    parts = nodeid.split("::")[1:]
    if not parts:
        reasons.append(Reason("nodeid_invalid", f"nodeid has no test component: {nodeid!r}"))
        return EligibilityResult(False, tuple(reasons))
    # Strip parametrize id suffix from the last component (e.g. "test_x[1]" → "test_x").
    parts[-1] = parts[-1].split("[", 1)[0]

    test_func, class_node = _resolve_test(tree, parts)
    if test_func is None:
        reasons.append(Reason("not_found", f"test {nodeid!r} not found in {source_path.name}"))
        return EligibilityResult(False, tuple(reasons))

    # 3. Decorator + signature + body checks on the test function itself
    reasons.extend(_check_decorators(test_func))
    reasons.extend(_check_signature(test_func))
    reasons.extend(_check_body(test_func, aliases))

    # 4. For unittest-style: also vet setUp / tearDown class methods
    if class_node is not None:
        reasons.extend(_check_unittest_lifecycle(class_node))

    return EligibilityResult(eligible=not reasons, reasons=tuple(reasons))


# ----------------------------------------------------------------------------
# Module-level helpers
# ----------------------------------------------------------------------------


def _collect_aliases(tree: ast.Module) -> dict[str, str]:
    """Map local names to fully-qualified module dotted paths."""
    aliases: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                local = alias.asname or alias.name.split(".")[0]
                aliases[local] = alias.name
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                local = alias.asname or alias.name
                aliases[local] = f"{module}.{alias.name}" if module else alias.name
    return aliases


def _check_module_imports(
    tree: ast.Module, allow_pure_conftest_imports: bool
) -> list[Reason]:
    reasons: list[Reason] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                reasons.extend(_classify_module_import(alias.name, node.lineno))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            # Relative or bare conftest import
            is_conftest = (
                module == "conftest"
                or module.endswith(".conftest")
                or (node.level > 0 and any(a.name == "conftest" for a in node.names))
            )
            if is_conftest and not allow_pure_conftest_imports:
                reasons.append(
                    Reason("conftest_import", f"imports from sibling conftest ({module or 'conftest'})", node.lineno)
                )
            if module:
                reasons.extend(_classify_module_import(module, node.lineno))
            for alias in node.names:
                if alias.name in _HYPOTHESIS_DECORATOR_NAMES and module.startswith("hypothesis"):
                    reasons.append(
                        Reason("hypothesis_import", f"imports hypothesis.{alias.name}", node.lineno)
                    )
    return reasons


def _classify_module_import(module: str, lineno: int) -> list[Reason]:
    # Per SPEC §5.2, only `hypothesis` and `conftest` imports are ineligible at
    # the import level. Network/subprocess/filesystem ineligibility is judged
    # by call-site usage in the test body — module-level imports are common
    # (e.g., a test file imports `subprocess` for one test but our target uses
    # only pure-Python paths) and would over-reject.
    out: list[Reason] = []
    top = module.split(".")[0]
    if top in _HYPOTHESIS_TOP_LEVELS:
        out.append(Reason("hypothesis_import", f"imports hypothesis ({module!r})", lineno))
    return out


# ----------------------------------------------------------------------------
# Test-node resolution
# ----------------------------------------------------------------------------


def _resolve_test(
    tree: ast.Module, parts: list[str]
) -> tuple[ast.FunctionDef | None, ast.ClassDef | None]:
    if len(parts) == 1:
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == parts[0]:
                return node, None
        return None, None
    if len(parts) == 2:
        class_name, method_name = parts
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        return item, node
                return None, node
    return None, None


# ----------------------------------------------------------------------------
# Decorator / signature checks
# ----------------------------------------------------------------------------


def _check_decorators(func: ast.FunctionDef) -> list[Reason]:
    reasons: list[Reason] = []
    for deco in func.decorator_list:
        # @given(...) or @given
        if isinstance(deco, ast.Name) and deco.id in _HYPOTHESIS_DECORATOR_NAMES:
            reasons.append(Reason("hypothesis_decorator", f"@{deco.id} decorator", deco.lineno))
            continue
        target = deco.func if isinstance(deco, ast.Call) else deco
        if isinstance(target, ast.Name) and target.id in _HYPOTHESIS_DECORATOR_NAMES:
            reasons.append(Reason("hypothesis_decorator", f"@{target.id} decorator", deco.lineno))
            continue
        # @pytest.mark.X or @something.mark.X
        attr_chain = _attribute_chain(target)
        if attr_chain and len(attr_chain) >= 3 and attr_chain[-2] == "mark":
            marker = attr_chain[-1]
            if marker in _INVOCATION_MARKERS:
                reasons.append(
                    Reason("pytest_marker", f"@{'.'.join(attr_chain)} changes test invocation", deco.lineno)
                )
    return reasons


def _check_signature(func: ast.FunctionDef) -> list[Reason]:
    reasons: list[Reason] = []
    args = func.args
    posonly = list(args.posonlyargs)
    pos = list(args.args)
    kwonly = list(args.kwonlyargs)
    all_named = posonly + pos + kwonly
    for arg in all_named:
        if arg.arg in _FIXTURE_WHITELIST:
            continue
        reasons.append(
            Reason("fixture_arg", f"signature uses non-whitelisted parameter {arg.arg!r}", func.lineno)
        )
    return reasons


# ----------------------------------------------------------------------------
# Body checks
# ----------------------------------------------------------------------------


def _check_body(func: ast.FunctionDef, aliases: dict[str, str]) -> list[Reason]:
    reasons: list[Reason] = []
    has_random_seed = False
    has_np_random_seed = False
    random_uses: list[tuple[int, str]] = []
    np_random_uses: list[tuple[int, str]] = []

    for node in ast.walk(func):
        if isinstance(node, ast.Call):
            dotted = _resolve_call_target(node.func, aliases)
            if dotted is None:
                # Anonymous calls (lambdas, indexed callables) — skip
                continue

            top = dotted.split(".")[0]

            # network / subprocess usage via attribute access
            if top in _NETWORK_TOP_LEVELS:
                reasons.append(
                    Reason("network_call", f"call to {dotted!r} touches network", node.lineno)
                )
            if top in _SUBPROCESS_TOP_LEVELS:
                reasons.append(
                    Reason("subprocess_call", f"call to {dotted!r} runs a subprocess", node.lineno)
                )

            # Filesystem outside tempfile: bare `open(...)` is the easy catch.
            if dotted == "open":
                reasons.append(
                    Reason("filesystem_open", "call to open() — direct filesystem access", node.lineno)
                )

            # Randomness — module form: random.X / numpy.random.X
            if dotted == "random.seed" or dotted.endswith(".random.seed"):
                if dotted == "random.seed":
                    has_random_seed = True
                else:
                    has_np_random_seed = True
            elif dotted.startswith("random."):
                random_uses.append((node.lineno, dotted))
            elif ".random." in dotted or dotted.endswith(".random"):
                # e.g. numpy.random.rand, np.random.normal — but exclude pyrandom
                if not dotted.startswith("random."):
                    np_random_uses.append((node.lineno, dotted))

    if random_uses and not has_random_seed:
        first = random_uses[0]
        reasons.append(
            Reason(
                "unseeded_random",
                f"uses {first[1]!r} without a prior random.seed(...)",
                first[0],
            )
        )
    if np_random_uses and not has_np_random_seed:
        first = np_random_uses[0]
        reasons.append(
            Reason(
                "unseeded_np_random",
                f"uses {first[1]!r} without a prior numpy.random.seed(...)",
                first[0],
            )
        )

    return reasons


# ----------------------------------------------------------------------------
# unittest setUp / tearDown checks
# ----------------------------------------------------------------------------


_TRIVIAL_LIFECYCLE = {"setUp", "tearDown", "setUpClass", "tearDownClass"}


def _check_unittest_lifecycle(cls: ast.ClassDef) -> list[Reason]:
    reasons: list[Reason] = []
    for item in cls.body:
        if isinstance(item, ast.FunctionDef) and item.name in _TRIVIAL_LIFECYCLE:
            if not _is_trivial_lifecycle_body(item.body):
                reasons.append(
                    Reason(
                        "unittest_lifecycle_nontrivial",
                        f"{cls.name}.{item.name} has non-trivial body (needs LLM extraction review)",
                        item.lineno,
                    )
                )
    return reasons


def _is_trivial_lifecycle_body(body: list[ast.stmt]) -> bool:
    """Trivial = body is only `pass`, simple self-attribute literal assignments,
    or super() lifecycle calls. Anything else (calls with side effects, control
    flow, comprehensions) is non-trivial."""
    for stmt in body:
        if isinstance(stmt, ast.Pass):
            continue
        if isinstance(stmt, ast.Expr) and _is_super_lifecycle_call(stmt.value):
            continue
        if isinstance(stmt, ast.Assign) and _is_simple_self_assign(stmt):
            continue
        return False
    return True


def _is_super_lifecycle_call(node: ast.expr) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Call)
        and isinstance(node.func.value.func, ast.Name)
        and node.func.value.func.id == "super"
        and node.func.attr in _TRIVIAL_LIFECYCLE
    )


def _is_simple_self_assign(stmt: ast.Assign) -> bool:
    if len(stmt.targets) != 1:
        return False
    target = stmt.targets[0]
    if not (isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == "self"):
        return False
    return _is_safe_literal(stmt.value)


def _is_safe_literal(node: ast.expr) -> bool:
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        return all(_is_safe_literal(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(_is_safe_literal(k) for k in node.keys if k is not None) \
            and all(_is_safe_literal(v) for v in node.values)
    return False


# ----------------------------------------------------------------------------
# Generic AST helpers
# ----------------------------------------------------------------------------


def _attribute_chain(node: ast.expr) -> list[str] | None:
    """Convert `a.b.c` (Attribute nested over Name) into ["a", "b", "c"].
    Returns None for shapes that aren't a clean dotted chain."""
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def _resolve_call_target(func: ast.expr, aliases: dict[str, str]) -> str | None:
    """Resolve a call's func into a fully-qualified dotted name using import aliases.

    Examples (with `import numpy as np`, `from random import randint`):
        np.random.rand     → "numpy.random.rand"
        random.choice      → "random.choice"  (assuming stdlib random imported)
        randint            → "random.randint"
        open               → "open"
    Returns None for non-dotted call shapes (e.g., attribute on a call result).
    """
    if isinstance(func, ast.Name):
        if func.id in aliases:
            return aliases[func.id]
        return func.id
    chain = _attribute_chain(func)
    if chain is None:
        return None
    head, *tail = chain
    resolved_head = aliases.get(head, head)
    return ".".join([resolved_head, *tail])
