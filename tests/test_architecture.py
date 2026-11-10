"""Enforce the hexagonal dependency rule: dependencies point inward only.

adapters ──► application ──► domain
(bootstrap wires adapters to ports)
"""

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "src" / "screening_agent"

# Imports each layer must never use, by module prefix.
FORBIDDEN = {
    "domain": [
        "screening_agent.application",
        "screening_agent.adapters",
        "screening_agent.bootstrap",
        "screening_agent.config",
        "pipecat",
        "google",
        "loguru",
    ],
    "application": [
        "screening_agent.adapters",
        "screening_agent.bootstrap",
        "screening_agent.config",
        "pipecat",
        "google",
    ],
    "adapters/outbound": ["screening_agent.adapters.inbound", "screening_agent.bootstrap"],
}


def imported_modules(path: Path) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            modules.add(node.module)
    return modules


def layer_files():
    for layer, forbidden in FORBIDDEN.items():
        for path in sorted((PACKAGE / layer).rglob("*.py")):
            yield pytest.param(path, forbidden, id=str(path.relative_to(PACKAGE)))


@pytest.mark.parametrize(("path", "forbidden"), list(layer_files()))
def test_layer_imports_point_inward(path: Path, forbidden: list[str]):
    violations = sorted(
        module
        for module in imported_modules(path)
        if any(module == prefix or module.startswith(f"{prefix}.") for prefix in forbidden)
    )
    assert not violations, f"{path.name} imports {violations}"
