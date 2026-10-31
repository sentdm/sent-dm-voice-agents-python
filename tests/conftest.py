from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


def fixture_files(kind: str) -> list[Path]:
    return sorted((FIXTURES / kind).glob("*/*.json"))


@pytest.fixture
def load():
    return lambda path: json.loads(Path(path).read_text())
