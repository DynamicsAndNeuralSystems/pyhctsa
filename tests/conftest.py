import re
from pathlib import Path

import pytest

HCTSA_YAML = Path(__file__).resolve().parents[1] / "pyhctsa" / "configurations" / "hctsa.yaml"

def _module_sections() -> dict:
    # split hctsa.yaml on its top-level (unindented) module keys, keeping each
    # section as raw text so custom tags such as !range are left untouched
    sections, name = {}, None
    for line in HCTSA_YAML.read_text(encoding="utf-8").splitlines(keepends=True):
        match = re.match(r"^(\w+):\s*$", line)
        if match:
            name = match.group(1)
            sections[name] = ""
        if name:
            sections[name] += line
    return sections

@pytest.fixture
def module_config(tmp_path):
    """Factory that writes a single-module config, taken from hctsa.yaml, to a temp file."""
    sections = _module_sections()
    def _make(module: str) -> str:
        path = tmp_path / f"{module}.yaml"
        path.write_text(sections[module], encoding="utf-8")
        return str(path)
    return _make
