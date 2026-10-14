import os
import re
from pathlib import Path

import numpy as np
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


E1000_SUBSET = Path(__file__).resolve().parent / "data" / "e1000_first6.csv"


def load_e1000_subset() -> list:
    """The first six series of the Empirical1000 dataset (a test-only copy, not packaged), as
    returned by ``get_dataset('e1000')[:6]``."""
    from pyhctsa.utils import _load_csv
    return _load_csv(str(E1000_SUBSET))


@pytest.fixture(scope="session")
def e1000_subset() -> list:
    return load_e1000_subset()


def e1000_available() -> bool:
    """The full Empirical1000 dataset is used only if it is already in the user cache or
    ``PYHCTSA_TEST_DOWNLOAD=1`` allows the tests to download it (33 MB)."""
    from pyhctsa.utils import _E1000, _data_cache_dir
    cached = os.path.exists(os.path.join(_data_cache_dir(), _E1000["filename"]))
    return cached or os.environ.get("PYHCTSA_TEST_DOWNLOAD") == "1"
