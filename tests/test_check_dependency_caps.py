""".github/workflows/ci/check_dependency_caps.py (the monthly check for dependency Python-version caps that can be
raised), on canned PyPI JSON: no network."""
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from packaging.tags import Tag

SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "ci" / "check_dependency_caps.py"
spec = importlib.util.spec_from_file_location("check_dependency_caps", SCRIPT)
chk = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = chk  # (dataclasses look their module up here)
spec.loader.exec_module(chk)

HMM = "hmmlearn; python_version < '3.14' and (sys_platform != 'win32' or platform_machine != 'ARM64')"
ALL = ["linux_x86_64", "linux_aarch64", "macos_arm64", "windows_amd64"]


def _files(*names, yanked=(), requires_python=None):
    return [{"filename": n, "yanked": n in yanked, "requires_python": requires_python} for n in names]


def _wheels(name, version, interp_abi):
    """The usual cibuildwheel set of wheels for one interpreter/ABI pair."""
    return [f"{name}-{version}-{interp_abi}-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
            f"{name}-{version}-{interp_abi}-manylinux_2_17_aarch64.manylinux2014_aarch64.whl",
            f"{name}-{version}-{interp_abi}-macosx_11_0_arm64.whl",
            f"{name}-{version}-{interp_abi}-macosx_10_9_x86_64.whl",
            f"{name}-{version}-{interp_abi}-win_amd64.whl"]


def _pypi(releases):
    return {"info": {"version": max(releases)}, "releases": releases}


# ---- finding caps in the dependency list ----
def test_no_cap_is_a_no_op():
    assert chk.find_caps(["numpy>=2", "scipy", "hmmlearn", "arch; sys_platform != 'win32'"]) == []


def test_finds_caps_without_a_list_of_names():
    caps = chk.find_caps(["numpy>=2", HMM, "ripser; python_version < '3.15'"])
    assert [(c.requirement.name, c.cap) for c in caps] == [("hmmlearn", (3, 14)), ("ripser", (3, 15))]
    assert caps[0].line == HMM
    # Windows on ARM is excluded by the marker, but is not one of pyhctsa's wheel platforms anyway
    assert [p.name for p in caps[0].platforms] == ALL


def test_cap_must_bind():
    # a clause inside an `or` only caps the platforms the other branch does not cover
    caps = chk.find_caps(["foo; python_version < '3.14' or sys_platform == 'linux'"])
    assert [p.name for p in caps[0].platforms] == ["macos_arm64", "windows_amd64"]
    # one that never binds on pyhctsa's platforms is not a cap
    assert chk.find_caps(["foo; python_version < '3.14' or sys_platform != 'cygwin'"]) == []
    # nor is a lower bound
    assert chk.find_caps(["foo; python_version >= '3.11'"]) == []


def test_full_version_cap_and_double_quotes():
    caps = chk.find_caps(['foo; python_full_version < "3.13"'])
    assert caps[0].cap == (3, 13)


# ---- matching wheel tags ----
@pytest.mark.parametrize("tag, ok", [
    (Tag("cp314", "cp314", "win_amd64"), True),
    (Tag("cp313", "cp313", "win_amd64"), False),
    (Tag("cp314", "cp314t", "win_amd64"), False),  # free-threaded
    (Tag("cp39", "abi3", "win_amd64"), True),  # the stable ABI carries forward
    (Tag("cp315", "abi3", "win_amd64"), False),
    (Tag("cp214", "abi3", "win_amd64"), False),
    (Tag("py3", "none", "any"), True),
    (Tag("py38", "none", "any"), True),
    (Tag("py2", "none", "any"), False),
    (Tag("cp314", "none", "win_amd64"), True),
])
def test_tag_supports_cpython(tag, ok):
    assert chk.tag_supports_cpython(tag, (3, 14)) is ok


@pytest.mark.parametrize("platform, tag, ok", [
    ("linux_x86_64", "manylinux_2_28_x86_64", True),
    ("linux_x86_64", "manylinux2014_x86_64", True),
    ("linux_x86_64", "manylinux_2_34_x86_64", False),  # needs a newer glibc than pyhctsa's own wheels
    ("linux_x86_64", "musllinux_1_2_x86_64", False),
    ("linux_x86_64", "manylinux_2_17_aarch64", False),
    ("linux_aarch64", "manylinux2014_aarch64", True),
    ("macos_arm64", "macosx_14_0_arm64", True),
    ("macos_arm64", "macosx_10_9_universal2", True),
    ("macos_arm64", "macosx_10_9_x86_64", False),
    ("windows_amd64", "win_amd64", True),
    ("windows_amd64", "win_arm64", False),
    ("windows_amd64", "win32", False),
])
def test_platform_tags(platform, tag, ok):
    assert {p.name: p for p in chk.PLATFORMS}[platform].matches(tag) is ok


def test_coverage_full_partial_and_none():
    full = _files(*_wheels("foo", "1.0", "cp314-cp314"))
    assert all(chk.wheel_coverage(full, (3, 14), chk.PLATFORMS).values())
    partial = _files("foo-1.0-cp314-cp314-win_amd64.whl", "foo-1.0-cp314-cp314-macosx_11_0_arm64.whl",
                     "foo-1.0.tar.gz")
    cov = chk.wheel_coverage(partial, (3, 14), chk.PLATFORMS)
    assert [p for p, w in cov.items() if w] == ["macos_arm64", "windows_amd64"]
    assert not any(chk.wheel_coverage(full, (3, 15), chk.PLATFORMS).values())


def test_coverage_abi3_and_pure_python():
    abi3 = _files(*_wheels("foo", "1.0", "cp310-abi3"))
    assert all(chk.wheel_coverage(abi3, (3, 15), chk.PLATFORMS).values())
    pure = _files("foo-1.0-py3-none-any.whl")
    assert all(chk.wheel_coverage(pure, (3, 15), chk.PLATFORMS).values())


def test_coverage_skips_yanked_and_requires_python():
    full = _wheels("foo", "1.0", "cp314-cp314")
    assert not any(chk.wheel_coverage(_files(*full, yanked=full), (3, 14), chk.PLATFORMS).values())
    assert not any(chk.wheel_coverage(_files(*full, requires_python="<3.14"), (3, 14), chk.PLATFORMS).values())
    assert all(chk.wheel_coverage(_files(*full, requires_python=">=3.10"), (3, 14), chk.PLATFORMS).values())


def test_latest_release_skips_prereleases_and_yanked():
    pypi = _pypi({"1.0": _files("foo-1.0.tar.gz"),
                  "1.1": _files("foo-1.1.tar.gz", yanked=["foo-1.1.tar.gz"]),
                  "2.0rc1": _files("foo-2.0rc1.tar.gz")})
    assert chk.latest_release(pypi)[0] == "1.0"
    # without "releases", the latest release's own files
    assert chk.latest_release({"info": {"version": "3.0"}, "urls": _files("foo-3.0.tar.gz")})[0] == "3.0"


# ---- whole checks ----
def test_check_not_ready_partial():
    cap = chk.find_caps([HMM])[0]
    pypi = _pypi({"0.3.3": _files("hmmlearn-0.3.3.tar.gz", "hmmlearn-0.3.3-cp314-cp314-win_amd64.whl",
                                  *_wheels("hmmlearn", "0.3.3", "cp313-cp313"))})
    r = chk.check(cap, pypi, Path("."))
    assert not r["ready"] and r["release"] == "0.3.3"
    assert r["checks"] == [{"python": "3.14", "ready": False, "partial": True, "platforms": ["windows_amd64"],
                            "missing": ["linux_x86_64", "linux_aarch64", "macos_arm64"]}]
    assert "partial" in chk.summarize([r]) and "keep the cap" in chk.summarize([r])


def test_check_ready_raises_past_every_covered_version(tmp_path):
    (tmp_path / "README.md").write_text("Python 3.10 to 3.13 gets hmmlearn.\nUnrelated line.\n")
    cap = chk.find_caps([HMM])[0]
    pypi = _pypi({"0.4.0": _files(*_wheels("hmmlearn", "0.4.0", "cp314-cp314"),
                                  *_wheels("hmmlearn", "0.4.0", "cp315-cp315"))})
    r = chk.check(cap, pypi, tmp_path)
    assert r["ready"] and r["allow"] == ["3.14", "3.15"] and r["new_cap"] == "3.16"
    assert r["new_requirement"] == HMM.replace("3.14", "3.16")
    assert r["issue_title"] == "Raise the Python-version cap for hmmlearn to allow 3.14"
    assert HMM in r["issue_body"] and r["new_requirement"] in r["issue_body"]
    assert "README.md:1: Python 3.10 to 3.13 gets hmmlearn." in r["issue_body"]
    assert "Unrelated" not in r["issue_body"]


def test_main_no_cap_is_a_no_op(tmp_path, capsys, monkeypatch):
    pytest.importorskip("tomllib")
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    (tmp_path / "pyproject.toml").write_text('[project]\ndependencies = ["numpy>=2", "hmmlearn"]\n')

    def no_network(name):
        raise AssertionError("PyPI queried without a cap")

    assert chk.main([str(tmp_path / "pyproject.toml"), "--json", str(tmp_path / "out.json")], fetch=no_network) == 0
    assert "nothing to check" in capsys.readouterr().out
    assert json.loads((tmp_path / "out.json").read_text()) == {"dependencies": [], "ready": []}


def test_main_writes_github_output(tmp_path, monkeypatch):
    pytest.importorskip("tomllib")
    out = tmp_path / "github_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(out))
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    (tmp_path / "pyproject.toml").write_text(f'[project]\ndependencies = ["numpy", "{HMM}"]\n')
    pypi = _pypi({"0.4.0": _files(*_wheels("hmmlearn", "0.4.0", "cp314-cp314"))})
    assert chk.main([str(tmp_path / "pyproject.toml")], fetch=lambda name: pypi) == 0
    assert out.read_text() == 'ready=["hmmlearn"]\n'
