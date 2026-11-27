"""Find dependencies whose Python-version cap can be raised (run monthly by dependency-caps.yml).

A dependency listed in pyproject.toml with a marker such as ``python_version < '3.14'`` is installed only on the Python
versions for which it is known to publish wheels, so that a failed source build can never fail the pyhctsa install.
For each such cap, this looks up the dependency's latest release on PyPI and checks whether it has wheels for CPython
X.Y (the first version past the cap) on every platform pyhctsa publishes wheels for.
A cap is ready to raise only on that wheel evidence, which by itself shows that CPython X.Y exists.
If X.Y is covered, X.Y+1 and later are checked too, so a cap that has fallen behind is raised in one step.

Usage: python check_dependency_caps.py [PYPROJECT] [--json OUT]
PYPROJECT defaults to the repository's pyproject.toml; the docs lines named in the report are looked up next to it.
Only the package name is sent to PyPI.
"""
import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from packaging.requirements import Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.tags import Tag
from packaging.utils import InvalidWheelFilename, parse_wheel_filename
from packaging.version import InvalidVersion, Version

REPO = Path(__file__).resolve().parents[3]
DOCS = ["README.md", "docs/source/installation.rst"]  # state which Python versions get each library, and feature counts
USER_AGENT = "dependency-cap-check/1.0 (python-urllib)"  # generic: nothing about the repository or user
MAX_AHEAD = 4  # Python versions past the cap to check when the first is covered
MANYLINUX_GLIBC = (2, 28)  # pyhctsa's own Linux wheels are manylinux_2_28 (CIBW_MANYLINUX_*_IMAGE)

# a marker clause capping the Python version from above; confirmed by evaluating the marker (see find_caps)
_CAP = re.compile(r"""\bpython_(?:full_)?version\s*<\s*["'](\d+)\.(\d+)["']""")


@dataclass(frozen=True)
class Platform:
    name: str
    environment: dict  # marker variables (the Python version is added per check)
    matches: Callable[[str], bool]  # does a wheel platform tag install here?


def _manylinux(arch: str) -> Callable[[str], bool]:
    legacy = {"manylinux1": (2, 5), "manylinux2010": (2, 12), "manylinux2014": (2, 17)}

    def matches(platform_tag: str) -> bool:
        m = re.fullmatch(r"manylinux_(\d+)_(\d+)_(\w+)", platform_tag)
        if m:
            glibc, tag_arch = (int(m[1]), int(m[2])), m[3]
        else:
            m = re.fullmatch(r"(manylinux1|manylinux2010|manylinux2014)_(\w+)", platform_tag)
            if not m:
                return False
            glibc, tag_arch = legacy[m[1]], m[2]
        return tag_arch == arch and glibc <= MANYLINUX_GLIBC

    return matches


def _env(sys_platform: str, system: str, machine: str, os_name: str) -> dict:
    return {"sys_platform": sys_platform, "platform_system": system, "platform_machine": machine, "os_name": os_name,
            "implementation_name": "cpython", "platform_python_implementation": "CPython",
            "platform_release": "", "platform_version": ""}


# The platforms pyhctsa publishes wheels for (build_and_publish.yml: cibuildwheel with CIBW_ARCHS=auto64 on
# ubuntu-latest, ubuntu-24.04-arm, macos-latest (Apple silicon, so arm64 only) and windows-latest; manylinux_2_28, no
# musllinux). A cap is ready to raise when wheels cover all of these.
# Any macOS deployment target counts; on Linux, a wheel must need no newer glibc than pyhctsa's own wheels do.
PLATFORMS = [
    Platform("linux_x86_64", _env("linux", "Linux", "x86_64", "posix"), _manylinux("x86_64")),
    Platform("linux_aarch64", _env("linux", "Linux", "aarch64", "posix"), _manylinux("aarch64")),
    Platform("macos_arm64", _env("darwin", "Darwin", "arm64", "posix"),
             lambda tag: re.fullmatch(r"macosx_\d+_\d+_(arm64|universal2)", tag) is not None),
    Platform("windows_amd64", _env("win32", "Windows", "AMD64", "nt"), lambda tag: tag == "win_amd64"),
]


def _vstr(version: tuple) -> str:
    return f"{version[0]}.{version[1]}"


def marker_allows(req: Requirement, platform: Platform, python: tuple) -> bool:
    """Whether the requirement's marker installs it on this platform under CPython `python` (major, minor)."""
    if req.marker is None:
        return True
    env = dict(platform.environment, python_version=_vstr(python), python_full_version=f"{_vstr(python)}.0",
               implementation_version=f"{_vstr(python)}.0", extra="")
    return req.marker.evaluate(env)


@dataclass
class Cap:
    line: str  # the dependency string as written in pyproject.toml
    requirement: Requirement
    cap: tuple  # (major, minor): installed only on Python versions below this
    platforms: list  # the PLATFORMS it is installed on just below the cap: the ones its wheels must cover


def find_caps(dependencies: list, platforms: list = PLATFORMS) -> list:
    """The dependencies whose marker caps the Python version from above (``python_version < 'X.Y'``).

    A clause found in the marker text counts only if the marker really is a cap: it installs the dependency on some
    platform under X.(Y-1) and not under X.Y (so a clause inside an ``or`` that does not bind is ignored).
    """
    caps = []
    for line in dependencies:
        req = Requirement(line)
        if req.marker is None:
            continue
        for major, minor in sorted({(int(a), int(b)) for a, b in _CAP.findall(str(req.marker))}):
            if minor == 0:
                continue
            below, at = (major, minor - 1), (major, minor)
            covered = [p for p in platforms if marker_allows(req, p, below) and not marker_allows(req, p, at)]
            if covered:
                caps.append(Cap(line, req, at, covered))
                break
    return caps


def tag_supports_cpython(tag: Tag, python: tuple) -> bool:
    """Whether a wheel tag's interpreter and ABI install on CPython `python` (ignoring the platform).

    ``cpXY-cpXY`` for that version; ``cpNM-abi3`` for any N.M up to it (the stable ABI carries forward);
    ``cpXY-none``; and pure-Python ``py3``/``py3M`` (M up to the minor version) with ABI ``none``.
    Free-threaded (``cp314t``) and debug ABIs do not count.
    """
    major, minor = python
    cp = f"cp{major}{minor}"
    if tag.abi == "abi3":
        m = re.fullmatch(r"cp(\d)(\d+)", tag.interpreter)
        return m is not None and int(m[1]) == major and int(m[2]) <= minor
    if tag.abi == "none":
        if tag.interpreter == cp:
            return True
        m = re.fullmatch(r"py(\d)(\d*)", tag.interpreter)
        return m is not None and int(m[1]) == major and (m[2] == "" or int(m[2]) <= minor)
    return tag.interpreter == cp and tag.abi == cp


def wheel_coverage(files: list, python: tuple, platforms: list) -> dict:
    """For each platform, the wheels (PyPI file entries) that pip would install on CPython `python` there."""
    coverage = {p.name: [] for p in platforms}
    for f in files:
        if f.get("yanked") or not f["filename"].endswith(".whl"):
            continue
        requires_python = f.get("requires_python")
        if requires_python:
            try:
                if not SpecifierSet(requires_python).contains(_vstr(python)):
                    continue
            except InvalidSpecifier:
                pass
        try:
            tags = parse_wheel_filename(f["filename"])[3]
        except InvalidWheelFilename:
            continue
        for p in platforms:
            if any(tag_supports_cpython(t, python) and (t.platform == "any" or p.matches(t.platform)) for t in tags):
                coverage[p.name].append(f["filename"])
    return coverage


def latest_release(pypi: dict, specifier: SpecifierSet = SpecifierSet()) -> Optional[tuple]:
    """(version, files) of the newest release that is neither a pre-release nor fully yanked, within `specifier`."""
    releases = pypi.get("releases")
    if releases is None:  # (PyPI may drop "releases" from this API; "urls" are the latest release's files)
        releases = {pypi["info"]["version"]: pypi.get("urls", [])}
    best = None
    for version, files in releases.items():
        try:
            v = Version(version)
        except InvalidVersion:
            continue
        live = [f for f in files if not f.get("yanked")]
        if v.is_prerelease or not live or not specifier.contains(v):
            continue
        if best is None or v > best[0]:
            best = (v, live)
    return None if best is None else (str(best[0]), best[1])


def fetch_pypi(name: str) -> dict:
    """PyPI's JSON for a project; the request carries only the package name and a generic User-Agent."""
    request = urllib.request.Request(f"https://pypi.org/pypi/{name}/json",
                                     headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.load(response)
        except (urllib.error.URLError, TimeoutError):
            if attempt == 2:
                raise
            time.sleep(5 * (attempt + 1))


def docs_lines(root: Path, name: str, versions: list) -> list:
    """`file:line: text` for the docs lines that mention the package or one of the Python versions."""
    patterns = [re.compile(rf"\b{re.escape(name)}\b", re.IGNORECASE)]
    patterns += [re.compile(rf"(?<![\d.]){re.escape(v)}(?!\d)") for v in versions]
    found = []
    for rel in DOCS:
        path = root / rel
        if not path.exists():
            continue
        for i, text in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if any(p.search(text) for p in patterns):
                found.append(f"{rel}:{i}: {text.strip()}")
    return found


def issue_text(name: str, line: str, new_line: str, cap: tuple, new_cap: tuple, release: str, checks: list,
               docs: list) -> tuple:
    allowed = [c["python"] for c in checks if c["ready"]]
    versions = allowed[0] if len(allowed) == 1 else f"{allowed[0]} to {allowed[-1]}"
    title = f"Raise the Python-version cap for {name} to allow {_vstr(cap)}"
    wheels = "\n".join(f"- Python {c['python']}: " + ", ".join(f"`{p}`" for p in c["platforms"]) for c in checks
                       if c["ready"])
    body = f"""{name} {release} (its latest release on PyPI) publishes wheels for CPython {versions} on every platform \
pyhctsa builds wheels for:

{wheels}

So the cap that keeps it from being installed on Python {_vstr(cap)} and later can be raised to {_vstr(new_cap)}.

**In `pyproject.toml`** (`[project].dependencies`), change

```
"{line}",
```

to

```
"{new_line}",
```

and update the comment above it if it names the Python version.

**In the docs**, update which Python versions get {name} and the feature counts that follow from it \
(`README.md` and `docs/source/installation.rst`). Lines that mention {name} or the versions involved:

```
{chr(10).join(docs) if docs else "(none found)"}
```

**Before merging**, check that the unit-test CI (`run_unit_tests.yaml`) runs on Python {versions} with {name} \
installed and passes (add the version to its matrix if it is missing).
If CI fails there, keep the cap for now and leave this issue open: while it is open, the check opens no other.

_Opened by the monthly `dependency-caps.yml` workflow (`.github/workflows/ci/check_dependency_caps.py`)._
"""
    return title, body


def check(cap: Cap, pypi: dict, root: Path) -> dict:
    """Readiness of one capped dependency, from its PyPI JSON."""
    name = cap.requirement.name
    result = {"name": name, "requirement": cap.line, "cap": _vstr(cap.cap),
              "platforms": [p.name for p in cap.platforms], "release": None, "checks": [], "ready": False}
    latest = latest_release(pypi, cap.requirement.specifier)
    if latest is None:
        return result
    result["release"], files = latest
    python = cap.cap
    for _ in range(MAX_AHEAD):
        coverage = wheel_coverage(files, python, cap.platforms)
        covered = [p for p, wheels in coverage.items() if wheels]
        ready = len(covered) == len(cap.platforms)
        result["checks"].append({"python": _vstr(python), "ready": ready, "partial": bool(covered) and not ready,
                                 "platforms": covered, "missing": [p for p in coverage if p not in covered]})
        if not ready:
            break
        python = (python[0], python[1] + 1)
    allowed = [c["python"] for c in result["checks"] if c["ready"]]
    if allowed:
        major, minor = (int(x) for x in allowed[-1].split("."))
        new_cap = (major, minor + 1)  # one past the newest version with wheels everywhere
        new_line = _CAP.sub(lambda m: m[0].replace(f"{m[1]}.{m[2]}", _vstr(new_cap))
                            if (int(m[1]), int(m[2])) == cap.cap else m[0], cap.line)
        docs = docs_lines(root, name, [_vstr((cap.cap[0], cap.cap[1] - 1))] + allowed)
        title, body = issue_text(name, cap.line, new_line, cap.cap, new_cap, result["release"], result["checks"], docs)
        result.update(ready=True, allow=allowed, new_cap=_vstr(new_cap), new_requirement=new_line,
                      issue_title=title, issue_body=body)
    return result


def summarize(results: list) -> str:
    lines = []
    for r in results:
        head = f"{r['name']} (installed on Python < {r['cap']})"
        if r["release"] is None:
            lines.append(f"{head}: no non-yanked final release on PyPI")
            continue
        lines.append(f"{head}, latest release {r['release']}:")
        for c in r["checks"]:
            state = "READY" if c["ready"] else "partial" if c["partial"] else "no wheels"
            detail = f"wheels for {', '.join(c['platforms'])}" if c["platforms"] else ""
            if c["missing"] and c["platforms"]:
                detail += f"; none for {', '.join(c['missing'])}"
            lines.append(f"  CPython {c['python']}: {state}" + (f" ({detail})" if detail else ""))
        lines.append(f"  -> raise the cap to {r['new_cap']}" if r["ready"] else "  -> keep the cap")
    return "\n".join(lines)


def load_dependencies(pyproject: Path) -> list:
    import tomllib  # (Python 3.11+; imported here so the tests can load this module on 3.10)

    with open(pyproject, "rb") as f:
        return tomllib.load(f).get("project", {}).get("dependencies", [])


def main(argv: Optional[list] = None, fetch: Callable[[str], dict] = fetch_pypi) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("pyproject", nargs="?", type=Path, default=REPO / "pyproject.toml")
    parser.add_argument("--json", type=Path, help="write the machine-readable result here")
    args = parser.parse_args(argv)

    caps = find_caps(load_dependencies(args.pyproject))
    results = [check(cap, fetch(cap.requirement.name), args.pyproject.resolve().parent) for cap in caps]
    if results:
        text = summarize(results)
    else:
        text = f"No dependency in {args.pyproject} has a python_version upper cap: nothing to check."
    print(text)

    out = {"dependencies": results, "ready": [r["name"] for r in results if r["ready"]]}
    if args.json:
        args.json.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    if os.environ.get("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a", encoding="utf-8") as f:
            f.write(f"ready={json.dumps(out['ready'])}\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(f"### Dependency Python-version caps\n\n```\n{text}\n```\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
