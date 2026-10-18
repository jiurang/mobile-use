"""The APK import surface: what the agent must be able to import without extras.

The on-phone runner (runner-apk) ships a deliberately small dependency set: the
Android path (adbutils, uiautomator2, the langchain core, the cross-compiled
Rust wheels) and nothing else. Importing the agent on a real phone used to
require the cloud iOS transport stack anyway, because
``minitap.mobile_use.context`` carries an ``ios_client`` slot typed as
``IosClientWrapper`` at module scope — so ``import minitap.mobile_use.sdk``
dragged in idb, facebook-wda and Appium and died before the first screenshot.

The excluded set is **derived, not hand-maintained**: it is whatever this
project declares in ``pyproject.toml`` minus what the APK actually installs, read
from Chaquopy's build output. A hand-written list is how this test lied before —
``limrun_api`` was missing from it, so the probe happily reported OK while the
APK could not boot. Add a dependency to pyproject.toml and it is blocked here
automatically on the next run.

PyPI names that do not match their import name live in :data:`PYPI_TO_MODULE`.
That table is the only thing to maintain by hand, and a name in it that the APK
*does* install is reported by :func:`test_pypi_to_module_table_has_no_stale_entries`
rather than silently making a test vacuous.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Chaquopy's resolved package list for the debug variant. Its absence (a CI box
# that never built the APK) is not a failure — there is nothing to compare to.
APK_PACKAGE_DIR = (
    REPO_ROOT.parent
    / "mobile-manager"
    / "runner-apk"
    / "app"
    / "build"
    / "python"
    / "pip"
    / "debug"
    / "common"
)

# Distributions whose import name is not derivable from the wheel name.
PYPI_TO_MODULE = {
    "fb-idb": "idb",
    "facebook-wda": "wda",
    "appium-python-client": "appium",
    "opencv-python-headless": "cv2",
    "python-dotenv": "dotenv",
    "pillow": "PIL",
    "uuid-utils": "uuid_utils",
    "sseclient-py": "sseclient",
}


def _declared_distributions() -> set[str]:
    """Every distribution this project declares, main and extras."""
    with (REPO_ROOT / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project", {})
    names = {
        _name_of(req) for req in project.get("dependencies", []) if req
    }
    for extra in project.get("optional-dependencies", {}).values():
        names.update(_name_of(req) for req in extra if req)
    return names


def _name_of(requirement: str) -> str:
    return requirement.split("[")[0].split("<")[0].split(">")[0].split("=")[0].split("!")[0].split("~")[0].split(";")[0].strip()


def _apk_modules() -> set[str]:
    if not APK_PACKAGE_DIR.is_dir():
        return set()
    return {
        child.name.lower()
        for child in APK_PACKAGE_DIR.iterdir()
        if child.is_dir() and not child.name.endswith(".dist-info")
    }


def _excluded_modules() -> set[str] | None:
    """Module names the APK does not install but this project declares.

    Returns ``None`` when the APK has not been built, so callers can skip
    instead of asserting against an empty (meaningless) exclusion set.
    """
    installed = _apk_modules()
    if not installed:
        return None
    excluded = set()
    for dist in _declared_distributions():
        normalized = dist.lower().replace("_", "-")
        if normalized in PYPI_TO_MODULE:
            module = PYPI_TO_MODULE[normalized]
            if module.lower() in installed:
                continue  # APK ships it; mapping is stale
            excluded.add(module)
            continue
        # Wheel names use "-", import dirs use "_": langchain-core installs a
        # `langchain_core` package. Compare both spellings or the exclusion set
        # blocks packages the APK really has and the agent can never start.
        if {dist.lower(), normalized, normalized.replace("-", "_")} & installed:
            continue
        excluded.add(normalized.replace("-", "_"))
    return excluded


PROBE = textwrap.dedent(
    """
    import sys

    # sys.modules[name] = None makes `import name` raise ImportError by design
    # ("import of X halted; None in sys.modules"). A meta_path finder was tried
    # first and silently blocked nothing on this interpreter, which made every
    # step below report OK — hence the self-check, which is not optional.
    BLOCKED = {blocked!r}
    for _name in BLOCKED:
        sys.modules[_name] = None

    _leaked = []
    for _probe in BLOCKED:
        if not _probe:
            continue
        try:
            __import__(_probe)
            _leaked.append(_probe)
        except ImportError:
            pass
    if _leaked:
        print("BLOCKER_BROKEN=" + ",".join(_leaked))
        raise SystemExit(1)

    import minitap.mobile_use.sdk                                       # noqa: F401
    import minitap.mobile_use.context                                   # noqa: F401
    import minitap.mobile_use.controllers.android_controller            # noqa: F401
    import minitap.mobile_use.controllers.controller_factory           # noqa: F401
    import minitap.mobile_use.controllers.platform_specific_commands_controller  # noqa: F401,E501
    import minitap.mobile_use.clients.ui_automator_client             # noqa: F401
    import minitap.mobile_use.services.llm                             # noqa: F401
    import minitap.mobile_use.clients.browserstack_client as bs
    import minitap.mobile_use.clients.idb_client as idb
    import minitap.mobile_use.clients.wda_client as wdc

    # `mod is not None` filters out the sentinels this probe planted itself —
    # counting those would make the check pass/fail for the wrong reason.
    pulled = sorted(
        m for m, mod in sys.modules.items()
        if mod is not None and m.split(".")[0] in BLOCKED
    )
    print("PULLED=" + ",".join(pulled))
    print("WRAPPER=" + bs.BrowserStackClientWrapper.__name__)
    print("IDB=" + idb.IdbClientWrapper.__name__)
    print("WDA=" + wdc.WdaClientWrapper.__name__)
    """
)


def _excluded_or_skip() -> set[str]:
    excluded = _excluded_modules()
    if excluded is None:
        pytest.skip(
            f"no APK package list at {APK_PACKAGE_DIR}; build runner-apk first"
        )
    return excluded


def _run_probe() -> subprocess.CompletedProcess[str]:
    script = PROBE.format(blocked=sorted(_excluded_or_skip()))
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=180,
    )


def test_critical_exclusions_are_still_being_blocked():
    """Guards against the probe quietly going vacuous.

    The exclusion set is derived, so a future `pip install` in
    runner-apk/app/build.gradle that adds Appium or idb would silently make
    every "the agent imports fine" test pass without testing anything. These are
    the packages whose absence actually broke a real phone boot, so they must
    stay excluded for this suite to mean anything.
    """
    excluded = _excluded_or_skip()
    must_stay_blocked = {
        "idb",            # fb-idb: broke boot via context -> ios_client
        "wda",            # facebook-wda: same path
        "appium",         # via browserstack_client; selenium is its dependency
                          # and appium's import line comes first, so blocking
                          # appium alone still catches that regression
        "limrun_api",     # via sdk.agent
        "posthog",        # via services.llm -> telemetry
    }
    missing = sorted(p for p in must_stay_blocked if p not in excluded)
    assert missing == [], (
        f"these are installed in the APK now, so this suite no longer tests the "
        f"lazy-import regressions it exists for: {missing}"
    )


def test_apk_exclusion_is_actually_enforced():
    """Guards the guard.

    A simulator that silently blocks nothing makes every other test here pass
    for the wrong reason, and the failure only shows up later as an APK that
    will not boot. Assert the mechanism itself, not the imports.
    """
    excluded = _excluded_or_skip()
    proc = _run_probe()
    assert proc.returncode == 0, (
        "the APK dependency simulator is not working, so the tests below prove "
        f"nothing (excluded set was {len(excluded)} module(s)):\n"
        f"{proc.stdout}\n{proc.stderr}"
    )


def test_agent_imports_without_the_cloud_ios_stack():
    """The regression: `import minitap.mobile_use.sdk` needed idb, wda and Appium.

    `context` carries an optional `ios_client` slot typed as `IosClientWrapper`,
    and `ios_client` names the idb, wda, BrowserStack and cloud wrappers in that
    union — so importing the context on an Android phone pulled in the whole
    cloud iOS transport tree. Each wrapper's heavy third-party import is now
    lazy, which is what keeps this green.

    Run in a subprocess: the probe mutates ``sys.modules``.
    """
    _excluded_or_skip()
    proc = _run_probe()
    assert proc.returncode == 0, (
        "importing the agent pulled in a dependency the APK does not ship:\n"
        f"{proc.stdout}\n{proc.stderr}"
    )
    for expected in (
        "BrowserStackClientWrapper",
        "IdbClientWrapper",
        "WdaClientWrapper",
    ):
        assert f"={expected}" in proc.stdout, (
            f"{expected} must stay importable — laziness must not remove the class"
        )


def test_no_excluded_module_is_merely_imported_lazily_too_soon():
    """Lazy is only a fix if the module is not pulled in during import.

    A ``try/except ImportError`` around the old top-level import would pass the
    test above and still drag hundreds of modules into ``sys.modules``.
    """
    _excluded_or_skip()
    proc = _run_probe()
    assert proc.returncode == 0, proc.stderr
    line = next(
        (l for l in proc.stdout.splitlines() if l.startswith("PULLED=")), ""
    )
    pulled = [m for m in line.partition("=")[2].split(",") if m]
    assert pulled == [], (
        f"these modules were imported during agent startup: {pulled}"
    )
