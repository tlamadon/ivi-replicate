"""Consistency checks for the replication package (CPU only, no fits)."""
import glob
import importlib.util
import os
import re
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [REPO, os.path.join(REPO, "scripts")]

spec = importlib.util.spec_from_file_location("replicate", os.path.join(REPO, "replicate.py"))
replicate = importlib.util.module_from_spec(spec)
sys.modules["replicate"] = replicate
spec.loader.exec_module(replicate)


@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("boot", [False, True])
def test_dag_is_valid(smoke, boot):
    tasks = replicate.toposort(replicate.all_tasks(smoke=smoke, with_bootstrap=boot))
    seen = set()
    for t in tasks:
        assert all(d in seen for d in t.deps), t.id
        assert os.path.exists(os.path.join(REPO, "scripts", t.script)), t.script
        seen.add(t.id)


def test_every_reference_bundle_is_produced():
    outputs = {o for t in replicate.all_tasks() for o in t.outputs}
    tracked = subprocess.run(["git", "-C", REPO, "ls-files", "output"],
                             capture_output=True, text=True).stdout.split()
    refs = {os.path.basename(p) for p in tracked
            if os.path.dirname(p) in ("output/bundles", "output/data")}
    assert set(replicate.BUNDLE_MAP.values()) == refs
    for src in replicate.BUNDLE_MAP:
        assert src in outputs, src


def test_summary_cells_match_reference_bundles():
    """Each fit listed in the summary bundles is a pipeline output."""
    import json
    outputs = {os.path.basename(o) for t in replicate.all_tasks() for o in t.outputs}
    for name in ("ar1n_parameter_summary", "hetero_scale_parameter_summary",
                 "hockeystick_parameter_summary", "hockeystick_parameter_summary_longt"):
        with open(os.path.join(REPO, replicate.BUNDLE_PATH[f"{name}.json"])) as f:
            fits = json.load(f)["fits"]
        fits = fits.values() if isinstance(fits, dict) else fits
        for fit in fits:
            assert os.path.basename(fit["source_file"]) in outputs, fit["source_file"]


def test_stochastic_entry_points_seed_everything():
    for t in replicate.all_tasks(with_bootstrap=True):
        src = open(os.path.join(REPO, "scripts", t.script)).read()
        if re.search(r"^import torch|^from mlye", src, re.M):
            assert "seed_everything(" in src, t.script


def test_scripthut_stack_matches_lockfile():
    lock = subprocess.run(
        ["uv", "export", "--frozen", "--no-dev", "--no-emit-project", "--no-hashes",
         "--no-header", "--no-annotate"],
        cwd=REPO, capture_output=True, text=True)
    if lock.returncode != 0:
        pytest.skip("uv not available")
    pins = sorted({l.split(";")[0].strip() for l in lock.stdout.splitlines()
                   if l.strip() and not l.startswith("#")})
    yaml_txt = open(os.path.join(REPO, "scripthut.yaml")).read()
    stack_pins = re.search(r'lock: "([^"]+)"', yaml_txt).group(1).split()
    assert sorted(stack_pins) == pins
    for p in pins:
        assert p in yaml_txt.split("prep:")[1], p


@pytest.mark.parametrize("script", sorted(
    os.path.basename(p) for p in glob.glob(os.path.join(REPO, "scripts", "*.py"))))
def test_experiment_imports(script):
    """Every experiment module imports against the trimmed `mlye`."""
    mod = script[:-3]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([REPO, os.path.join(REPO, "scripts")]),
           "METHOD": "vi", "BPP_EXT": "/nonexistent"}
    r = subprocess.run([sys.executable, "-c", f"import {mod}"], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=300)
    if r.returncode != 0 and "FileNotFoundError" in r.stderr and mod == "psid_bpp_table1_check":
        pytest.skip("table-1 check runs on import and needs the ICPSR files")
    assert r.returncode == 0, r.stderr[-2000:]


def test_hut_workflows_are_up_to_date(tmp_path, monkeypatch):
    """workflows/hut.replication_*.json must be exactly what `python replicate.py hut` writes."""
    monkeypatch.setattr(replicate, "REPO", str(tmp_path))
    replicate.cmd_hut(None)
    generated = sorted((tmp_path / "workflows").glob("hut.replication_*.json"))
    committed = sorted(glob.glob(os.path.join(REPO, "workflows", "hut.replication_*.json")))
    assert [p.name for p in generated] == [os.path.basename(p) for p in committed], \
        "run `python replicate.py hut`"
    for p in generated:
        assert open(os.path.join(REPO, "workflows", p.name)).read() == p.read_text(), \
            f"{p.name} stale: run `python replicate.py hut`"


def test_hut_workflows_end_with_collect():
    for p in glob.glob(os.path.join(REPO, "workflows", "hut.replication_*.json")):
        import json
        tasks = json.load(open(p))["tasks"]
        last = tasks[-1]
        assert last["id"].endswith("-collect"), p
        assert set(last["dependencies"]) == {t["id"] for t in tasks[:-1]}, p


def test_figure_registry():
    """Every artifact has a generator, a paper number and readable inputs;
    every bundle feeds at least one artifact; the compile/EPS tasks cover all."""
    bundles = set(replicate.BUNDLE_MAP.values())
    numbers = [f["number"] for f in replicate.FIGURES]
    assert len(numbers) == len(set(numbers)), "duplicate paper numbers"
    for fig in replicate.FIGURES:
        assert os.path.exists(os.path.join(REPO, "scripts", f"fig_{fig['name']}.py")), fig["name"]
        assert re.fullmatch(r"(fig|tab)[A-Z]?\d+", fig["number"]), fig
        for b in replicate.figure_bundles(fig):
            assert b in bundles, fig
    tasks = {t.id: t for t in replicate.all_tasks()}
    gens = {f"fig-{f['name']}" for f in replicate.FIGURES}
    assert set(tasks["fig-compile"].deps) == gens == set(tasks["fig-eps"].deps)
    assert len(tasks["fig-eps"].outputs) == sum(not f.get("online") for f in replicate.FIGURES)
    for t in tasks.values():
        assert t.layer == (1 if t.id == "bpp-data" or t.block == "data"
                           else 3 if t.block == "figures" else 2), t.id
    used = {b for f in replicate.FIGURES for b in replicate.figure_bundles(f)}
    assert used == bundles, f"bundles feeding no artifact: {bundles - used}"
