from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"
GITHUB_REMOTE = "git@github.com:afazeliUofT/revised_TWC.git"
PACKAGE_NAME = "BayesRoute_TWC_Gate0_v2_2_Full.zip"
EXPECTED_REVISION = "gate0_v2_2_20260808"
DOWNLOADS = Path("/mnt/c/Users/alifa/Downloads")


def run(
    cmd: list[str], cwd: Path | None = None, check: bool = True
) -> subprocess.CompletedProcess:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_package(local_root: Path) -> Path:
    for path in (local_root / PACKAGE_NAME, DOWNLOADS / PACKAGE_NAME, Path.cwd() / PACKAGE_NAME):
        if path.is_file():
            return path
    raise SystemExit(f"Missing {PACKAGE_NAME} in the working directory or {DOWNLOADS}")


def verify_manifest(root: Path) -> None:
    revision = json.loads((root / "PACKAGE_REVISION.json").read_text(encoding="utf-8"))
    if revision.get("revision") != EXPECTED_REVISION:
        raise SystemExit(f"Wrong package revision: {revision}")
    checked = 0
    for raw in (root / "MANIFEST.sha256").read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, rel = raw.split(None, 1)
        rel = rel.strip().lstrip("*")
        path = root / rel
        if not path.is_file() or sha256(path) != expected:
            raise SystemExit(f"Manifest failure: {rel}")
        checked += 1
    if checked < 45:
        raise SystemExit(f"Manifest unexpectedly small: {checked}")
    print(f"INTERNAL_MANIFEST_PASS: {checked} files")


def ensure_repo(root: Path) -> None:
    if not (root / ".git").exists():
        run(["git", "init"], cwd=root)
        run(["git", "branch", "-M", "main"], cwd=root)
        run(["git", "remote", "add", "origin", GITHUB_REMOTE], cwd=root)
    else:
        run(["git", "remote", "set-url", "origin", GITHUB_REMOTE], cwd=root)


def commit_push(root: Path) -> None:
    run(["git", "add", "-A"], cwd=root)
    changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root).returncode != 0
    if changed:
        run(["git", "commit", "-m", "Gate-0 v2.2: fair tuning and diagnostic workflow"], cwd=root)
    run(["git", "push", "-u", "origin", "main"], cwd=root)


def replace_managed_tree(extracted: Path, local_root: Path) -> None:
    managed_dirs = ["configs", "docs", "scripts", "slurm", "src", "wrappers"]
    managed_files = [
        ".gitignore",
        "README.md",
        "MANIFEST.sha256",
        "PACKAGE_REVISION.json",
        "pyproject.toml",
        "requirements_light.txt",
        "requirements_rorqual.txt",
    ]
    managed_files.extend(path.name for path in local_root.glob("RUN_*.py"))
    managed_files.extend(path.name for path in extracted.glob("RUN_*.py"))
    for name in sorted(set(managed_dirs + managed_files)):
        path = local_root / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
    for item in extracted.iterdir():
        if item.name in {"outputs", "results"}:
            continue
        target = local_root / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def main() -> None:
    global REMOTE, REMOTE_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default=REMOTE)
    ap.add_argument("--remote-root", default=REMOTE_ROOT)
    args = ap.parse_args()
    REMOTE = args.remote
    REMOTE_ROOT = args.remote_root

    local_root = Path.cwd().resolve()
    package = find_package(local_root)
    with tempfile.TemporaryDirectory(prefix="gate0_v22_setup_") as tmp:
        with zipfile.ZipFile(package, "r") as archive:
            archive.extractall(tmp)
        extracted = Path(tmp) / "bayesroute_rx_twc"
        if not extracted.is_dir():
            raise SystemExit("Unexpected package layout")
        verify_manifest(extracted)
        replace_managed_tree(extracted, local_root)

    shutil.rmtree(local_root / "results" / "from_rorqual", ignore_errors=True)
    (local_root / "results" / "from_rorqual").mkdir(parents=True, exist_ok=True)
    (local_root / "results" / "from_rorqual" / "README.md").write_text(
        "Awaiting Gate-0 v2.2 Rorqual evidence.\n", encoding="utf-8"
    )
    shutil.rmtree(local_root / "outputs", ignore_errors=True)
    (local_root / "outputs").mkdir(exist_ok=True)
    (local_root / "outputs" / ".keep").write_text("", encoding="utf-8")

    run(["python3", "-m", "compileall", "-q", "src", "scripts"], cwd=local_root)
    for path in sorted(local_root.rglob("__pycache__"), reverse=True):
        shutil.rmtree(path, ignore_errors=True)
    for path in local_root.rglob("*.pyc"):
        path.unlink(missing_ok=True)

    ensure_repo(local_root)
    commit_push(local_root)

    control_path = Path.home() / ".ssh" / "cm-revised-twc-rorqual"
    control_path.parent.mkdir(parents=True, exist_ok=True)
    ssh_opts = [
        "-o", "ControlMaster=auto",
        "-o", "ControlPersist=600",
        "-o", f"ControlPath={control_path}",
    ]
    rsync_ssh = "ssh " + " ".join(shlex.quote(x) for x in ssh_opts)
    run([
        "rsync", "-av", "--delete",
        "--exclude", ".git/",
        "--exclude", ".venv/",
        "--exclude", "results/",
        "--exclude", "outputs/",
        "--exclude", "*.zip",
        "--exclude", "__pycache__/",
        "--exclude", "*.pyc",
        "-e", rsync_ssh,
        f"{local_root}/",
        f"{REMOTE}:{REMOTE_ROOT}/",
    ])
    qroot = shlex.quote(REMOTE_ROOT)
    remote_command = f"""set -euo pipefail
cd {qroot}
rm -rf outputs/smoke outputs/optuna outputs/checkpoints outputs/logs \
  outputs/eval outputs/reports outputs/plots outputs/github_artifacts outputs/slurm outputs/gates
mkdir -p outputs/setup
python3 scripts/rorqual_setup_env.py --venv .venv --out outputs/setup
"""
    run(["ssh", *ssh_opts, REMOTE, remote_command])
    print("GATE0_V2_2_SETUP_DEPLOYED")
    print("Next: python3 RUN_01_SUBMIT_SMOKE_TEST.py")


if __name__ == "__main__":
    main()
