#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ZIP_NAME = "BayesRoute_TWC_Gate0_Mechanism_Diagnostic_v1.zip"
EXPECTED_SHA256 = "4398fd1c6523997dcf1f098564a3a8498f063388319fa2d7ab776110f9502ce5"
PATCH_ROOT_NAME = "BayesRoute_TWC_Gate0_Mechanism_Diagnostic_v1"
EXPECTED_PACKAGE_REVISION = "gate0_v2_4_20260809"
REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"
GITHUB_REMOTE = "git@github.com:afazeliUofT/revised_TWC.git"
DOWNLOADS = Path("/mnt/c/Users/alifa/Downloads")


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(str(value) for value in cmd), flush=True)
    proc = subprocess.run([str(value) for value in cmd], cwd=cwd, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_zip(local_root: Path) -> Path:
    candidates = [DOWNLOADS / ZIP_NAME, local_root / ZIP_NAME]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"Missing {ZIP_NAME}. Put it in {DOWNLOADS} or {local_root}."
    )


def safe_extract(zip_path: Path, destination: Path) -> None:
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as archive:
        root = destination.resolve()
        for info in archive.infolist():
            target = (destination / info.filename).resolve()
            if root != target and root not in target.parents:
                raise RuntimeError(f"Unsafe ZIP path: {info.filename}")
        archive.extractall(destination)


def verify_manifest(patch_root: Path) -> int:
    manifest = patch_root / "DIAGNOSTIC_MANIFEST.sha256"
    if not manifest.is_file():
        raise RuntimeError("DIAGNOSTIC_MANIFEST.sha256 is missing")
    checked = 0
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(None, 1)
        relative = relative.strip().lstrip("*")
        path = patch_root / relative
        if not path.is_file():
            raise RuntimeError(f"Manifest file is missing: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Manifest mismatch: {relative}\nexpected={expected}\nactual={actual}"
            )
        checked += 1
    if checked != 9:
        raise RuntimeError(f"Expected 9 diagnostic files, verified {checked}")
    return checked


def verify_local_revision(local_root: Path) -> None:
    revision_path = local_root / "PACKAGE_REVISION.json"
    if not revision_path.is_file():
        raise SystemExit("Run this wrapper from /home/afazeli2006/revised_TWC")
    data = json.loads(revision_path.read_text(encoding="utf-8"))
    if data.get("revision") != EXPECTED_PACKAGE_REVISION:
        raise SystemExit(
            "Incompatible package revision: "
            f"{data.get('revision')} != {EXPECTED_PACKAGE_REVISION}"
        )


def copy_patch(patch_root: Path, local_root: Path) -> None:
    for path in patch_root.rglob("*"):
        if not path.is_file() or path.name == "DIAGNOSTIC_MANIFEST.sha256":
            continue
        relative = path.relative_to(patch_root)
        target = local_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        if target.suffix in {".py", ".sbatch"}:
            target.chmod(target.stat().st_mode | 0o111)
    manifest_target = local_root / "diagnostics" / "GATE0_MECHANISM_DIAGNOSTIC_MANIFEST.sha256"
    manifest_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(patch_root / "DIAGNOSTIC_MANIFEST.sha256", manifest_target)


def commit_push(local_root: Path) -> None:
    run(["git", "remote", "set-url", "origin", GITHUB_REMOTE], cwd=local_root)
    run(["git", "add", "-A"], cwd=local_root)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=local_root
    ).returncode != 0
    if changed:
        run(
            [
                "git",
                "commit",
                "-m",
                "Gate-0 diagnostic: isolate uncertainty, routing, and detector controls",
            ],
            cwd=local_root,
        )
    run(["git", "push", "-u", "origin", "main"], cwd=local_root)


def main() -> None:
    global REMOTE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    REMOTE_ROOT = args.remote_root

    local_root = Path.cwd().resolve()
    verify_local_revision(local_root)
    zip_path = locate_zip(local_root)
    actual = sha256_file(zip_path)
    if actual != EXPECTED_SHA256:
        raise SystemExit(
            f"ZIP SHA-256 mismatch\nexpected={EXPECTED_SHA256}\nactual={actual}"
        )
    print(f"PACKAGE_SHA256_PASS: {actual}")

    with tempfile.TemporaryDirectory(prefix="brx_diag_patch_") as temporary:
        temporary_path = Path(temporary)
        safe_extract(zip_path, temporary_path)
        patch_root = temporary_path / PATCH_ROOT_NAME
        if not patch_root.is_dir():
            raise RuntimeError(f"Missing patch root: {PATCH_ROOT_NAME}")
        checked = verify_manifest(patch_root)
        print(f"DIAGNOSTIC_MANIFEST_PASS: {checked} files")
        copy_patch(patch_root, local_root)

    run(
        [
            "python3",
            "-m",
            "compileall",
            "-q",
            "scripts/gate0_mechanism_diagnostic.py",
            "RUN_03B_SUBMIT_GATE0_MECHANISM_DIAGNOSTIC.py",
            "RUN_06_REMOTE_STATUS.py",
        ],
        cwd=local_root,
    )
    commit_push(local_root)

    control_path = str(Path.home() / ".ssh" / "cm-revised-twc-rorqual")
    rsync_command = [
        "rsync",
        "-av",
        "--delete",
        "--exclude", ".git/",
        "--exclude", ".venv/",
        "--exclude", "results/",
        "--exclude", "outputs/",
        "--exclude", "*.zip",
        "--exclude", "__pycache__/",
        "--exclude", "*.pyc",
        "-e",
        f"ssh -o ControlMaster=auto -o ControlPersist=600 -o ControlPath={control_path}",
        str(local_root) + "/",
        f"{REMOTE}:{REMOTE_ROOT}/",
    ]
    run(rsync_command)

    remote_script = f"""set -euo pipefail
cd {REMOTE_ROOT}
source .venv/bin/activate
export PYTHONPATH=\"$PWD/src:${{PYTHONPATH:-}}\"
python -m compileall -q scripts/gate0_mechanism_diagnostic.py RUN_03B_SUBMIT_GATE0_MECHANISM_DIAGNOSTIC.py
python - <<'REMOTE_PY'
import json
from pathlib import Path
import yaml
revision = json.loads(Path('PACKAGE_REVISION.json').read_text())
assert revision.get('revision') == '{EXPECTED_PACKAGE_REVISION}', revision
required = [
    Path('outputs/checkpoints/initial/best.pt'),
    Path('outputs/reports/initial_train_summary.json'),
    Path('outputs/reports/initial_eval_summary.json'),
    Path('configs/gate0_mechanism_diagnostic.yaml'),
    Path('scripts/gate0_mechanism_diagnostic.py'),
]
for path in required:
    assert path.exists(), path
train = json.loads(Path('outputs/reports/initial_train_summary.json').read_text())
evaluation = json.loads(Path('outputs/reports/initial_eval_summary.json').read_text())
assert train.get('complete') is True and int(train.get('steps', 0)) == 1200, train
assert evaluation.get('complete') is True and int(evaluation.get('rows', 0)) == 960, evaluation
config = yaml.safe_load(Path('configs/gate0_mechanism_diagnostic.yaml').read_text())
assert config.get('package_revision') == '{EXPECTED_PACKAGE_REVISION}', config
print('REMOTE_GATE0_DIAGNOSTIC_PREFLIGHT_PASS')
print('DIAGNOSTIC_VARIANTS', len(config.get('required_variants', [])))
REMOTE_PY
"""
    run(
        [
            "ssh",
            "-o", "ControlMaster=auto",
            "-o", "ControlPersist=600",
            "-o", f"ControlPath={control_path}",
            REMOTE,
            remote_script,
        ]
    )
    print("GATE0_MECHANISM_DIAGNOSTIC_V1_DEPLOYED")
    print("Next: python3 RUN_03B_SUBMIT_GATE0_MECHANISM_DIAGNOSTIC.py")


if __name__ == "__main__":
    main()
