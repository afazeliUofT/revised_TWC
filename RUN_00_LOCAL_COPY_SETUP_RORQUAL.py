from __future__ import annotations
import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"
GITHUB_REMOTE = "git@github.com:afazeliUofT/revised_TWC.git"
PACKAGE_NAME = "BayesRoute_TWC_Gate0_v2_Repair.zip"
DOWNLOADS = Path("/mnt/c/Users/alifa/Downloads")


def run(cmd: list[str], cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    proc = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True)
    if check and proc.returncode != 0:
        raise SystemExit(proc.returncode)
    return proc


def ssh(command: str, check: bool = True) -> subprocess.CompletedProcess:
    return run(["ssh", REMOTE, command], check=check)


def find_package() -> Path:
    for path in [Path.cwd() / PACKAGE_NAME, DOWNLOADS / PACKAGE_NAME]:
        if path.exists():
            return path
    raise SystemExit(f"Missing {PACKAGE_NAME} in the working directory or {DOWNLOADS}")


def ensure_repo(root: Path) -> None:
    if not (root / ".git").exists():
        run(["git", "init"], cwd=root)
        run(["git", "branch", "-M", "main"], cwd=root)
        run(["git", "remote", "add", "origin", GITHUB_REMOTE], cwd=root)
    else:
        run(["git", "remote", "set-url", "origin", GITHUB_REMOTE], cwd=root)


def commit_push(root: Path) -> None:
    run(["git", "add", "-A"], cwd=root)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=root
    ).returncode != 0
    if changed:
        run([
            "git", "commit", "-m",
            "Gate-0 v2 repair: activate coupling and strengthen health gate",
        ], cwd=root)
    run(["git", "push", "-u", "origin", "main"], cwd=root)


def main() -> None:
    global REMOTE, REMOTE_ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--remote", default=REMOTE)
    ap.add_argument("--remote-root", default=REMOTE_ROOT)
    args = ap.parse_args()
    REMOTE = args.remote
    REMOTE_ROOT = args.remote_root

    local_root = Path.cwd().resolve()
    package = find_package()
    stage = local_root / "_gate0_v2_stage"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir()
    with zipfile.ZipFile(package, "r") as archive:
        archive.extractall(stage)
    extracted = stage / "bayesroute_rx_twc"
    if not extracted.is_dir():
        raise SystemExit("Unexpected repair ZIP layout")

    # Replace only controlled source/configuration files. Preserve .git and the ZIP.
    controlled_dirs = ["configs", "docs", "scripts", "slurm", "src", "wrappers"]
    controlled_files = [
        ".gitignore", "README.md", "MANIFEST.sha256", "PACKAGE_REVISION.json",
        "pyproject.toml", "requirements_light.txt", "requirements_rorqual.txt",
    ]
    controlled_files += [p.name for p in local_root.glob("RUN_*.py")]
    for name in controlled_dirs + controlled_files:
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
    shutil.rmtree(stage)

    # Remove stale evidence from the superseded implementation. Git history retains it.
    shutil.rmtree(local_root / "results" / "from_rorqual", ignore_errors=True)
    (local_root / "results" / "from_rorqual").mkdir(parents=True, exist_ok=True)
    (local_root / "results" / "from_rorqual" / "README.md").write_text(
        "Awaiting Gate-0 v2 Rorqual evidence.\n", encoding="utf-8"
    )
    shutil.rmtree(local_root / "outputs", ignore_errors=True)
    (local_root / "outputs").mkdir(exist_ok=True)
    (local_root / "outputs" / ".keep").write_text("", encoding="utf-8")
    (local_root / "empty_file.txt").unlink(missing_ok=True)

    ensure_repo(local_root)
    commit_push(local_root)

    ssh(f"mkdir -p {REMOTE_ROOT}")
    run([
        "rsync", "-av", "--delete",
        "--exclude", ".git/",
        "--exclude", ".venv/",
        "--exclude", "results/",
        "--exclude", "outputs/",
        "--exclude", "*.zip",
        str(local_root) + "/",
        f"{REMOTE}:{REMOTE_ROOT}/",
    ])
    ssh(
        f"cd {REMOTE_ROOT} && "
        "rm -rf outputs/smoke outputs/optuna outputs/checkpoints outputs/logs "
        "outputs/eval outputs/reports outputs/plots outputs/github_artifacts outputs/slurm "
        "outputs/gates && mkdir -p outputs/setup && "
        "python3 scripts/rorqual_setup_env.py --venv .venv --out outputs/setup"
    )
    print("GATE0_V2_REPAIR_DEPLOYED")
    print("Next: python3 RUN_01_SUBMIT_SMOKE_TEST.py")


if __name__ == "__main__":
    main()
