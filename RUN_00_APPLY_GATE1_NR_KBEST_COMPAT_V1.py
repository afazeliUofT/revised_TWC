#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

ZIP_NAME = "BayesRoute_TWC_Gate1_NR_KBest_Compat_v1.zip"
EXPECTED_SHA256 = "2ebf7de4887aaa285e34831be4611e77264d5efac86b6252a3266a06004bdb28"
PATCH_ROOT_NAME = "BayesRoute_TWC_Gate1_NR_KBest_Compat_v1"
EXPECTED_GATE1_REVISION = "gate1_nr_integration_v1"
EXPECTED_BASE_REVISION = "gate0_v2_4_20260809"
COMPAT_VERSION = "sionna_2_0_1_list2llr_eager_exact_v1"

REMOTE = "rsadve1@rorqual.alliancecan.ca"
REMOTE_ROOT = "/home/rsadve1/links/scratch/revised_TWC"
GITHUB_REMOTE = "git@github.com:afazeliUofT/revised_TWC.git"
DOWNLOADS = Path("/mnt/c/Users/alifa/Downloads")


def run(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(str(x) for x in cmd), flush=True)
    result = subprocess.run([str(x) for x in cmd], cwd=cwd, text=True)
    if check and result.returncode != 0:
        raise SystemExit(result.returncode)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_zip(root: Path) -> Path:
    for candidate in (DOWNLOADS / ZIP_NAME, root / ZIP_NAME):
        if candidate.is_file():
            return candidate
    raise SystemExit(f"Missing {ZIP_NAME} in {DOWNLOADS} or {root}")


def safe_extract(zip_path: Path, destination: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as archive:
        base = destination.resolve()
        for item in archive.infolist():
            target = (destination / item.filename).resolve()
            if target != base and base not in target.parents:
                raise RuntimeError(f"Unsafe ZIP path: {item.filename}")
        archive.extractall(destination)


def verify_patch_manifest(patch_root: Path) -> int:
    manifest = patch_root / "PATCH_MANIFEST.sha256"
    if not manifest.is_file():
        raise RuntimeError("PATCH_MANIFEST.sha256 is missing")
    count = 0
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(None, 1)
        relative = relative.strip().lstrip("*")
        path = patch_root / relative
        if not path.is_file():
            raise RuntimeError(f"Missing patch file: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"Patch manifest mismatch: {relative}\nexpected={expected}\nactual={actual}"
            )
        count += 1
    if count != 2:
        raise RuntimeError(f"Expected 2 patch files, verified {count}")
    return count


def replace_once(path: Path, old: str, new: str, marker: str) -> None:
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"Could not apply deterministic patch to {path}: expected one match, found {count}"
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def patch_nr_gate1(path: Path) -> None:
    old = '''def standard_receiver(
    context: NRContext,
    *,
    perfect_csi: bool = False,
    kbest_k: int | None = None,
    return_crc: bool = True,
) -> Any:
    from sionna.phy.mimo import StreamManagement
    from sionna.phy.nr import PUSCHReceiver
    from sionna.phy.ofdm import KBestDetector

    detector = None
    stream_management = None
    if kbest_k is not None:
        rx_tx_association = np.ones([1, context.case.num_users], dtype=bool)
        stream_management = StreamManagement(
            rx_tx_association, int(context.case.num_layers_per_user)
        )
        detector = KBestDetector(
            output="bit",
            num_streams=int(context.case.num_streams),
            k=int(kbest_k),
            resource_grid=context.transmitter.resource_grid,
            stream_management=stream_management,
            constellation_type="qam",
            num_bits_per_symbol=context.transmitter._num_bits_per_symbol,
            device=str(context.device),
        )
    return PUSCHReceiver(
        context.transmitter,
        channel_estimator="perfect" if perfect_csi else None,
        mimo_detector=detector,
        return_tb_crc_status=bool(return_crc),
        stream_management=stream_management,
        input_domain="freq",
        device=str(context.device),
    )
'''
    new = '''def standard_receiver(
    context: NRContext,
    *,
    perfect_csi: bool = False,
    kbest_k: int | None = None,
    return_crc: bool = True,
) -> Any:
    from sionna.phy.mimo import StreamManagement
    from sionna.phy.nr import PUSCHReceiver
    from sionna.phy.ofdm import KBestDetector

    detector = None
    stream_management = None
    kbest_compatibility = None
    if kbest_k is not None:
        # BayesRoute Gate-1 compatibility patch:
        # Sionna 2.0.1 compiles only an equal+any helper in List2LLRSimple.
        # Alliance's CUDA PyTorch build has no Triton, so use the exact eager
        # expression. K-best candidates, distances, LLR equations, and outputs
        # are unchanged.
        from .sionna_kbest_compat import configure_sionna_kbest_compat

        kbest_compatibility = configure_sionna_kbest_compat(force_eager=True)
        if not kbest_compatibility.get("passed", False):
            raise RuntimeError(
                f"Sionna K-best compatibility self-test failed: {kbest_compatibility}"
            )

        rx_tx_association = np.ones([1, context.case.num_users], dtype=bool)
        stream_management = StreamManagement(
            rx_tx_association, int(context.case.num_layers_per_user)
        )
        detector = KBestDetector(
            output="bit",
            num_streams=int(context.case.num_streams),
            k=int(kbest_k),
            resource_grid=context.transmitter.resource_grid,
            stream_management=stream_management,
            constellation_type="qam",
            num_bits_per_symbol=context.transmitter._num_bits_per_symbol,
            device=str(context.device),
        )
    receiver = PUSCHReceiver(
        context.transmitter,
        channel_estimator="perfect" if perfect_csi else None,
        mimo_detector=detector,
        return_tb_crc_status=bool(return_crc),
        stream_management=stream_management,
        input_domain="freq",
        device=str(context.device),
    )
    if kbest_compatibility is not None:
        receiver._bayesroute_kbest_compatibility = kbest_compatibility
    return receiver
'''
    replace_once(
        path,
        old,
        new,
        marker="_bayesroute_kbest_compatibility",
    )


def patch_gate1_smoke(path: Path) -> None:
    old = '''def kbest_check(
    case: NRCase,
    device: torch.device,
    batch_size: int,
    ebno_db: float,
    k: int,
) -> dict[str, Any]:
    context = build_nr_context(case, device)
    batch = context.sample(batch_size=batch_size, ebno_db=ebno_db)
    receiver = standard_receiver(
        context, perfect_csi=True, kbest_k=int(k), return_crc=True
    )
    with torch.no_grad():
        metrics = run_standard_receiver(
            receiver, batch, batch.information_bits, perfect_csi=True
        )
    valid = all(
        0.0 <= float(metrics[key]) <= 1.0
        for key in ("information_ber", "tbler", "crc_failure_rate")
    )
    return {
        "passed": bool(valid),
        "metrics": metrics,
        "case": case.__dict__,
        "num_streams": case.num_streams,
        "k": int(k),
    }
'''
    new = '''def kbest_check(
    case: NRCase,
    device: torch.device,
    batch_size: int,
    ebno_db: float,
    k: int,
) -> dict[str, Any]:
    context = build_nr_context(case, device)
    batch = context.sample(batch_size=batch_size, ebno_db=ebno_db)
    receiver = standard_receiver(
        context, perfect_csi=True, kbest_k=int(k), return_crc=True
    )
    compatibility = dict(
        getattr(receiver, "_bayesroute_kbest_compatibility", {})
    )
    with torch.no_grad():
        metrics = run_standard_receiver(
            receiver, batch, batch.information_bits, perfect_csi=True
        )
    metric_valid = all(
        0.0 <= float(metrics[key]) <= 1.0
        for key in ("information_ber", "tbler", "crc_failure_rate")
    )
    compatibility_valid = bool(
        compatibility.get("passed")
        and compatibility.get("backend") == "eager_exact"
        and compatibility.get("active_semantics_exact")
        and compatibility.get("installed_sionna_files_modified") is False
    )
    return {
        "passed": bool(metric_valid and compatibility_valid),
        "metrics_valid": bool(metric_valid),
        "compatibility_valid": compatibility_valid,
        "compatibility": compatibility,
        "metrics": metrics,
        "case": case.__dict__,
        "num_streams": case.num_streams,
        "k": int(k),
    }
'''
    replace_once(
        path,
        old,
        new,
        marker='"compatibility_valid": compatibility_valid',
    )

    old_check = '        "kbest_path": bool(kbest["passed"]),\n'
    new_check = (
        '        "kbest_eager_exact_compatibility": bool(\n'
        '            kbest.get("compatibility_valid", False)\n'
        '        ),\n'
        '        "kbest_path": bool(kbest["passed"]),\n'
    )
    replace_once(
        path,
        old_check,
        new_check,
        marker='"kbest_eager_exact_compatibility"',
    )


def patch_submit_wrapper(path: Path) -> None:
    old = '''gate0 = Path('outputs/gates/GATE0_MECHANISM_DIAGNOSTIC.txt').read_text()
assert 'CLASSIFICATION: GATE0_MECHANISM_SUPPORTED' in gate0
print('GATE1_NR_MANIFEST_PREFLIGHT_PASS', checked)
print('PGCA_AGMP_BASELINE_INCLUDED NO')
'''
    new = '''from bayesroute.sionna_kbest_compat import configure_sionna_kbest_compat
compat = configure_sionna_kbest_compat(force_eager=True)
assert compat.get('passed') is True, compat
assert compat.get('backend') == 'eager_exact', compat
assert compat.get('active_semantics_exact') is True, compat
gate0 = Path('outputs/gates/GATE0_MECHANISM_DIAGNOSTIC.txt').read_text()
assert 'CLASSIFICATION: GATE0_MECHANISM_SUPPORTED' in gate0
print('GATE1_NR_MANIFEST_PREFLIGHT_PASS', checked)
print('GATE1_NR_KBEST_COMPAT_PREFLIGHT_PASS', compat['compat_version'], compat['backend'])
print('PGCA_AGMP_BASELINE_INCLUDED NO')
'''
    replace_once(
        path,
        old,
        new,
        marker="GATE1_NR_KBEST_COMPAT_PREFLIGHT_PASS",
    )


def update_revision(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("revision") != EXPECTED_GATE1_REVISION:
        raise RuntimeError(f"Unexpected Gate-1 revision: {data}")
    data["compatibility_patch"] = COMPAT_VERSION
    data["kbest_list2llr_backend"] = "eager_exact"
    data["triton_required_for_gate1"] = False
    data["installed_sionna_files_modified"] = False
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def append_readme(path: Path) -> None:
    marker = "## Sionna K-best compatibility"
    text = path.read_text(encoding="utf-8")
    if marker in text:
        return
    text += (
        "\n## Sionna K-best compatibility\n\n"
        "Sionna 2.0.1 uses `torch.compile` only to fuse an equal-and-reduce helper in\n"
        "its list-to-LLR conversion. On Alliance builds without Triton, Gate-1 replaces\n"
        "that helper at process scope by the exact eager expression\n"
        "`(path_inds == c).any(dim=-2)`. No installed Sionna file is edited and the\n"
        "K-best algorithm, candidate list, distance metric, and LLR definition are\n"
        "unchanged. Gate reports record the active backend explicitly.\n"
    )
    path.write_text(text, encoding="utf-8")


def regenerate_gate1_manifest(root: Path) -> int:
    manifest = root / "GATE1_NR_MANIFEST.sha256"
    existing: list[str] = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        _, relative = raw.split(None, 1)
        relative = relative.strip().lstrip("*")
        if relative not in existing:
            existing.append(relative)
    new_module = "src/bayesroute/sionna_kbest_compat.py"
    if new_module not in existing:
        insert_at = existing.index("src/bayesroute/nr_gate1.py") + 1
        existing.insert(insert_at, new_module)
    lines: list[str] = []
    for relative in existing:
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"Cannot regenerate manifest; missing {relative}")
        lines.append(f"{sha256_file(path)}  {relative}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(existing)


def verify_local_contract(root: Path) -> None:
    base = json.loads((root / "PACKAGE_REVISION.json").read_text(encoding="utf-8"))
    gate1 = json.loads((root / "GATE1_NR_REVISION.json").read_text(encoding="utf-8"))
    if base.get("revision") != EXPECTED_BASE_REVISION:
        raise SystemExit(f"Wrong base revision: {base}")
    if gate1.get("revision") != EXPECTED_GATE1_REVISION:
        raise SystemExit(f"Wrong Gate-1 revision: {gate1}")
    for path in (
        root / "src/bayesroute/nr_gate1.py",
        root / "scripts/gate1_nr_smoke.py",
        root / "RUN_07_SUBMIT_GATE1_NR_SMOKE.py",
        root / "wrappers/RUN_07_SUBMIT_GATE1_NR_SMOKE.py",
    ):
        if not path.is_file():
            raise SystemExit(f"Missing Gate-1 source file: {path}")


def clean_failed_evidence_local(root: Path) -> None:
    for pattern in (
        "results/from_rorqual/gates/GATE1_NR_SMOKE.*",
        "results/from_rorqual/slurm/brx_nr_smoke-*.out",
        "results/from_rorqual/slurm/brx_nr_smoke-*.err",
    ):
        for path in root.glob(pattern):
            if path.is_file():
                path.unlink()


def commit_and_push(root: Path) -> None:
    run(["git", "remote", "set-url", "origin", GITHUB_REMOTE], cwd=root)
    run(["git", "add", "-A"], cwd=root)
    changed = subprocess.run(
        ["git", "diff", "--cached", "--quiet"], cwd=root
    ).returncode != 0
    if changed:
        run(
            [
                "git",
                "commit",
                "-m",
                "Gate-1 NR: exact eager Sionna K-best compatibility without Triton",
            ],
            cwd=root,
        )
    run(["git", "push", "-u", "origin", "main"], cwd=root)


def deploy_remote(root: Path, remote_root: str) -> None:
    control_path = str(Path.home() / ".ssh" / "cm-revised-twc-rorqual")
    run(
        [
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
            str(root) + "/",
            f"{REMOTE}:{remote_root}/",
        ]
    )
    remote_script = f'''set -euo pipefail
cd {remote_root}
source .venv/bin/activate
export PYTHONPATH="$PWD/src:${{PYTHONPATH:-}}"
rm -f outputs/gates/GATE1_NR_SMOKE.json outputs/gates/GATE1_NR_SMOKE.txt
rm -f outputs/slurm/brx_nr_smoke-*.out outputs/slurm/brx_nr_smoke-*.err
python -m compileall -q src/bayesroute/nr_gate1.py src/bayesroute/sionna_kbest_compat.py scripts/gate1_nr_smoke.py RUN_07_SUBMIT_GATE1_NR_SMOKE.py
python - <<'REMOTE_PY'
from pathlib import Path
import hashlib, json
from bayesroute.sionna_kbest_compat import configure_sionna_kbest_compat
revision = json.loads(Path('GATE1_NR_REVISION.json').read_text())
assert revision.get('revision') == '{EXPECTED_GATE1_REVISION}', revision
assert revision.get('compatibility_patch') == '{COMPAT_VERSION}', revision
assert revision.get('kbest_list2llr_backend') == 'eager_exact', revision
assert revision.get('triton_required_for_gate1') is False, revision
checked = 0
for raw in Path('GATE1_NR_MANIFEST.sha256').read_text().splitlines():
    if not raw.strip():
        continue
    expected, relative = raw.split(None, 1)
    path = Path(relative.strip().lstrip('*'))
    assert path.is_file(), path
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path
    checked += 1
assert checked == 18, checked
report = configure_sionna_kbest_compat(force_eager=True)
assert report.get('passed') is True, report
assert report.get('backend') == 'eager_exact', report
assert report.get('active_semantics_exact') is True, report
assert report.get('installed_sionna_files_modified') is False, report
print('REMOTE_GATE1_NR_MANIFEST_PASS', checked)
print('REMOTE_GATE1_NR_KBEST_COMPAT_PASS', report['compat_version'], report['backend'])
print('TRITON_REQUIRED_FOR_GATE1', 'NO')
REMOTE_PY
'''
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


def main() -> None:
    global REMOTE_ROOT
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    args = parser.parse_args()
    REMOTE_ROOT = args.remote_root

    root = Path.cwd().resolve()
    verify_local_contract(root)

    zip_path = locate_zip(root)
    actual = sha256_file(zip_path)
    if actual != EXPECTED_SHA256:
        raise SystemExit(
            f"ZIP SHA-256 mismatch\nexpected={EXPECTED_SHA256}\nactual={actual}"
        )
    print("PACKAGE_SHA256_PASS:", actual)

    with tempfile.TemporaryDirectory(prefix="gate1_kbest_compat_") as temporary:
        temp = Path(temporary)
        safe_extract(zip_path, temp)
        patch_root = temp / PATCH_ROOT_NAME
        if not patch_root.is_dir():
            raise RuntimeError(f"Missing patch root {PATCH_ROOT_NAME}")
        count = verify_patch_manifest(patch_root)
        print("PATCH_MANIFEST_PASS:", count, "files")
        target = root / "src/bayesroute/sionna_kbest_compat.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            patch_root / "src/bayesroute/sionna_kbest_compat.py",
            target,
        )

    patch_nr_gate1(root / "src/bayesroute/nr_gate1.py")
    patch_gate1_smoke(root / "scripts/gate1_nr_smoke.py")
    patch_submit_wrapper(root / "RUN_07_SUBMIT_GATE1_NR_SMOKE.py")
    patch_submit_wrapper(root / "wrappers/RUN_07_SUBMIT_GATE1_NR_SMOKE.py")
    update_revision(root / "GATE1_NR_REVISION.json")
    append_readme(root / "README_GATE1_NR.md")
    manifest_count = regenerate_gate1_manifest(root)
    if manifest_count != 18:
        raise RuntimeError(f"Expected 18 Gate-1 manifest files, got {manifest_count}")
    print("GATE1_NR_MANIFEST_REGENERATED:", manifest_count, "files")

    run(
        [
            "python3",
            "-m",
            "compileall",
            "-q",
            "src/bayesroute/nr_gate1.py",
            "src/bayesroute/sionna_kbest_compat.py",
            "scripts/gate1_nr_smoke.py",
            "RUN_07_SUBMIT_GATE1_NR_SMOKE.py",
            "wrappers/RUN_07_SUBMIT_GATE1_NR_SMOKE.py",
        ],
        cwd=root,
    )
    clean_failed_evidence_local(root)
    commit_and_push(root)
    deploy_remote(root, REMOTE_ROOT)

    print("GATE1_NR_KBEST_COMPAT_V1_DEPLOYED")
    print("Next: python3 RUN_07_SUBMIT_GATE1_NR_SMOKE.py")


if __name__ == "__main__":
    main()
