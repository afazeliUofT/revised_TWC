#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, platform, shutil, subprocess, sys, textwrap
from pathlib import Path


def run(cmd, check=True):
    print("+", " ".join(cmd), flush=True)
    p = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(p.stdout)
    if check and p.returncode != 0:
        raise SystemExit(p.returncode)
    return p


def pycheck(venv_py: Path, code: str):
    return run([str(venv_py), "-c", code], check=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", default=".venv")
    ap.add_argument("--out", default="outputs/setup")
    args = ap.parse_args()
    root = Path.cwd()
    out = root / args.out
    out.mkdir(parents=True, exist_ok=True)
    venv = root / args.venv

    py = shutil.which("python3") or shutil.which("python")
    if py is None:
        raise SystemExit("No python3 found on login node.")
    if not venv.exists():
        run([py, "-m", "venv", "--system-site-packages", str(venv)])
    venv_py = venv / "bin" / "python"
    run([str(venv_py), "-m", "pip", "install", "--upgrade", "pip", "wheel", "setuptools"], check=False)
    run([str(venv_py), "-m", "pip", "install", "-r", "requirements_light.txt"], check=True)

    # Try imports. Install Sionna only if missing. Torch is usually provided by system modules/site packages.
    torch_check = pycheck(venv_py, "import torch; print('TORCH_OK', torch.__version__, torch.cuda.is_available())")
    sionna_check = pycheck(venv_py, "import sionna; print('SIONNA_OK', getattr(sionna,'__version__','unknown'))")
    if sionna_check.returncode != 0:
        run([str(venv_py), "-m", "pip", "install", "sionna-no-rt==2.0.1"], check=False)
    report_code = r"""
import json, sys, platform
rep = {"python": sys.version, "platform": platform.platform()}
try:
    import torch
    rep["torch"] = torch.__version__
    rep["cuda_available"] = bool(torch.cuda.is_available())
    rep["cuda_device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available(): rep["cuda_name"] = torch.cuda.get_device_name(0)
except Exception as e:
    rep["torch_error"] = repr(e)
try:
    import sionna
    rep["sionna"] = getattr(sionna, "__version__", "unknown")
except Exception as e:
    rep["sionna_error"] = repr(e)
try:
    import optuna, pandas, matplotlib, yaml
    rep["light_packages"] = "ok"
except Exception as e:
    rep["light_packages_error"] = repr(e)
print(json.dumps(rep, indent=2))
assert "torch" in rep, "torch import failed"
assert "sionna" in rep, "sionna import failed"
assert rep.get("light_packages") == "ok", "light package import failed"
"""
    p = run([str(venv_py), "-c", report_code], check=False)
    (out / "venv_health_stdout.txt").write_text(p.stdout, encoding="utf-8")
    status = "PASS" if p.returncode == 0 else "FAIL"
    (out / "VENV_OK.txt").write_text(f"{status}: venv={venv}\n", encoding="utf-8")
    print(f"\nVENV_SETUP_{status}: {venv}")
    if p.returncode != 0:
        raise SystemExit(p.returncode)

if __name__ == "__main__":
    main()
