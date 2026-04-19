# path: src/00_bootstrap.py
"""
Bootstrap project paths and reproducibility artifacts.

Writes reports/run_config.json (paths + expected input names), reports/env.txt, and a frozen requirements list.
Run this once before the analysis steps to make every script resolve paths deterministically.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def _run(cmd: list[str]) -> tuple[int, str]:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + (p.stderr or "")
    return p.returncode, out.strip()


def _pip_version(py: str) -> str:
    code, out = _run([py, "-m", "pip", "--version"])
    return out if code == 0 and out else "pip_version_unknown"


def _pip_freeze_sorted(py: str) -> list[str]:
    code, out = _run([py, "-m", "pip", "freeze"])
    if code != 0:
        raise RuntimeError(out or "pip freeze failed")
    lines = [ln.strip() for ln in out.splitlines() if ln.strip()]
    lines.sort(key=lambda s: s.lower())
    return lines


def _normalize_abs(p: str) -> str:
    return str(Path(p).expanduser().resolve())


def _guess_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if cwd.name == "src" and (cwd.parent / "src").exists():
        return cwd.parent
    if (cwd / "src").exists():
        return cwd
    if (cwd.parent / "src").exists():
        return cwd.parent
    return cwd


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default="")
    ap.add_argument("--seed", type=int, default=2026)

    # Optional overrides for non-default layouts (not required for normal runs)
    ap.add_argument("--zip-path", default="")
    ap.add_argument("--data-raw", default="")

    args = ap.parse_args(argv)

    project_root = Path(_normalize_abs(args.project_root)) if args.project_root else _guess_project_root()

    zip_path = Path(_normalize_abs(args.zip_path)) if args.zip_path else None
    data_raw_override = Path(_normalize_abs(args.data_raw)) if args.data_raw else None

    dirs = {
        "project_root": str(project_root),
        "zip_path": str(zip_path) if zip_path else "",
        "data_raw_dir": str((data_raw_override or (project_root / "data_raw")).resolve()),
        "data_work_dir": str((project_root / "data_work").resolve()),
        "out_dir": str((project_root / "out").resolve()),
        "reports_dir": str((project_root / "reports").resolve()),
        "src_dir": str((project_root / "src").resolve()),
    }

    for k in ["data_raw_dir", "data_work_dir", "out_dir", "reports_dir", "src_dir"]:
        Path(dirs[k]).mkdir(parents=True, exist_ok=True)

    py = sys.executable
    freeze_lines = _pip_freeze_sorted(py)
    requirements_lock_path = Path(dirs["reports_dir"]) / "requirements_lock.txt"
    _write_text(requirements_lock_path, "\n".join(freeze_lines) + "\n")

    env_lines = [
        f"os={platform.system()}",
        f"os_release={platform.release()}",
        f"platform={platform.platform()}",
        f"python_executable={py}",
        f"python_version={platform.python_version()}",
        f"pip_version={_pip_version(py)}",
        f"project_root={dirs['project_root']}",
        f"data_raw_dir={dirs['data_raw_dir']}",
        f"zip_path={dirs['zip_path']}" if dirs["zip_path"] else "zip_path=(not used)",
        f"requirements_lock={str(requirements_lock_path.resolve())}",
    ]
    env_path = Path(dirs["reports_dir"]) / "env.txt"
    _write_text(env_path, "\n".join(env_lines) + "\n")

    run_config = {
        "seed": int(args.seed),
        "paths": dirs,
        "expected_files": {
            "whl_2025_csv": "whl_2025.csv",
            "whl_2025_xlsx": "whl_2025.xlsx",
            "matchups_xlsx": "WHSDSC_Rnd1_matchups.xlsx",
            "data_dictionary_xlsx": "WHSDSC_2026_DataDictionary.xlsx",
            "glossary_docx": "WHSDSC 2026 Glossary.docx",
            "workbook_pdf": "WHSDSC 2026 Workbook - Hockey.pdf",
        },
        "output_contract": {
            "phase1a_power_rankings_csv": "out/power_rankings.csv",
            "phase1a_round1_probs_csv": "out/round1_homewin_probs.csv",
            "phase1b_top10_csv": "out/line_disparity_top10.csv",
            "phase1c_png_glob": "out/phase1c_*.png",
            "phase1d_text": "reports/phase1d_answers.txt",
        },
    }
    run_config_path = Path(dirs["reports_dir"]) / "run_config.json"
    _write_text(run_config_path, json.dumps(run_config, ensure_ascii=False, sort_keys=True, indent=2) + "\n")

    print("BOOTSTRAP_OK")
    print(f"env_txt={env_path.resolve()}")
    print(f"run_config_json={run_config_path.resolve()}")
    print(f"requirements_lock={requirements_lock_path.resolve()}")
    print(f"data_raw_dir={dirs['data_raw_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
