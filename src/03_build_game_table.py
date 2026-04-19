"""
Build a game-level table from record-level rows.

Aggregates the raw dataset into one raw per game_id using a fixed contract (SUM/FIRST/MAX columns).
Writes data_work/games_2025.csv for downstream modeling steps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUM_COLS = [
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
    "home_shots",
    "away_shots",
    "home_assists",
    "away_assists",
    "home_penalties_committed",
    "home_penalty_minutes",
    "away_penalties_committed",
    "away_penalty_minutes",
    "toi",
]
FIRST_COLS = ["home_team", "away_team"]
MAX_COLS = ["went_ot"]

REQUIRED_COLS = ["game_id"] + FIRST_COLS + MAX_COLS + SUM_COLS


def _read_run_config(project_root: Path) -> dict:
    p = project_root / "reports" / "run_config.json"
    if not p.exists():
        raise FileNotFoundError(f"run_config.json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _resolve_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "reports" / "run_config.json").exists():
        return cwd
    if (cwd.parent / "reports" / "run_config.json").exists():
        return cwd.parent
    raise FileNotFoundError("Could not locate reports/run_config.json. Run from PROJECT_ROOT or PROJECT_ROOT/src.")


def _ensure_dirs(project_root: Path) -> dict:
    run_config = _read_run_config(project_root)
    paths = run_config.get("paths", {})
    data_raw_dir = Path(paths.get("data_raw_dir", project_root / "data_raw")).resolve()
    data_work_dir = Path(paths.get("data_work_dir", project_root / "data_work")).resolve()
    reports_dir = Path(paths.get("reports_dir", project_root / "reports")).resolve()

    data_work_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    expected = run_config.get("expected_files", {})
    whl_csv_name = expected.get("whl_2025_csv", "whl_2025.csv")
    whl_csv_path = data_raw_dir / whl_csv_name
    if not whl_csv_path.exists():
        alt = data_raw_dir / "whl_2025.csv"
        if alt.exists():
            whl_csv_path = alt
        else:
            raise FileNotFoundError(f"whl_2025.csv not found in data_raw_dir: {data_raw_dir}")

    return {
        "project_root": project_root,
        "data_raw_dir": data_raw_dir,
        "data_work_dir": data_work_dir,
        "reports_dir": reports_dir,
        "whl_csv_path": whl_csv_path,
    }


def _sum_with_nan_safe(s: pd.Series) -> float:
    # numeric coercion then sum; NaN ignored
    return pd.to_numeric(s, errors="coerce").sum()


def main() -> int:
    project_root = _resolve_project_root()
    ctx = _ensure_dirs(project_root)

    whl_csv_path: Path = ctx["whl_csv_path"]
    data_work_dir: Path = ctx["data_work_dir"]
    reports_dir: Path = ctx["reports_dir"]

    # Load only required columns for SSOT table
    df = pd.read_csv(whl_csv_path, usecols=REQUIRED_COLS)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in whl_2025.csv: {missing}")

    for c in SUM_COLS + MAX_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    agg_spec = {c: _sum_with_nan_safe for c in SUM_COLS}
    agg_spec.update({c: "first" for c in FIRST_COLS})
    agg_spec.update({c: "max" for c in MAX_COLS})

    games = df.groupby("game_id", as_index=False).agg(agg_spec)

    games["penalties_committed"] = (
        games["home_penalties_committed"].fillna(0) + games["away_penalties_committed"].fillna(0)
    )
    games["penalty_minutes"] = (
        games["home_penalty_minutes"].fillna(0) + games["away_penalty_minutes"].fillna(0)
    )

    # Ensure column order: identifiers -> first -> max -> sums
    ordered_cols = (
        ["game_id"] 
        + FIRST_COLS 
        + MAX_COLS 
        + [
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
            "home_shots",
            "away_shots",
            "home_assists",
            "away_assists",
            "home_penalties_committed",
            "away_penalties_committed",
            "home_penalty_minutes",
            "away_penalty_minutes",
            "penalties_committed",
            "penalty_minutes",
            "toi",
        ]
    )
    games = games[ordered_cols]

    rows = int(len(games))
    expected_rows = 1312  # from workbook + STEP02 sanity
    row_ok = (rows == expected_rows)

    # Aggregate sum checks: original totals vs aggregated totals
    checks = {}
    atol = 1e-6
    for c in SUM_COLS:
        orig_sum = float(pd.to_numeric(df[c], errors="coerce").sum())
        agg_sum = float(pd.to_numeric(games[c], errors="coerce").sum())
        abs_diff = float(abs(orig_sum - agg_sum))
        rel_diff = float(abs_diff / (abs(orig_sum) + 1e-12))
        ok = (abs_diff <= atol) or (rel_diff <= 1e-10)
        checks[c] = {
            "original_sum": orig_sum,
            "aggregated_sum": agg_sum,
            "abs_diff": abs_diff,
            "rel_diff": rel_diff,
            "ok": bool(ok),
        }

    out_games_path = data_work_dir / "games_2025.csv"
    games.to_csv(out_games_path, index=False)

    schema = {col: str(dtype) for col, dtype in games.dtypes.items()}
    schema_path = reports_dir / "games_2025_schema.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")

    agg_checks_path = reports_dir / "games_2025_aggregate_checks.json"
    agg_checks_path.write_text(json.dumps({"rows": rows, "row_ok": row_ok, "checks": checks},
                                          ensure_ascii=False, indent=2), encoding="utf-8")

    total_goals = (games["home_goals"].fillna(0) + games["away_goals"].fillna(0)).astype(float)
    goals_summary = {
        "home_goals": {
            "min": float(games["home_goals"].min()),
            "mean": float(games["home_goals"].mean()),
            "max": float(games["home_goals"].max()),
        },
        "away_goals": {
            "min": float(games["away_goals"].min()),
            "mean": float(games["away_goals"].mean()),
            "max": float(games["away_goals"].max()),
        },
        "total_goals": {
            "min": float(total_goals.min()),
            "mean": float(total_goals.mean()),
            "max": float(total_goals.max()),
        },
    }

    print("=== STEP03 BUILD GAME TABLE ===")
    print(f"whl_csv_path: {whl_csv_path.resolve()}")
    print(f"output: {out_games_path.resolve()}")
    print(f"rows in games_2025: {rows} (expected {expected_rows})")
    print(f"row_count_ok: {row_ok}")

    print("\n--- Aggregate checks (OK/FAIL) ---")
    # print only OK/FAIL per column + abs_diff
    for c in SUM_COLS:
        st = checks[c]
        print(f"{c}: {'OK' if st['ok'] else 'FAIL'} | abs_diff={st['abs_diff']:.6g}")

    print("\n--- Goals distribution (min/mean/max) ---")
    print(
        f"home_goals: {goals_summary['home_goals']['min']:.0f} / "
        f"{goals_summary['home_goals']['mean']:.3f} / {goals_summary['home_goals']['max']:.0f}"
    )
    print(
        f"away_goals: {goals_summary['away_goals']['min']:.0f} / "
        f"{goals_summary['away_goals']['mean']:.3f} / {goals_summary['away_goals']['max']:.0f}"
    )
    print(
        f"total_goals: {goals_summary['total_goals']['min']:.0f} / "
        f"{goals_summary['total_goals']['mean']:.3f} / {goals_summary['total_goals']['max']:.0f}"
    )

    # Exit non-zero if core validations fail
    all_agg_ok = all(v["ok"] for v in checks.values())
    overall_ok = row_ok and all_agg_ok
    print(f"\nOVERALL: {'PASS' if overall_ok else 'FAIL'}")

    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
