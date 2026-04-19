"""
Line disparity (first_off vs second_off).

Computes team-level xG/60 for top-2 forward lines and reports the ratio first_off / second_off.
Writes out/line_disparity.csv and out/line_disparity_top10.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REQ_COLS = [
    "game_id",
    "home_team",
    "away_team",
    "home_off_line",
    "away_off_line",
    "home_xg",
    "away_xg",
    "toi",
]


def _resolve_project_root() -> Path:
    cwd = Path.cwd().resolve()
    if (cwd / "reports" / "run_config.json").exists():
        return cwd
    if (cwd.parent / "reports" / "run_config.json").exists():
        return cwd.parent
    raise FileNotFoundError("Could not locate reports/run_config.json. Run from PROJECT_ROOT or PROJECT_ROOT/src.")


def _read_run_config(project_root: Path) -> dict:
    p = project_root / "reports" / "run_config.json"
    if not p.exists():
        raise FileNotFoundError(f"run_config.json not found: {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _get_paths(project_root: Path) -> dict:
    rc = _read_run_config(project_root)
    paths = rc.get("paths", {})
    data_raw_dir = Path(paths.get("data_raw_dir", project_root / "data_raw")).resolve()
    out_dir = Path(paths.get("out_dir", project_root / "out")).resolve()
    reports_dir = Path(paths.get("reports_dir", project_root / "reports")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    expected = rc.get("expected_files", {})
    whl_csv_name = expected.get("whl_2025_csv", "whl_2025.csv")
    whl_csv_path = data_raw_dir / whl_csv_name
    if not whl_csv_path.exists():
        alt = data_raw_dir / "whl_2025.csv"
        if alt.exists():
            whl_csv_path = alt
        else:
            raise FileNotFoundError(f"whl_2025.csv not found in data_raw_dir: {data_raw_dir}")

    return {"whl_csv_path": whl_csv_path, "out_dir": out_dir, "reports_dir": reports_dir}


def _safe_div(n: float, d: float) -> float:
    if d is None or d == 0 or np.isnan(d):
        return np.nan
    return n / d


def main() -> int:
    project_root = _resolve_project_root()
    p = _get_paths(project_root)

    whl_csv_path: Path = p["whl_csv_path"]
    out_dir: Path = p["out_dir"]
    reports_dir: Path = p["reports_dir"]

    df = pd.read_csv(whl_csv_path, usecols=REQ_COLS)

    df["home_xg"] = pd.to_numeric(df["home_xg"], errors="coerce")
    df["away_xg"] = pd.to_numeric(df["away_xg"], errors="coerce")
    df["toi"] = pd.to_numeric(df["toi"], errors="coerce")

    # Build "team-line" long table: (team, off_line, xg, toi)
    home = pd.DataFrame(
        {
            "team": df["home_team"].astype(str),
            "off_line": df["home_off_line"].astype(str),
            "xg": df["home_xg"],
            "toi": df["toi"],
        }
    )
    away = pd.DataFrame(
        {
            "team": df["away_team"].astype(str),
            "off_line": df["away_off_line"].astype(str),
            "xg": df["away_xg"],
            "toi": df["toi"],
        }
    )
    long = pd.concat([home, away], ignore_index=True)

    # Teams list (expected 32)
    teams = sorted(pd.unique(long["team"]).tolist())

    agg = (
        long.groupby(["team", "off_line"], as_index=False)
        .agg(xg_sum=("xg", "sum"), toi_sum=("toi", "sum"))
        .copy()
    )

    agg = agg[agg["off_line"].isin(["first_off", "second_off"])].copy()

    # Pivot to wide
    wide = agg.pivot(index="team", columns="off_line", values=["xg_sum", "toi_sum"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns.to_flat_index()]
    wide = wide.reset_index()

    # Ensure all teams present (even if missing one line)
    wide = pd.DataFrame({"team": teams}).merge(wide, on="team", how="left")

    # Fill missing sums with 0, but keep TOI missing as 0 too (so xG60 becomes NaN due to division check)
    for c in ["xg_sum_first_off", "xg_sum_second_off", "toi_sum_first_off", "toi_sum_second_off"]:
        if c not in wide.columns:
            wide[c] = 0.0
    wide[["xg_sum_first_off", "xg_sum_second_off"]] = wide[
        ["xg_sum_first_off", "xg_sum_second_off"]
    ].fillna(0.0)
    wide[["toi_sum_first_off", "toi_sum_second_off"]] = wide[
        ["toi_sum_first_off", "toi_sum_second_off"]
    ].fillna(0.0)

    # Compute xG/60
    wide["first_off_xg60"] = wide.apply(
        lambda r: _safe_div(r["xg_sum_first_off"], r["toi_sum_first_off"]) * 3600
        if r["toi_sum_first_off"] > 0
        else np.nan,
        axis=1,
    )
    wide["second_off_xg60"] = wide.apply(
        lambda r: _safe_div(r["xg_sum_second_off"], r["toi_sum_second_off"]) * 3600
        if r["toi_sum_second_off"] > 0
        else np.nan,
        axis=1,
    )
    wide["disparity_ratio"] = wide.apply(
        lambda r: _safe_div(r["first_off_xg60"], r["second_off_xg60"]),
        axis=1,
    )

    out = wide.rename(
        columns={
            "xg_sum_first_off": "first_off_xg",
            "toi_sum_first_off": "first_off_toi",
            "xg_sum_second_off": "second_off_xg",
            "toi_sum_second_off": "second_off_toi",
        }
    )[
        [
            "team",
            "first_off_toi",
            "first_off_xg",
            "first_off_xg60",
            "second_off_toi",
            "second_off_xg",
            "second_off_xg60",
            "disparity_ratio",
        ]
    ].copy()

    all_path = out_dir / "line_disparity.csv"
    out.to_csv(all_path, index=False)

    # Top10 (drop NaN ratios)
    top10 = (
        out.dropna(subset=["disparity_ratio"])
        .sort_values("disparity_ratio", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    top10_path = out_dir / "line_disparity_top10.csv"
    top10.to_csv(top10_path, index=False)

    n_teams = int(out["team"].nunique())
    nan_ratio_teams = int(out["disparity_ratio"].isna().sum())

    print("=== STEP07 LINE DISPARITY (first_off vs second_off) ===")
    print(f"input:  {whl_csv_path.resolve()}")
    print(f"output: {all_path.resolve()}")
    print(f"output: {top10_path.resolve()}")
    print(f"teams in output: {n_teams} (expected 32)")
    print(f"teams with NaN ratio (e.g., missing TOI): {nan_ratio_teams}")

    print("\n--- TOP 10 by disparity_ratio ---")
    show = top10[["team", "disparity_ratio", "first_off_xg60", "second_off_xg60", "first_off_toi", "second_off_toi"]]
    with pd.option_context("display.width", 160):
        print(show.to_string(index=False))

    # DoD-ish check: must have 32 teams & top10 exactly 10 rows (unless NaNs reduce it)
    ok_teams = (n_teams == 32)
    ok_top10 = (len(top10) == 10)
    overall = ok_teams and ok_top10
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")

    # If top10 not 10 due to NaNs, still return non-zero to force attention
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
