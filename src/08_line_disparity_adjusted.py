"""
Line disparity with opponent-defense adjustment.

Splits production by opponent defense pairing and re-weights minutes to account for matchup difficulty.
Writes adjusted tables under out/ and a baseline-vs-adjusted comparison under reports/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


# Defense-matchup adjustment parameters.
OFF_LINES = ["first_off", "second_off"]
DEF_PAIRS = ["first_def", "second_def"]

W_FIRST_DEF = 1.10
W_SECOND_DEF = 1.00

REQ_COLS = [
    "home_team",
    "away_team",
    "home_off_line",
    "away_off_line",
    "home_def_pairing",
    "away_def_pairing",
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

    baseline_candidates = [
        out_dir / "line_disparity.csv",
        out_dir / "line_disparity_all.csv",
        out_dir / "line_disparity_all_teams.csv",
    ]
    baseline_path = None
    for pth in baseline_candidates:
        if pth.exists():
            baseline_path = pth
            break

    return {
        "whl_csv_path": whl_csv_path,
        "out_dir": out_dir,
        "reports_dir": reports_dir,
        "baseline_path": baseline_path,
    }


def _safe_rate(xg_sum: float, toi_sum: float) -> float:
    if toi_sum is None or toi_sum <= 0 or np.isnan(toi_sum):
        return np.nan
    return (xg_sum / toi_sum) * 3600.0


def main() -> int:
    project_root = _resolve_project_root()
    paths = _get_paths(project_root)

    whl_csv_path: Path = paths["whl_csv_path"]
    out_dir: Path = paths["out_dir"]
    reports_dir: Path = paths["reports_dir"]
    baseline_path: Path | None = paths["baseline_path"]

    df = pd.read_csv(whl_csv_path, usecols=REQ_COLS)

    df["home_xg"] = pd.to_numeric(df["home_xg"], errors="coerce")
    df["away_xg"] = pd.to_numeric(df["away_xg"], errors="coerce")
    df["toi"] = pd.to_numeric(df["toi"], errors="coerce")

    home = pd.DataFrame(
        {
            "team": df["home_team"].astype(str),
            "off_line": df["home_off_line"].astype(str),
            "opp_def": df["away_def_pairing"].astype(str),
            "xg": df["home_xg"],
            "toi": df["toi"],
        }
    )
    away = pd.DataFrame(
        {
            "team": df["away_team"].astype(str),
            "off_line": df["away_off_line"].astype(str),
            "opp_def": df["home_def_pairing"].astype(str),
            "xg": df["away_xg"],
            "toi": df["toi"],
        }
    )

    long = pd.concat([home, away], ignore_index=True)

    long = long[long["off_line"].isin(OFF_LINES) & long["opp_def"].isin(DEF_PAIRS)].copy()

    teams = sorted(pd.unique(pd.concat([df["home_team"], df["away_team"]]).astype(str)).tolist())

    agg = (
        long.groupby(["team", "off_line", "opp_def"], as_index=False)
        .agg(xg_sum=("xg", "sum"), toi_sum=("toi", "sum"))
        .copy()
    )

    wide = agg.pivot(index=["team", "off_line"], columns="opp_def", values=["xg_sum", "toi_sum"])
    wide.columns = [f"{a}_{b}" for a, b in wide.columns.to_flat_index()]  # xg_sum_first_def etc.
    wide = wide.reset_index()

    base_grid = pd.MultiIndex.from_product([teams, OFF_LINES], names=["team", "off_line"]).to_frame(index=False)
    wide = base_grid.merge(wide, on=["team", "off_line"], how="left")

    for c in [
        "xg_sum_first_def",
        "toi_sum_first_def",
        "xg_sum_second_def",
        "toi_sum_second_def",
    ]:
        if c not in wide.columns:
            wide[c] = 0.0
    wide[["xg_sum_first_def", "toi_sum_first_def", "xg_sum_second_def", "toi_sum_second_def"]] = wide[
        ["xg_sum_first_def", "toi_sum_first_def", "xg_sum_second_def", "toi_sum_second_def"]
    ].fillna(0.0)

    wide["xg60_vs_first_def"] = wide.apply(lambda r: _safe_rate(r["xg_sum_first_def"], r["toi_sum_first_def"]), axis=1)
    wide["xg60_vs_second_def"] = wide.apply(
        lambda r: _safe_rate(r["xg_sum_second_def"], r["toi_sum_second_def"]), axis=1
    )

    # Weighted adjustment in (xg, toi) space (preferred):
    # adj_xg = w1*xg_first + w2*xg_second
    # adj_toi = w1*toi_first + w2*toi_second
    # adj_xg60 = adj_xg / adj_toi * 3600
    wide["adj_xg_sum"] = (W_FIRST_DEF * wide["xg_sum_first_def"]) + (W_SECOND_DEF * wide["xg_sum_second_def"])
    wide["adj_toi_sum"] = (W_FIRST_DEF * wide["toi_sum_first_def"]) + (W_SECOND_DEF * wide["toi_sum_second_def"])
    wide["xg60_adj"] = wide.apply(lambda r: _safe_rate(r["adj_xg_sum"], r["adj_toi_sum"]), axis=1)

    first = wide[wide["off_line"] == "first_off"].copy()
    second = wide[wide["off_line"] == "second_off"].copy()

    out = pd.DataFrame({"team": teams})

    def _add(prefix: str, src: pd.DataFrame) -> None:
        cols = [
            "team",
            "xg_sum_first_def",
            "toi_sum_first_def",
            "xg60_vs_first_def",
            "xg_sum_second_def",
            "toi_sum_second_def",
            "xg60_vs_second_def",
            "adj_xg_sum",
            "adj_toi_sum",
            "xg60_adj",
        ]
        tmp = src[cols].copy()
        tmp = tmp.rename(
            columns={
                "xg_sum_first_def": f"{prefix}_xg_vs_first_def",
                "toi_sum_first_def": f"{prefix}_toi_vs_first_def",
                "xg60_vs_first_def": f"{prefix}_xg60_vs_first_def",
                "xg_sum_second_def": f"{prefix}_xg_vs_second_def",
                "toi_sum_second_def": f"{prefix}_toi_vs_second_def",
                "xg60_vs_second_def": f"{prefix}_xg60_vs_second_def",
                "adj_xg_sum": f"{prefix}_adj_xg",
                "adj_toi_sum": f"{prefix}_adj_toi",
                "xg60_adj": f"{prefix}_xg60_adj",
            }
        )
        nonlocal out
        out = out.merge(tmp, on="team", how="left")

    _add("first_off", first)
    _add("second_off", second)

    out["disparity_ratio_adj"] = out["first_off_xg60_adj"] / out["second_off_xg60_adj"]
    out["disparity_ratio_adj"] = out["disparity_ratio_adj"].replace([np.inf, -np.inf], np.nan)

    adjusted_all_path = out_dir / "line_disparity_adjusted_all.csv"
    out.to_csv(adjusted_all_path, index=False)

    top10 = (
        out.dropna(subset=["disparity_ratio_adj"])
        .sort_values("disparity_ratio_adj", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    adjusted_top10_path = out_dir / "line_disparity_adjusted_top10.csv"
    top10.to_csv(adjusted_top10_path, index=False)

    overlap = None
    movers = None
    compare_path = reports_dir / "line_disparity_compare_baseline_vs_adjusted.csv"

    if baseline_path is not None:
        base = pd.read_csv(baseline_path)
        if "disparity_ratio" not in base.columns:
            if "ratio" in base.columns:
                base = base.rename(columns={"ratio": "disparity_ratio"})
            else:
                raise ValueError(f"Baseline file exists but missing disparity_ratio/ratio: {baseline_path}")

        base = base[["team", "disparity_ratio"]].copy()
        base["baseline_rank"] = base["disparity_ratio"].rank(ascending=False, method="min").astype(int)

        adj = out[["team", "disparity_ratio_adj"]].copy()
        adj["adjusted_rank"] = adj["disparity_ratio_adj"].rank(ascending=False, method="min").astype(int)

        cmp = base.merge(adj, on="team", how="inner")
        cmp["rank_delta"] = cmp["baseline_rank"] - cmp["adjusted_rank"]  # + means improved (moved up)
        cmp["abs_delta"] = cmp["rank_delta"].abs()

        base_top10 = set(base.sort_values("disparity_ratio", ascending=False).head(10)["team"].tolist())
        adj_top10 = set(adj.sort_values("disparity_ratio_adj", ascending=False).head(10)["team"].tolist())
        overlap = len(base_top10.intersection(adj_top10))

        movers = cmp.sort_values(["abs_delta", "team"], ascending=[False, True]).head(3).copy()

        cmp.to_csv(compare_path, index=False)
    else:
        pd.DataFrame({"note": ["baseline file not found; comparison skipped"]}).to_csv(compare_path, index=False)

    # Write a short rationale snippet for the report.
    rationale = (
        "We computed baseline first/second line xG/60 using total TOI, then added an opponent-defense adjustment for interpretability.\n"
        "Specifically, we split each team’s first_off and second_off production by the opponent’s defense pairing (first_def vs second_def).\n"
        f"We upweighted production against first_def by {W_FIRST_DEF:.2f} (vs {W_SECOND_DEF:.2f} for second_def), then recomputed xG/60 using weighted xG and weighted TOI.\n"
        "This keeps the metric comparable across teams while acknowledging that not all minutes are equally difficult due to matchup quality.\n"
        "We report both baseline and adjusted top-10 lists and quantify rank shifts to make the impact of the adjustment transparent.\n"
    )
    rationale_path = reports_dir / "adjustment_rationale.txt"
    rationale_path.write_text(rationale, encoding="utf-8")

    n_teams = int(out["team"].nunique())
    nan_ratio = int(out["disparity_ratio_adj"].isna().sum())

    print("=== STEP08 LINE DISPARITY ADJUSTED (vs opponent defense pairing) ===")
    print(f"input:  {whl_csv_path.resolve()}")
    if baseline_path is not None:
        print(f"baseline: {baseline_path.resolve()}")
    else:
        print("baseline: (not found) - comparison will be skipped")
    print(f"output: {adjusted_all_path.resolve()}")
    print(f"output: {adjusted_top10_path.resolve()}")
    print(f"output: {compare_path.resolve()}")
    print(f"rationale: {rationale_path.resolve()}")
    print(f"teams: {n_teams} (expected 32) | NaN adj ratios: {nan_ratio}")

    print("\n--- ADJUSTED TOP 10 ---")
    show = top10[
        [
            "team",
            "disparity_ratio_adj",
            "first_off_xg60_adj",
            "second_off_xg60_adj",
            "first_off_adj_toi",
            "second_off_adj_toi",
        ]
    ].copy()
    with pd.option_context("display.width", 180):
        print(show.to_string(index=False))

    if overlap is not None and movers is not None:
        print(f"\nbaseline vs adjusted TOP10 overlap: {overlap}/10")
        print("\n--- Biggest rank movers (abs delta) ---")
        with pd.option_context("display.width", 180):
            print(
                movers[["team", "baseline_rank", "adjusted_rank", "rank_delta", "disparity_ratio", "disparity_ratio_adj"]]
                .to_string(index=False)
            )

    ok_teams = (n_teams == 32)
    ok_top10 = (len(top10) == 10)
    overall = ok_teams and ok_top10
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")

    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
