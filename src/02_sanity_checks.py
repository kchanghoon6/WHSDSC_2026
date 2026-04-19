"""
Sanity-check teh raw WHL 2025 record-level dataset.

Validates required columns, expected league/game counts, and basic schedule balance.
Outputs quick QA artifacts under reports/ (CSV summaries + sanity_summary.json).
"""


from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_COLS = [
    "game_id",
    "record_id",
    "home_team",
    "away_team",
    "toi",
    "went_ot",
    "home_goals",
    "away_goals",
]


def _read_run_config(project_root: Path) -> dict:
    run_config_path = project_root / "reports" / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(f"run_config.json not found at: {run_config_path}")
    return json.loads(run_config_path.read_text(encoding="utf-8"))


def _resolve_paths(run_config: dict) -> dict:
    paths = run_config.get("paths", {})
    project_root = Path(paths.get("project_root", "")).resolve()
    data_raw_dir = Path(paths.get("data_raw_dir", "")).resolve()

    if not project_root.exists():
        raise FileNotFoundError(f"project_root does not exist: {project_root}")
    if not data_raw_dir.exists():
        raise FileNotFoundError(f"data_raw_dir does not exist: {data_raw_dir}")

    expected = run_config.get("expected_files", {})
    whl_csv_name = expected.get("whl_2025_csv", "whl_2025.csv")
    whl_csv_path = data_raw_dir / whl_csv_name
    if not whl_csv_path.exists():
        # fallback: try direct name
        alt = data_raw_dir / "whl_2025.csv"
        if alt.exists():
            whl_csv_path = alt
        else:
            raise FileNotFoundError(f"whl_2025.csv not found in data_raw: {data_raw_dir}")

    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    return {
        "project_root": project_root,
        "data_raw_dir": data_raw_dir,
        "whl_csv_path": whl_csv_path,
        "reports_dir": reports_dir,
    }


def _safe_float_series(s: pd.Series) -> pd.Series:
    # TOI can be float; coerce and keep NaN if malformed
    return pd.to_numeric(s, errors="coerce")


def main() -> int:
    cwd = Path.cwd().resolve()

    project_root = cwd
    if (cwd / "reports" / "run_config.json").exists():
        project_root = cwd
    elif (cwd.parent / "reports" / "run_config.json").exists():
        project_root = cwd.parent
    else:
        raise FileNotFoundError(
            "Could not locate reports/run_config.json. Run this from PROJECT_ROOT or PROJECT_ROOT/src."
        )

    run_config = _read_run_config(project_root)
    paths = _resolve_paths(run_config)

    whl_csv_path: Path = paths["whl_csv_path"]
    reports_dir: Path = paths["reports_dir"]

    # NOTE: dtype hints keep team names as strings; went_ot often 0/1 but read as int.
    usecols = list(dict.fromkeys(REQUIRED_COLS))  # unique while preserving order
    df = pd.read_csv(whl_csv_path, usecols=usecols)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in whl_2025.csv: {missing}")

    df["toi"] = _safe_float_series(df["toi"])
    df["went_ot"] = pd.to_numeric(df["went_ot"], errors="coerce").fillna(0).astype(int)

    teams_union = pd.unique(
        pd.concat(
            [df["home_team"].astype(str), df["away_team"].astype(str)],
            ignore_index=True
        )
    )
    num_teams = int(len(teams_union))

    n_games = int(df["game_id"].nunique(dropna=True))

    # Each game_id can appear in multiple rows; deduplicate before counting team appearances.
    
    game_pairs = (
        df[["game_id", "home_team", "away_team"]]
        .drop_duplicates(subset=["game_id"])
        .copy()
    )

    pair_counts = (
        df.groupby("game_id")[["home_team", "away_team"]]
        .nunique()
        .rename(columns={"home_team": "home_team_nunique", "away_team": "away_team_nunique"})
    )
    bad_game_ids = pair_counts[
        (pair_counts["home_team_nunique"] != 1) | (pair_counts["away_team_nunique"] != 1)
    ].index.tolist()

    home_counts = game_pairs["home_team"].value_counts()
    away_counts = game_pairs["away_team"].value_counts()
    team_games = (home_counts.add(away_counts, fill_value=0)).astype(int).sort_index()

    team_games_values = team_games.values
    tg_min = int(np.min(team_games_values)) if len(team_games_values) else 0
    tg_median = float(np.median(team_games_values)) if len(team_games_values) else 0.0
    tg_max = int(np.max(team_games_values)) if len(team_games_values) else 0


    # TOI sanity: sum per game_id (record-level segments -> game total seconds).
    
    game_toi = df.groupby("game_id", as_index=False).agg(
        toi_sum=("toi", "sum"),
        went_ot=("went_ot", "max"),
        home_goals=("home_goals", "sum"),
        away_goals=("away_goals", "sum"),
    )

    game_toi_sorted = game_toi.sort_values("toi_sum", ascending=False).reset_index(drop=True)
    top20_toi = game_toi_sorted.head(20).copy()

    # Home win defined as home_goals > away_goals using game-level totals.
    # If tie exists (unlikely given hockey ends with winner), count as 0.5 to avoid distortion.
    home_wins = (game_toi["home_goals"] > game_toi["away_goals"]).astype(int)
    ties = (game_toi["home_goals"] == game_toi["away_goals"]).astype(int)
    home_win_rate = float((home_wins.sum() + 0.5 * ties.sum()) / len(game_toi)) if len(game_toi) else 0.0

    went_ot_rate = float(game_toi["went_ot"].mean()) if len(game_toi) else 0.0

    team_games_out = (
        team_games.rename_axis("team")
        .reset_index(name="games_played")
        .sort_values(["games_played", "team"], ascending=[False, True])
    )
    team_games_out_path = reports_dir / "team_games_counts.csv"
    team_games_out.to_csv(team_games_out_path, index=False)

    game_toi_out_path = reports_dir / "game_toi_summary.csv"
    game_toi_full = game_toi_sorted.copy()
    game_toi_full.insert(0, "toi_rank_desc", np.arange(1, len(game_toi_full) + 1))
    game_toi_full.to_csv(game_toi_out_path, index=False)

    sanity = {
        "teams": num_teams,
        "games": n_games,
        "went_ot_rate": went_ot_rate,
        "home_win_rate": home_win_rate,
        "team_games_distribution": {
            "min": tg_min,
            "median": tg_median,
            "max": tg_max,
            "expected_each_team": 82,
        },
        "game_id_pair_consistency": {
            "bad_game_ids_count": len(bad_game_ids),
            "bad_game_ids_sample": bad_game_ids[:20],
        },
        "files_written": {
            "team_games_counts_csv": str(team_games_out_path.resolve()),
            "game_toi_summary_csv": str(game_toi_out_path.resolve()),
        },
    }
    sanity_path = reports_dir / "sanity_summary.json"
    sanity_path.write_text(json.dumps(sanity, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== STEP02 SANITY CHECKS ===")
    print(f"whl_csv_path: {whl_csv_path.resolve()}")
    print(f"teams (union home/away): {num_teams}")
    print(f"games (unique game_id): {n_games}")
    print(f"went_ot_rate: {went_ot_rate:.4f}")
    print(f"home_win_rate (ties as 0.5): {home_win_rate:.4f}")
    print(f"team_games_played min/median/max: {tg_min} / {tg_median:.1f} / {tg_max}")
    print(f"game_id pair consistency bad_game_ids_count: {len(bad_game_ids)}")
    if bad_game_ids:
        print(f"bad_game_ids_sample (up to 10): {bad_game_ids[:10]}")

    print("\n--- TOP 10 games by TOI sum (seconds) ---")
    top10 = top20_toi.head(10)[["game_id", "toi_sum", "went_ot", "home_goals", "away_goals"]]
    with pd.option_context("display.max_rows", 20, "display.max_columns", 20, "display.width", 120):
        print(top10.to_string(index=False))

    # DoD / PASS-FAIL criteria
    # PASS if exact expected counts and schedule balance, and pair consistency has no bad ids.
    pass_teams = (num_teams == 32)
    pass_games = (n_games == 1312)
    pass_balance = (tg_min == 82 and tg_max == 82)
    pass_pairs = (len(bad_game_ids) == 0)

    print("\n=== DoD (PASS CRITERIA) ===")
    print(f"teams==32: {pass_teams}")
    print(f"games==1312: {pass_games}")
    print(f"team_games min=max=82: {pass_balance}")
    print(f"home/away pair fixed per game_id (no bad ids): {pass_pairs}")

    overall_pass = pass_teams and pass_games and pass_balance and pass_pairs
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")

    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
