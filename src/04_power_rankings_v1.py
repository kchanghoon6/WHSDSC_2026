"""
Baseline power rankings from expected goals.

Computes a simple, explainable strength score using per-game xG for/against.
Writes out/league_table.csv and out/power_ranking_v1.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_GAME_COLS = [
    "game_id",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
    "home_xg",
    "away_xg",
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


def _paths_from_run_config(project_root: Path) -> dict:
    rc = _read_run_config(project_root)
    paths = rc.get("paths", {})
    data_work_dir = Path(paths.get("data_work_dir", project_root / "data_work")).resolve()
    out_dir = Path(paths.get("out_dir", project_root / "out")).resolve()
    reports_dir = Path(paths.get("reports_dir", project_root / "reports")).resolve()
    data_work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    return {"data_work_dir": data_work_dir, "out_dir": out_dir, "reports_dir": reports_dir}


def _coerce_numeric(df: pd.DataFrame, cols: list[str]) -> None:
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")


def main() -> int:
    project_root = _resolve_project_root()
    paths = _paths_from_run_config(project_root)

    games_path = paths["data_work_dir"] / "games_2025.csv"
    if not games_path.exists():
        raise FileNotFoundError(f"games_2025.csv not found at: {games_path}. Run STEP 3 first.")
    
    df = pd.read_csv(games_path)

    missing = [c for c in REQUIRED_GAME_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in games_2025.csv: {missing}")
    
    _coerce_numeric(df, ["home_goals", "away_goals", "home_xg", "away_xg"])

    # Build league table (standings)
    # Determine outcomes per game:
    home_win = (df["home_goals"] > df["away_goals"]).astype(float)
    away_win = (df["away_goals"] > df["home_goals"]).astype(float)
    tie = (df["home_goals"] == df["away_goals"]).astype(float)

    # For ties (unlikely), split as 0.5 win / 0.5 loss for both teams so totals stay consistent
    home_wins = home_win + 0.5 * tie
    away_wins = away_win + 0.5 * tie
    home_losses = away_win + 0.5 * tie
    away_losses = home_win + 0.5 * tie

    home_rows = pd.DataFrame(
        {
            "team": df["home_team"].astype(str),
            "games_played": 1,
            "wins": home_wins,
            "losses": home_losses,
            "goals_for": df["home_goals"],
            "goals_against": df["away_goals"],
            "xg_for": df["home_xg"],
            "xg_against": df["away_xg"],
        }
    )

    away_rows = pd.DataFrame(
        {
            "team": df["away_team"].astype(str),
            "games_played": 1,
            "wins": away_wins,
            "losses": away_losses,
            "goals_for": df["away_goals"],
            "goals_against": df["home_goals"],
            "xg_for": df["away_xg"],
            "xg_against": df["home_xg"],
        }
    )

    team_game = pd.concat([home_rows, away_rows], ignore_index=True)

    league = (
        team_game.groupby("team", as_index=False)
        .agg(
            games_played=("games_played", "sum"),
            wins=("wins", "sum"),
            losses=("losses", "sum"),
            goals_for=("goals_for", "sum"),
            goals_against=("goals_against", "sum"),
            xg_for=("xg_for", "sum"),
            xg_against=("xg_against", "sum"),
        )
        .copy()
    )

    league["goal_diff"] = league["goals_for"] - league["goals_against"]
    league["xg_diff"] = league["xg_for"] - league["xg_against"]

    # Make wins/losses look clean if they are integers (ties normally absent)
    # Keep as float to be safe, but also create integer display columns if very close to int
    league["wins"] = league["wins"].round(3)
    league["losses"] = league["losses"].round(3)

    # Sort standings: wins desc, then goal_diff desc, then goals_for desc
    league_sorted = league.sort_values(
        by=["wins", "goal_diff", "goals_for", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    # Power ranking v1 (xG-based strength score)
    # Score = xG_for per game - xG_against per game
    league_sorted["xg_for_pg"] = league_sorted["xg_for"] / league_sorted["games_played"]
    league_sorted["xg_against_pg"] = league_sorted["xg_against"] / league_sorted["games_played"]
    league_sorted["strength_score_raw"] = league_sorted["xg_for_pg"] - league_sorted["xg_against_pg"]

    # Optional weak normalization (z-score) for readability
    mu = float(league_sorted["strength_score_raw"].mean())
    sd = float(league_sorted["strength_score_raw"].std(ddof=0))
    league_sorted["strength_score_z"] = (league_sorted["strength_score_raw"] - mu) / (sd if sd > 0 else 1.0)

    rankings = league_sorted.sort_values(
        by=["strength_score_raw", "wins", "goal_diff", "team"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)

    rankings["power_rank"] = np.arange(1, len(rankings) + 1)

    out_dir = paths["out_dir"]
    league_out_path = out_dir / "league_table.csv"
    power_out_path = out_dir / "power_rankings_v1.csv"

    # league_table: keep minimal required + helpful extra for later
    league_table_cols = [
        "team",
        "games_played",
        "wins",
        "losses",
        "goals_for",
        "goals_against",
        "goal_diff",
        "xg_for",
        "xg_against",
        "xg_diff",
    ]
    league_sorted[league_table_cols].to_csv(league_out_path, index=False)

    power_cols = [
        "power_rank",
        "team",
        "strength_score_raw",
        "strength_score_z",
        "xg_for_pg",
        "xg_against_pg",
        "wins",
        "losses",
        "goal_diff",
    ]
    rankings[power_cols].to_csv(power_out_path, index=False)

    # DoD checks
    teams_n = int(rankings["team"].nunique())
    ranks_unique = bool(rankings["power_rank"].is_unique)
    ranks_complete = bool(set(rankings["power_rank"]) == set(range(1, 33)))
    dod_ok = (teams_n == 32) and ranks_unique and ranks_complete

    print("=== STEP04 POWER RANKING V1 (xG baseline) ===")
    print(f"input: {games_path.resolve()}")
    print(f"output: {league_out_path.resolve()}")
    print(f"output: {power_out_path.resolve()}")
    print(f"teams: {teams_n} (expected 32)")
    print(f"rank unique: {ranks_unique} | rank complete 1..32: {ranks_complete}")

    print("\n--- TOP 10 (team, score, xG_for_pg, xG_against_pg) ---")
    top10 = rankings.head(10)[["power_rank", "team", "strength_score_raw", "xg_for_pg", "xg_against_pg"]]
    with pd.option_context("display.width", 140):
        print(top10.to_string(index=False))

    print("\n--- BOTTOM 5 ---")
    bot5 = rankings.tail(5)[["power_rank", "team", "strength_score_raw", "xg_for_pg", "xg_against_pg"]]
    with pd.option_context("display.width", 140):
        print(bot5.to_string(index=False))

    print("\n--- SCORE DISTRIBUTION (raw) min/mean/max ---")
    smin = float(rankings["strength_score_raw"].min())
    smean = float(rankings["strength_score_raw"].mean())
    smax = float(rankings["strength_score_raw"].max())
    print(f"{smin:.6f} / {smean:.6f} / {smax:.6f}")

    print(f"\nOVERALL: {'PASS' if dod_ok else 'FAIL'}")

    return 0 if dod_ok else 2

if __name__ == "__main__":
    raise SystemExit(main())
