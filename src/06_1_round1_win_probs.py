"""
Round 1 home-win probabilities (v1).

Uses game-level xG rates and a Skellam model to estimate P(home win) for each matchup.
Writes out/round1_homewin_probs.csv (16 matchups).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skellam


REQ_GAMES_COLS = [
    "game_id",
    "home_team",
    "away_team",
    "home_xg",
    "away_xg",
]

REQ_MATCHUP_COLS = {"home_team", "away_team", "game_id"}


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
    data_work_dir = Path(paths.get("data_work_dir", project_root / "data_work")).resolve()
    out_dir = Path(paths.get("out_dir", project_root / "out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # games_2025.csv from STEP 3
    games_path = data_work_dir / "games_2025.csv"
    if not games_path.exists():
        raise FileNotFoundError(f"games_2025.csv not found: {games_path} (run STEP 3)")
    
    # matchups xlsx
    expected = rc.get("expected_files", {})
    matchups_name = expected.get("matchups_xlsx", "WHSDSC_Rnd1_matchups.xlsx")
    matchups_path = data_raw_dir / matchups_name
    if not matchups_path.exists():
        alt = data_raw_dir / "WHSDSC_Rnd1_matchups.xlsx"
        if alt.exists():
            matchups_path = alt
        else:
            raise FileNotFoundError(f"Round1 matchups xlsx not found in: {data_raw_dir}")

    return {"games_path": games_path, "matchups_path": matchups_path, "out_dir": out_dir}


def _canon(s: str) -> str:
    # conservative normalization (keeps underscores)
    return str(s).strip().lower().replace(" ", "_")


def _read_matchups(path: Path) -> pd.DataFrame:
    # Read all sheets and pick one that contains home_team & away_team
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    for name, df in sheets.items():
        cols = {c.strip().lower() for c in df.columns}
        if REQ_MATCHUP_COLS.issubset(cols):
            # Normalize column names to expected
            col_map = {c: c.strip().lower() for c in df.columns}
            df = df.rename(columns=col_map)
            keep = ["home_team", "away_team", "game_id"]
            df = df[keep].copy()
            return df
        
    raise ValueError(f"No sheet in {path.name} contains columns: {REQ_MATCHUP_COLS}")


def _team_xg_stats(games: pd.DataFrame) -> pd.DataFrame:
    # Build per-team xg_for_pg, xg_against_pg from games_2025 (1312 games)
    home = pd.DataFrame(
        {
            "team": games["home_team"].astype(str),
            "xg_for": pd.to_numeric(games["home_xg"], errors="coerce"),
            "xg_against": pd.to_numeric(games["away_xg"], errors="coerce"),
            "gp": 1,
        },
    )
    away = pd.DataFrame(
        {
            "team": games["away_team"].astype(str),
            "xg_for": pd.to_numeric(games["away_xg"], errors="coerce"),
            "xg_against": pd.to_numeric(games["home_xg"], errors="coerce"),
            "gp": 1,
        },
    )
    t = pd.concat([home, away], ignore_index=True)
    stats = (
        t.groupby("team", as_index=False)
        .agg(gp=("gp", "sum"), xg_for=("xg_for", "sum"), xg_against=("xg_against", "sum"))
        .copy()
    )
    stats["xg_for_pg"] = stats["xg_for"] / stats["gp"]
    stats["xg_against_pg"] = stats["xg_against"] / stats["gp"]
    stats["team_canon"] = stats["team"].map(_canon)
    return stats


def _skellam_home_win_prob(lambda_home: float, lambda_away: float) -> float:
    # P(home>away) + 0.5*P(tie)
    # D = home - away ~ Skellam(lambda_home, lambda_away)
    cdf0 = skellam.cdf(0, lambda_home, lambda_away)  # P(D<=0)
    pmf0 = skellam.pmf(0, lambda_home, lambda_away)  # P(D=0)
    p_gt0 = 1.0 - cdf0
    p = float(p_gt0 + 0.5 * pmf0)
    return max(0.0, min(1.0, p))


def main() -> int:
    project_root = _resolve_project_root()
    paths = _get_paths(project_root)

    games = pd.read_csv(paths["games_path"], usecols=REQ_GAMES_COLS)
    miss = [c for c in REQ_GAMES_COLS if c not in games.columns]
    if miss:
        raise ValueError(f"Missing required columns in games_2025.csv: {miss}")
    
    stats = _team_xg_stats(games)
    stat_map = stats.set_index("team_canon")[["xg_for_pg", "xg_against_pg"]].to_dict(orient="index")

    # League-level home xG advantage (per game)
    home_xg_mean = float(pd.to_numeric(games["home_xg"], errors="coerce").mean())
    away_xg_mean = float(pd.to_numeric(games["away_xg"], errors="coerce").mean())
    home_adv = (home_xg_mean - away_xg_mean) / 2.0 # split advantange symmetrically

    # Read matchups (16 rows expected)
    matchups = _read_matchups(paths["matchups_path"])
    matchups["home_team"] = matchups["home_team"].astype(str)
    matchups["away_team"] = matchups["away_team"].astype(str)
    matchups["home_canon"] = matchups["home_team"].map(_canon)
    matchups["away_canon"] = matchups["away_team"].map(_canon)

    # Compute probabilities
    rows = []
    unknown = []
    for i, r in matchups.iterrows():
        h = r["home_canon"]
        a = r["away_canon"]

        if h not in stat_map or a not in stat_map:
            unknown.append((r.get("game_id"), f"row{i}"), r["home_canon"], r["away_canon"])
            continue

        h_for = float(stat_map[h]["xg_for_pg"])
        h_against = float(stat_map[h]["xg_against_pg"])
        a_for = float(stat_map[a]["xg_for_pg"])
        a_against = float(stat_map[a]["xg_against_pg"])

        # Simple, explainable blend:
        base_home = 0.5 * (h_for + a_against)
        base_away = 0.5 * (a_for + h_against)

        lam_home = max(1e-6, base_home + home_adv)
        lam_away = max(1e-6, base_away - home_adv)

        p_home_win = _skellam_home_win_prob(lam_home, lam_away)

        rows.append(
            {
                "game_id": r["game_id"] if "game_id" in matchups.columns else f"game_{i+1}",
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "p_home_win": p_home_win,
                "lambda_home": lam_home,
                "lambda_away": lam_away,
                "method_tag": "xg_blend_skellam_v1",
            }
        )

    if unknown:
        msg = "\n".join([f"{gid}: {ht} vs {at}" for gid, ht, at in unknown[:10]])
        raise ValueError(
            "Some matchup teams could not be mapped to team stats (name mismatch).\n"
            "First examples:\n" + msg
        )

    out = pd.DataFrame(rows)
    out_path = Path(paths["out_dir"] / "round1_homewin_probs.csv")
    out.to_csv(out_path, index=False)

    # DoD checks
    n = len(out)
    in_range = bool(((out["p_home_win"] >= 0) & (out["p_home_win"] <= 1)).all())
    no_nan = bool(out["p_home_win"].notna().all())
    row_ok = (n == 16)

    print("=== STEP06 ROUND 1 HOME WIN PROBS ===")
    print(f"matchups: {paths['matchups_path'].resolve()}")
    print(f"games:    {paths['games_path'].resolve()}")
    print(f"output:   {out_path.resolve()}")
    print(f"home_xg_mean={home_xg_mean:.6f} | away_xg_mean={away_xg_mean:.6f} | home_adv(split)={home_adv:.6f}")
    print(f"rows: {n} expected 16 | in_range: {in_range} | no_nan: {no_nan}")

    # Print CSV text (16 rows) for copy/paste
    print("\n--- CSV (copy/paste) ---")
    print(out.to_csv(index=False).strip())

    overall = row_ok and in_range and no_nan
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
