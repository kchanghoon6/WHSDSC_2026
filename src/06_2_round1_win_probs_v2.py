"""
Round 1 home-win probabilities (v2).

Adds a more explicit feature blend and produces a submission-ready CSV  (overwrites out/round1_homewin_probs.csv).
Also saves a versioned copy as out/round1_homewin_probs_v2.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skellam

import statsmodels.api as sm
import statsmodels.formula.api as smf


REQ_GAMES_COLS = [
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
]

REQ_MATCHUP_COLS = {"home_team", "away_team"}  # use game_id if available


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
    reports_dir = Path(paths.get("reports_dir", project_root / "reports")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    games_path = data_work_dir / "games_2025.csv"
    if not games_path.exists():
        raise FileNotFoundError(f"games_2025.csv not found: {games_path} (run STEP 3)")

    expected = rc.get("expected_files", {})
    matchups_name = expected.get("matchups_xlsx", "WHSDSC_Rnd1_matchups.xlsx")
    matchups_path = data_raw_dir / matchups_name
    if not matchups_path.exists():
        alt = data_raw_dir / "WHSDSC_Rnd1_matchups.xlsx"
        if alt.exists():
            matchups_path = alt
        else:
            raise FileNotFoundError(f"Round1 matchups xlsx not found in: {data_raw_dir}")

    return {
        "games_path": games_path,
        "matchups_path": matchups_path,
        "out_dir": out_dir,
        "reports_dir": reports_dir,
    }


def _canon(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


def _read_matchups(path: Path) -> pd.DataFrame:
    sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    for _, df in sheets.items():
        cols = {c.strip().lower() for c in df.columns}
        if REQ_MATCHUP_COLS.issubset(cols):
            col_map = {c: c.strip().lower() for c in df.columns}
            df = df.rename(columns=col_map)
            keep = ["home_team", "away_team"] + (["game_id"] if "game_id" in df.columns else [])
            return df[keep].copy()
    raise ValueError(f"No sheet in {path.name} contains columns: {REQ_MATCHUP_COLS}")


def _skellam_home_win_prob(lam_home: float, lam_away: float) -> float:
    # P(H>A) + 0.5*P(tie)
    cdf0 = skellam.cdf(0, lam_home, lam_away)
    pmf0 = skellam.pmf(0, lam_home, lam_away)
    p = float((1.0 - cdf0) + 0.5 * pmf0)
    return max(0.0, min(1.0, p))


def main() -> int:
    project_root = _resolve_project_root()
    paths = _get_paths(project_root)

    g = pd.read_csv(paths["games_path"], usecols=REQ_GAMES_COLS)
    miss = [c for c in REQ_GAMES_COLS if c not in g.columns]
    if miss:
        raise ValueError(f"Missing required columns in games_2025.csv: {miss}")

    g["home_goals"] = pd.to_numeric(g["home_goals"], errors="coerce")
    g["away_goals"] = pd.to_numeric(g["away_goals"], errors="coerce")

    home = pd.DataFrame(
        {
            "goals": g["home_goals"],
            "is_home": 1,
            "offense_team": g["home_team"].astype(str),
            "defense_team": g["away_team"].astype(str),
        }
    )
    away = pd.DataFrame(
        {
            "goals": g["away_goals"],
            "is_home": 0,
            "offense_team": g["away_team"].astype(str),
            "defense_team": g["home_team"].astype(str),
        }
    )
    df = pd.concat([home, away], ignore_index=True)

    # Fit Poisson GLM (same structure as STEP05)
    model = smf.glm(
        formula="goals ~ is_home + C(offense_team) + C(defense_team)",
        data=df,
        family=sm.families.Poisson(),
    )
    res = model.fit()

    params = res.params.to_dict()
    intercept = float(params.get("Intercept", 0.0))
    beta_home = float(params.get("is_home", 0.0))

    teams = sorted(pd.unique(pd.concat([g["home_team"], g["away_team"]]).astype(str)).tolist())
    attack = {t: float(params.get(f"C(offense_team)[T.{t}]", 0.0)) for t in teams}
    defense_allow = {t: float(params.get(f"C(defense_team)[T.{t}]", 0.0)) for t in teams}

    canon_to_team = {_canon(t): t for t in teams}

    def lam(off: str, deff: str, is_home_flag: int) -> float:
        eta = intercept + (beta_home * is_home_flag) + attack[off] + defense_allow[deff]
        return float(np.exp(eta))

    matchups = _read_matchups(paths["matchups_path"])
    matchups["home_team"] = matchups["home_team"].astype(str)
    matchups["away_team"] = matchups["away_team"].astype(str)
    matchups["home_canon"] = matchups["home_team"].map(_canon)
    matchups["away_canon"] = matchups["away_team"].map(_canon)

    unknown = []
    rows = []
    for i, r in matchups.iterrows():
        hc = r["home_canon"]
        ac = r["away_canon"]
        if hc not in canon_to_team or ac not in canon_to_team:
            unknown.append((r.get("game_id", f"row_{i}"), r["home_team"], r["away_team"]))
            continue

        ht = canon_to_team[hc]
        at = canon_to_team[ac]

        lam_home = lam(ht, at, 1)
        lam_away = lam(at, ht, 0)
        p_home = _skellam_home_win_prob(lam_home, lam_away)

        rows.append(
            {
                "game_id": r["game_id"] if "game_id" in matchups.columns else f"game_{i+1}",
                "home_team": r["home_team"],
                "away_team": r["away_team"],
                "p_home_win": p_home,
                "lambda_home": lam_home,
                "lambda_away": lam_away,
                "method_tag": "poisson_glm_attack_def_v2",
            }
        )

    if unknown:
        msg = "\n".join([f"{gid}: {ht} vs {at}" for gid, ht, at in unknown[:10]])
        raise ValueError(
            "Some matchup teams could not be mapped to team list from games_2025.csv (name mismatch).\n"
            "First examples:\n" + msg
        )

    out = pd.DataFrame(rows)

    # Save: keep v2 separate + also write the official submission filename
    out_v2_path = Path(paths["out_dir"]) / "round1_homewin_probs_v2.csv"
    out_submit_path = Path(paths["out_dir"]) / "round1_homewin_probs.csv"
    out.to_csv(out_v2_path, index=False)
    out.to_csv(out_submit_path, index=False)

    # DoD checks
    n = len(out)
    row_ok = (n == 16)
    in_range = bool(((out["p_home_win"] >= 0) & (out["p_home_win"] <= 1)).all())
    no_nan = bool(out["p_home_win"].notna().all())

    print("=== STEP06 ROUND 1 HOME WIN PROBS (V2: Poisson GLM) ===")
    print(f"matchups: {paths['matchups_path'].resolve()}")
    print(f"games:    {paths['games_path'].resolve()}")
    print(f"output(v2):     {out_v2_path.resolve()}")
    print(f"output(submit): {out_submit_path.resolve()}")
    print(f"home_adv_beta(log)={beta_home:.6f} | exp(beta)={np.exp(beta_home):.6f} | intercept={intercept:.6f}")
    print(f"rows: {n} (expected 16) | in_range: {in_range} | no_nan: {no_nan}")

    print("\n--- CSV (copy/paste) ---")
    print(out.to_csv(index=False).strip())

    overall = row_ok and in_range and no_nan
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
