"""
Power rankings with exposure offset (v3).

Fits a Poisson-style model with exposure/offset handling to stabilize comparisons across teams.
Writes out/power_ranking_v3_offset.csv and a submission copy as out/power_rankings.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import skellam


REQ_COLS = [
    "game_id",
    "home_team",
    "away_team",
    "home_goals",
    "away_goals",
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


def _get_seed(rc: dict) -> int:
    return int(rc.get("seed", 2026))


def _get_paths(project_root: Path) -> dict:
    rc = _read_run_config(project_root)
    paths = rc.get("paths", {})

    data_work_dir = Path(paths.get("data_work_dir", project_root / "data_work")).resolve()
    out_dir = Path(paths.get("out_dir", project_root / "out")).resolve()
    reports_dir = Path(paths.get("reports_dir", project_root / "reports")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    games_path = data_work_dir / "games_2025.csv"
    if not games_path.exists():
        raise FileNotFoundError(f"games_2025.csv not found: {games_path} (run STEP 3)")

    return {
        "rc": rc,
        "seed": _get_seed(rc),
        "games_path": games_path,
        "out_dir": out_dir,
        "reports_dir": reports_dir,
    }


def _skellam_home_win_prob(lambda_home: float, lambda_away: float) -> float:
    # P(home>away) + 0.5*P(tie)
    cdf0 = skellam.cdf(0, lambda_home, lambda_away)
    pmf0 = skellam.pmf(0, lambda_home, lambda_away)
    p_gt0 = 1.0 - cdf0
    p = float(p_gt0 + 0.5 * pmf0)
    return max(0.0, min(1.0, p))


def _safe_log(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


def main() -> int:
    project_root = _resolve_project_root()
    paths = _get_paths(project_root)
    seed = paths["seed"]
    np.random.seed(seed)

    games = pd.read_csv(paths["games_path"])
    miss = [c for c in REQ_COLS if c not in games.columns]
    if miss:
        raise ValueError(f"Missing required columns in games_2025.csv: {miss}")
    
    games["home_goals"] = pd.to_numeric(games["home_goals"], errors="coerce")
    games["away_goals"] = pd.to_numeric(games["away_goals"], errors="coerce")
    games["toi"] = pd.to_numeric(games["toi"], errors="coerce")

    if games[["home_goals", "away_goals", "toi"]].isna().any().any():
        raise ValueError("Found NaN in required columns (home_goals/away_goals/toi")
    
    # exposure in hours (TOI is seconds)
    toi_hours = (games["toi"].to_numpy(dtype=float) / 3600.0)
    log_exposure = np.log(np.clip(toi_hours, 1e-9, None))

    # build long format: one row per team per game
    home = pd.DataFrame(
        {
            "goals": games["home_goals"].astype(float),
            "is_home": 1.0,
            "team": games["home_team"].astype(str),
            "opp": games["away_team"].astype(str),
            "log_exposure": log_exposure,
        }
    )
    away = pd.DataFrame(
        {
            "goals": games["away_goals"].astype(float),
            "is_home": 0.0,
            "team": games["away_team"].astype(str),
            "opp": games["home_team"].astype(str),
            "log_exposure": log_exposure,
        }
    )
    long = pd.concat([home, away], ignore_index=True)

    teams = sorted(pd.unique(pd.concat([games["home_team"], games["away_team"]]).astype(str)))
    if len(teams) != 32:
        raise ValueError(f"Expected 32 teams, got {len(teams)}")
    
    ref_team = teams[0]  # deterministic reference for identifiability

    # design matrix: const + is_home + attack(team) dummies + defense_allow(opp) dummies
    X = pd.DataFrame({"const": 1.0, "is_home": long["is_home"].astype(float).to_numpy()})
    team_d = pd.get_dummies(long["team"], prefix="atk", drop_first=True, dtype=float)
    opp_d = pd.get_dummies(long["team"], prefix="def", drop_first=True, dtype=float)
    X = pd.concat([X, team_d, opp_d], axis=1).astype(float)

    y = long["goals"].astype(float).to_numpy()
    model = sm.GLM(y, X, family=sm.families.Poisson(), offset=long["log_exposure"].astype(float).to_numpy())
    res = model.fit()

    intercept = float(res.params["const"])
    home_beta = float(res.params["is_home"])

    attack_coef = {t: float(res.params.get(f"atk_{t}", 0.0)) for t in teams}
    defense_allow_coef = {t: float(res.params.get(f"def_{t}", 0.0)) for t in teams}

    # strength = attack - defense_allow (smaller defense_allow is better defense)
    raw_strength = {t: attack_coef[t] - defense_allow_coef[t] for t in teams}
    max_strength = max(raw_strength.values())
    strength_score = {t: raw_strength[t] - max_strength for t in teams}  # top team = 0, others <= 0

    # Evaluate on season games using Skellam with TOI exposure
    lam_home = np.exp(
        intercept
        + games["away_team"].astype(str).map(attack_coef).to_numpy(dtype=float)
        + games["home_team"].astype(str).map(defense_allow_coef).to_numpy(dtype=float)
        + log_exposure
    )
    lam_away = np.exp(
        intercept
        + games["home_team"].astype(str).map(attack_coef).to_numpy(dtype=float)
        + games["away_team"].astype(str).map(defense_allow_coef).to_numpy(dtype=float)
        + log_exposure
    )
    p_home = np.array([_skellam_home_win_prob(a, b) for a, b in zip(lam_home, lam_away)], dtype=float)

    y_win = (games["home_goals"].to_numpy(dtype=float) > games["away_goals"].to_numpy(dtype=float)).astype(int)
    ties = int((games["home_goals"] == games["away_goals"]).sum())  # likely 0

    eps = 1e-15
    brier = float(np.mean((p_home - y_win) ** 2))
    logloss = float(-np.mean(y_win * _safe_log(p_home, eps) + (1 - y_win) * _safe_log(1 - p_home, eps)))

    # rank table
    pr = pd.DataFrame(
        {
            "team": teams,
            "strength_score": [strength_score[t] for t in teams],
            "attack_coef": [attack_coef[t] for t in teams],
            "defense_allow_coef": [defense_allow_coef[t] for t in teams],
        }
    ).sort_values("strength_score", ascending=False, kind="mergesort")

    pr["power_rank"] = np.arange(1, len(pr) + 1)
    pr = pr[["power_rank", "team", "strength_score", "attack_coef", "defense_allow_coef"]]

    out_dir = paths["out_dir"]
    reports_dir = paths["reports_dir"]

    out_rank = out_dir / "power_rankings_v3_offset.csv"
    out_submit = out_dir / "power_rankings.csv"  # submit copy
    pr.to_csv(out_rank, index=False)
    pr.to_csv(out_submit, index=False)

    params_json = {
        "model_tag": "poisson_glm_attack_def_offset_v3",
        "reference_team": ref_team,
        "seed": seed,
        "intercept": intercept,
        "home_adv_beta_log": home_beta,
        "home_adv_mult": float(np.exp(home_beta)),
        "attack_coef": attack_coef,
        "defense_allow_coef": defense_allow_coef,
    }
    (reports_dir / "poisson_glm_v3_offset_params.json").write_text(json.dumps(params_json, indent=2), encoding="utf-8")

    metrics_json = {
        "model_tag": "poisson_glm_attack_def_offset_v3",
        "n_games": int(games.shape[0]),
        "ties": ties,
        "brier": brier,
        "logloss": logloss,
        "aic": float(res.aic),
        "deviance": float(res.deviance),
        "home_adv_beta_log": home_beta,
        "home_adv_mult": float(np.exp(home_beta)),
        "intercept": intercept,
    }    
    (reports_dir / "poisson_glm_v3_metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")

    top10 = pr.head(10).copy()
    bottom5 = pr.tail(5).copy()

    print("=== STEP12 POWER RANKINGS V3 (Poisson attack/defense + TOI offset) ===")
    print(f"input:  {paths['games_path'].resolve()}")
    print(f"output: {out_rank.resolve()}")
    print(f"submit: {out_submit.resolve()}")
    print(f"params: {(reports_dir / 'poisson_glm_v3_offset_params.json').resolve()}")
    print(f"home_adv_beta(log): {home_beta:.6f} | exp(beta): {np.exp(home_beta):.6f} | intercept: {intercept:.6f}")
    print(f"eval: brier={brier:.6f} | logloss:{logloss:.6f} | aic={res.aic:.2f} | deviance={res.deviance:.2f}")
    print(f"teams: {len(teams)} (expected 32) | rank complete: {pr['power_rank'].nunique()==32}")

    print("\n--- V3 TOP 10 ---")
    print(top10.to_string(index=False))

    print("\n--- V3 BOTTOM 5 ---")
    print(bottom5.to_string(index=False))

    overall = (len(teams) == 32) and (pr["power_rank"].nunique() == 32)
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
