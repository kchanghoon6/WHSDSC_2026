"""
Model based power rankings.

Fits a goals model and converts team strengths into win probabilities (Skellam-based).
Writes out/power_rankings_v2_model.csv and a comparison table under reports/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.stats import skellam

import statsmodels.api as sm
import statsmodels.formula.api as smf


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


def _paths(project_root: Path) -> dict:
    rc = _read_run_config(project_root)
    paths = rc.get("paths", {})
    data_work = Path(paths.get("data_work_dir", project_root / "data_work")).resolve()
    out_dir = Path(paths.get("out_dir", project_root / "out")).resolve()
    reports = Path(paths.get("reports_dir", project_root / "reports")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    return {"data_work": data_work, "out": out_dir, "reports": reports}


def _canon_team(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


def _skellam_homewin_prob(lam_home: float, lam_away: float) -> float:
    # P(H>A) + 0.5*P(tie)
    cdf0 = skellam.cdf(0, lam_home, lam_away)
    pmf0 = skellam.pmf(0, lam_home, lam_away)
    p = float((1.0 - cdf0) + 0.5 * pmf0)
    return max(0.0, min(1.0, p))


def main() -> int:
    project_root = _resolve_project_root()
    p = _paths(project_root)

    games_path = p["data_work"] / "games_2025.csv"
    if not games_path.exists():
        raise FileNotFoundError(f"Missing {games_path}. Run STEP 3 first.")

    g = pd.read_csv(games_path)

    need = ["home_team", "away_team", "home_goals", "away_goals"]
    missing = [c for c in need if c not in g.columns]
    if missing:
        raise ValueError(f"Missing columns in games_2025.csv: {missing}")

    g["home_goals"] = pd.to_numeric(g["home_goals"], errors="coerce")
    g["away_goals"] = pd.to_numeric(g["away_goals"], errors="coerce")

    # Build long format: two rows per game
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

    # Fit Poisson GLM: goals ~ home_adv + offense + defense
    # Note: defense_team coefficient represents "goals allowed tendency" (higher => weaker defense).
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

    # Reconstruct full attack/defense effect vectors (reference category => 0)
    attack = {}
    defense_allow = {}
    for t in teams:
        atk_key = f"C(offense_team)[T.{t}]"
        def_key = f"C(defense_team)[T.{t}]"
        attack[t] = float(params.get(atk_key, 0.0))
        defense_allow[t] = float(params.get(def_key, 0.0))

    # Strength score (explainable): attack - defense_allow
    v2 = pd.DataFrame(
        {
            "team": teams,
            "attack_coef": [attack[t] for t in teams],
            "defense_allow_coef": [defense_allow[t] for t in teams],
        }
    )
    v2["strength_score"] = v2["attack_coef"] - v2["defense_allow_coef"]
    mu = float(v2["strength_score"].mean())
    sd = float(v2["strength_score"].std(ddof=0))
    v2["strength_z"] = (v2["strength_score"] - mu) / (sd if sd > 0 else 1.0)

    v2 = v2.sort_values(["strength_score", "team"], ascending=[False, True]).reset_index(drop=True)
    v2["power_rank"] = np.arange(1, len(v2) + 1)

    # Evaluate on historical games (simple, transparent)
    # Build predicted lambda for each historical game, then p_home_win via Skellam
    def lam(off: str, deff: str, is_home_flag: int) -> float:
        eta = intercept + (beta_home * is_home_flag) + attack[off] + defense_allow[deff]
        return float(np.exp(eta))

    lam_home = []
    lam_away = []
    p_home = []
    y = []
    for _, r in g.iterrows():
        ht = str(r["home_team"])
        at = str(r["away_team"])
        lh = lam(ht, at, 1)
        la = lam(at, ht, 0)
        lam_home.append(lh)
        lam_away.append(la)
        p_home.append(_skellam_homewin_prob(lh, la))
        if r["home_goals"] > r["away_goals"]:
            y.append(1.0)
        elif r["home_goals"] < r["away_goals"]:
            y.append(0.0)
        else:
            y.append(0.5)

    y = np.array(y, dtype=float)
    p_home = np.array(p_home, dtype=float)

    brier = float(np.mean((p_home - y) ** 2))
    eps = 1e-12
    ph = np.clip(p_home, eps, 1 - eps)
    # treat tie as 0.5: use y in {0,0.5,1} - not perfect logloss, but consistent and transparent
    logloss = float(-np.mean(y * np.log(ph) + (1 - y) * np.log(1 - ph)))

    # Compare with v1 if present
    v1_path = p["out"] / "power_rankings_v1.csv"
    compare_rows = []
    spearman_val = None
    if v1_path.exists():
        v1 = pd.read_csv(v1_path)
        # Expect columns: power_rank, team (from our STEP4)
        if "power_rank" in v1.columns and "team" in v1.columns:
            m = v1[["team", "power_rank"]].merge(v2[["team", "power_rank"]], on="team", suffixes=("_v1", "_v2"))
            m["rank_delta"] = m["power_rank_v1"] - m["power_rank_v2"]
            m = m.sort_values("power_rank_v2").reset_index(drop=True)
            compare_rows = m.to_dict(orient="records")
            spearman_val = float(spearmanr(m["power_rank_v1"], m["power_rank_v2"]).statistic)

            (p["reports"] / "rank_compare_v1_v2.csv").write_text(m.to_csv(index=False), encoding="utf-8")

    out_path = p["out"] / "power_rankings_v2_model.csv"
    v2.to_csv(out_path, index=False)

    home_adv_path = p["reports"] / "home_advantage_estimate.json"
    home_adv_path.write_text(
        json.dumps(
            {
                "home_adv_beta_log": beta_home,
                "home_adv_multiplier_exp_beta": float(np.exp(beta_home)),
                "notes": "Poisson GLM: goals ~ is_home + offense_team + defense_team (defense coef = goals allowed tendency)",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    eval_path = p["reports"] / "model_eval_v2.json"
    eval_path.write_text(
        json.dumps(
            {
                "brier": brier,
                "logloss": logloss,
                "n_games": int(len(g)),
                "spearman_rankcorr_v1_v2": spearman_val,
                "aic": float(res.aic),
                "deviance": float(res.deviance),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== STEP05 POWER RANKINGS V2 (Poisson attack/defense) ===")
    print(f"input:  {games_path.resolve()}")
    print(f"output: {out_path.resolve()}")
    print(f"home_adv_beta(log): {beta_home:.6f} | exp(beta): {np.exp(beta_home):.6f}")
    print(f"eval: brier={brier:.6f} | logloss={logloss:.6f} | aic={res.aic:.2f} | deviance={res.deviance:.2f}")
    if spearman_val is not None:
        print(f"spearman(v1 vs v2 ranks): {spearman_val:.4f}")

    print("\n--- V2 TOP 10 ---")
    top10 = v2.head(10)[["power_rank", "team", "strength_score", "attack_coef", "defense_allow_coef"]]
    with pd.option_context("display.width", 160):
        print(top10.to_string(index=False))

    print("\n--- V2 BOTTOM 5 ---")
    bot5 = v2.tail(5)[["power_rank", "team", "strength_score", "attack_coef", "defense_allow_coef"]]
    with pd.option_context("display.width", 160):
        print(bot5.to_string(index=False))

    # DoD
    teams_ok = (v2["team"].nunique() == 32)
    rank_ok = v2["power_rank"].is_unique and set(v2["power_rank"]) == set(range(1, 33))
    overall = bool(teams_ok and rank_ok)
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
