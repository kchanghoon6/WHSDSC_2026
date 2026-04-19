"""
Bootsrap confidence intervals for line disparity.

Resample games/segments to estimate uncertainty around the first/second line disparity metric.
Writes CI tables under out/ for use in the CI visualization step.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm

OFF_LINES = ["first_off", "second_off"]
DEF_LINES = ["first_def", "second_def"]

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


def _canon_team(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


def _norm_token(x: str) -> str:
    s = str(x).strip().lower()
    s = s.replace(" ", "_").replace("-", "_")
    return s


def _norm_off_line(x: str) -> str:
    s = _norm_token(x)
    if s in {"first_off", "line1", "l1", "1", "first", "first_line", "line_1"}:
        return "first_off"
    if s in {"second_off", "line2", "l2", "2", "second", "second_line", "line_2"}:
        return "second_off"
    return s


def _norm_def_pair(x: str) -> str:
    s = _norm_token(x)
    if s in {"first_def", "first", "d1", "def1", "pair1", "pair_1", "line1_def"}:
        return "first_def"
    if s in {"second_def", "second", "d2", "def2", "pair2", "pair_2", "line2_def"}:
        return "second_def"
    return s


def _prepare_long_df(raw: pd.DataFrame) -> pd.DataFrame:
    cols = set(raw.columns)

    wide_needed = {
        "home_team",
        "away_team",
        "home_off_line",
        "away_off_line",
        "home_def_pairing",
        "away_def_pairing",
        "home_xg",
        "away_xg",
        "toi",
    }

    if wide_needed.issubset(cols):
        home = raw[["home_team", "home_off_line", "away_def_pairing", "home_xg", "toi"]].copy()
        home.columns = ["team_raw", "off_line_raw", "opp_def_raw", "xg", "toi"]
        away = raw[["away_team", "away_off_line", "home_def_pairing", "away_xg", "toi"]].copy()
        away.columns = ["team_raw", "off_line_raw", "opp_def_raw", "xg", "toi"]

        df = pd.concat([home, away], ignore_index=True)
        df["team_canon"] = df["team_raw"].map(_canon_team)
        df["off_line"] = df["off_line_raw"].map(_norm_off_line)
        df["opp_def_pair"] = df["opp_def_raw"].map(_norm_def_pair)

        df["xg"] = pd.to_numeric(df["xg"], errors="coerce")
        df["toi"] = pd.to_numeric(df["toi"], errors="coerce")

        df = df.dropna(subset=["team_canon", "off_line", "opp_def_pair", "xg", "toi"]).copy()
        df = df[(df["toi"] > 0) & (df["xg"] >= 0)].copy()
        df = df[df["off_line"].isin(OFF_LINES)].copy()
        df = df[df["opp_def_pair"].isin(DEF_LINES)].copy()

        return df[["team_canon", "off_line", "opp_def_pair", "xg", "toi"]].copy()
    
    long_team_candidates = ["team", "off_team", "offense_team", "for_team"]
    long_off_candidates = ["off_line", "offense_line", "line_off"]
    long_def_candidates = ["def_line", "defense_line", "opp_def_line", "line_def", "opp_def_pair", "opp_def_pairing"]
    xg_candidates = ["xg", "xg_total", "xg_for"]
    toi_candidates = ["toi", "toi_sec", "seconds", "time_on_ice"]

    def _pick_one(cands: List[str]) -> str:
        for c in cands:
            if c in cols:
                return c
        raise KeyError(f"Missing required column. Tried: {cands} | available={sorted(list(cols))}")
    
    team_col = _pick_one(long_team_candidates)
    off_col = _pick_one(long_off_candidates)
    def_col = _pick_one(long_def_candidates)
    xg_col = _pick_one(xg_candidates)
    toi_col = _pick_one(toi_candidates)

    df = raw[[team_col, off_col, def_col, xg_col, toi_col]].copy()  
    df.columns = ["team_raw", "off_line_raw", "opp_def_raw", "xg", "toi"]

    df["team_canon"] = df["team_raw"].map(_canon_team)
    df["off_line"] = df["off_line_raw"].map(_norm_off_line)
    df["opp_def_pair"] = df["opp_def_raw"].map(_norm_def_pair)

    df["xg"] = pd.to_numeric(df["xg"], errors="coerce")
    df["toi"] = pd.to_numeric(df["toi"], errors="coerce")

    df = df.dropna(subset=["team_canon", "off_line", "opp_def_pair", "xg", "toi"]).copy()
    df = df[(df["toi"] > 0) & (df["xg"] >= 0)].copy()
    df = df[df["off_line"].isin(OFF_LINES)].copy()
    df = df[df["opp_def_pair"].isin(DEF_LINES)].copy()

    return df[["team_canon", "off_line", "opp_def_pair", "xg", "toi"]].copy()


def _fit_poisson_glm(df: pd.DataFrame) -> sm.GLM:
    formula = "xg ~ 0 + C(team_canon):C(off_line) + C(opp_def_pair)"
    model = sm.GLM.from_formula(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df["toi"].to_numpy(dtype=np.float64))
    )
    res = model.fit(maxiter=200, disp=0)
    return res


def _team_line_eta(res, teams: List[str], baseline_opp: str) -> Tuple[np.ndarray, np.ndarray]:
    rows = []
    for t in teams:
        rows.append({"team_canon": t, "off_line": "first_off", "opp_def_pair": baseline_opp, "toi": 1.0, "xg": 0.0})
        rows.append({"team_canon": t, "off_line": "second_off", "opp_def_pair": baseline_opp, "toi": 1.0, "xg": 0.0})
    grid = pd.DataFrame(rows)

    di = res.model.data.design_info
    X = patsy.build_design_matrices([di], grid, return_type="dataframe")[0]
    eta = np.asarray(X @ res.params)

    eta_first = eta[0::2]
    eta_second = eta[1::2]
    return eta_first, eta_second


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_boot', type=int, default=200)
    ap.add_argument('--seed', type=int, default=20260219)
    ap.add_argument('--min_success', type=int, default=150)
    args = ap.parse_args()

    project_root = _resolve_project_root()
    cfg = _read_run_config(project_root)

    input_path = Path(cfg.get('whl_2025_csv', project_root / "data_raw" / "whl_2025.csv"))
    out_all = Path(cfg.get("line_disparity_reg_ci_all", project_root / "out" / "line_disparity_regression_ci_all.csv"))
    out_top10 = Path(cfg.get("line_disparity_reg_ci_top10", project_root / "out" / "line_disparity_regression_ci_top10.csv"))
    diag_path = Path(cfg.get("line_disparity_reg_ci_diag", project_root / "out" / "line_disparity_regression_ci_diagostics.json"))
    rationale_path = Path(cfg.get("line_disparity_reg_ci_rationale", project_root / "out" / "line_disparity_regression_ci_rationale.txt"))

    out_all.parent.mkdir(parents=True, exist_ok=True)
    out_top10.parent.mkdir(parents=True, exist_ok=True)
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    rationale_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== STEP17 LINE DISPARITY (BOOTSTAP CI: regression-based) ===")
    print(f"input:    {input_path}")

    raw = pd.read_csv(input_path)
    df = _prepare_long_df(raw)

    teams = sorted(df["team_canon"].unique().tolist())
    if len(teams) != 32:
        raise RuntimeError(f"teams: {len(teams)} (expected 32)")
    
    df["team_canon"] = pd.Categorical(df["team_canon"], categories=teams)
    df["off_line"] = pd.Categorical(df["off_line"], categories=OFF_LINES)
    df["opp_def_pair"] = pd.Categorical(df["opp_def_pair"], categories=DEF_LINES)

    baseline_opp = "first_def" if "first_def" in df["opp_def_pair"].cat.categories else str(df["opp_def_pair"].cat.categories[0])

    t0 = time.perf_counter()
    res_full = _fit_poisson_glm(df)
    eta_first_full, eta_second_full = _team_line_eta(res_full, teams, baseline_opp)

    ratio_point = np.exp(eta_first_full - eta_second_full)
    first_xg60_point = np.exp(eta_first_full) * 3600.0
    second_xg60_point = np.exp(eta_second_full) * 3600.0 

    rng = np.random.default_rng(args.seed)
    boot_ratios = []
    failures: Dict[str, int] = {}
    attempts = 0

    groups = list(df.groupby(["team_canon", "off_line"], observed=True))
    
    while len(boot_ratios) < args.n_boot and attempts < args.n_boot * 3:
        attempts += 1

        parts = []
        for (_, _), g in groups:
            g = g.reset_index(drop=True)
            idx = rng.integers(0, len(g), size=len(g))
            parts.append(g.iloc[idx])
        
        bdf = pd.concat(parts, ignore_index=True)
        bdf["team_canon"] = pd.Categorical(bdf["team_canon"], categories=teams)
        bdf["off_line"] = pd.Categorical(bdf["off_line"], categories=OFF_LINES)
        bdf["opp_def_pair"] = pd.Categorical(bdf["opp_def_pair"], categories=DEF_LINES)

        try:
            res_b = _fit_poisson_glm(bdf)
            eta_first_b, eta_second_b = _team_line_eta(res_b, teams, baseline_opp)
            ratio_b = np.exp(eta_first_b - eta_second_b)
            if np.any(~np.isfinite(ratio_b)):
                raise FloatingPointError("non-finite ratio in bootstrap")
            boot_ratios.append(ratio_b)
        except Exception as e:
            k = f"{type(e).__name__}"
            failures[k] = failures.get(k, 0) + 1
        
        if len(boot_ratios) % 25 == 0 and len(boot_ratios) > 0:
            print(f"bootstrap: {len(boot_ratios)}/{args.n_boot} (attempts={attempts})")
        
    if len(boot_ratios) < args.min_success:
        raise RuntimeError(
            f"bootstrap success too low: {len(boot_ratios)} (min_success={args.min_success}). failures={failures}"
        )
    
    boot_arr = np.vstack(boot_ratios)
    ci_lo = np.quantile(boot_arr, 0.025, axis=0)
    ci_hi = np.quantile(boot_arr, 0.975, axis=0)
    boot_mean = np.mean(boot_arr, axis=0)
    boot_sd = np.std(boot_arr, axis=0, ddof=1)

    out = pd.DataFrame(
        {
            "team": teams,
            "disparity_ratio_point": ratio_point,
            "disparity_ratio_boot_mean": boot_mean,
            "disparity_ratio_boot_sd": boot_sd,
            "ci95_lo": ci_lo,
            "ci95_hi": ci_hi,
            "first_off_xg60_point": first_xg60_point,
            "second_off_xg60_point": second_xg60_point,
        }
    ).sort_values("disparity_ratio_point", ascending=False, kind="mergesort")

    out.to_csv(out_all, index=False, encoding='utf-8')
    out.head(10).to_csv(out_top10, index=False, encoding='utf-8')

    diag = {
        "n_obs": int(df.shape[0]),
        "teams": int(len(teams)),
        "baseline_opp_def_pair": str(baseline_opp),
        "n_boot_target": int(args.n_boot),
        "n_boot_success": int(boot_arr.shape[0]),
        "attempts": int(attempts),
        "seed": int(args.seed),
        "failures": failures,
        "full_fit_aic": float(res_full.aic),
        "full_fit_deviance": float(getattr(res_full, "deviance", float("nan"))),
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    diag_path.write_text(json.dumps(diag, ensure_ascii=False, indent=2), encoding='utf-8')

    rationale = (
        "We use bootstrap re-fitting of the same regression (opponent-defense controls + log(TOI) offset) to quantify uncertainty "
        "in each team's line-disparity ratio. In each replicate, we resample within teamxoff_line groups so the team/line structure is preserved "
        "while sampling variability is reflected. The disparity ratio is computed under a fixed baseline opponent-denfense condition (first_def) "
        "by taking the difference in predicted log rates between first-line and second-line offense, which holds opponent-defense effects constant. "
        "The 95% CI indicates whether a team's disparity ranking is stable (narrow CI) or sensitive to noise (wide CI). "
        "This provides a reproducible robustness check alongside STEP08 (simple adjustment) and STEP15 (regression-based estimate)."
    )
    rationale_path.write_text(rationale + "\n", encoding="utf-8")

    print(f"teams: {len(teams)} (expected 32) | bootstrap_success={boot_arr.shape[0]}/{args.n_boot}")
    print(f"saved: {out_all} | bytes={out_all.stat().st_size}")
    print(f"saved: {out_top10} | bytes={out_top10.stat().st_size}")
    print(f"saved: {diag_path} | bytes={diag_path.stat().st_size}")
    print(f"saved: {rationale_path} | bytes={rationale_path.stat().st_size}")

    print("\n--- TOP 10 (point estimate, with CI) ---")
    top10 = out.head(10).copy()
    top10["ci"] = top10["ci95_lo"].map(lambda x: f"{x:.3f}") + "-" + top10["ci95_hi"].map(lambda x: f"{x:.3f}")
    print(top10[["team", "disparity_ratio_point", "ci", "first_off_xg60_point", "second_off_xg60_point"]].to_string(index=False))

    print("\n=== DoD (PASS CRITERIA) ===")
    print(f"teams==32: {len(teams)}==32")
    print(f"bootstrap_success>=min_success: {boot_arr.shape[0] >= args.min_success}")
    print("OVERALL: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())