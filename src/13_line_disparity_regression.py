"""
Regression-based line disparity.

Fits a Poisson GML (with team/oppponent effects and exposure) to estimate first-vs-second line effects.
Writes out/line_disparity_regression_all.csv and out/line_disparity_regression_top10.csv.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Dict, Tuple

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
except Exception as e:
    raise ImportError(
        "statsmodels is required for this step. Install via: pip install statsmodels"
    ) from e


ALLOWED_OFF_LINES = {"first_off", "second_off"}
ALLOWED_DEF_PAIR = {"first_def", "second_def"}

USECOLS = [
    "game_id",
    "home_team",
    "away_team",
    "home_off_line",
    "away_off_line",
    "home_def_pairing",
    "away_def_pairing",
    "toi",
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


def _pick_first_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _get_paths(project_root: Path) -> Dict[str, Path]:
    rc = _read_run_config(project_root)
    paths = rc.get("paths", {})

    data_raw_dir = Path(paths.get("data_raw_dir", project_root / "data_raw")).resolve()
    out_dir = Path(paths.get("out_dir", project_root / "out")).resolve()
    reports_dir = Path(paths.get("reports_dir", project_root / "reports")).resolve()

    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    expected = rc.get("expected_files", {})
    whl_name = expected.get("whl_csv", "whl_2025.csv")
    whl_csv_path = (data_raw_dir / whl_name).resolve()
    if not whl_csv_path.exists():
        alt = (data_raw_dir / "whl_2025.csv").resolve()
        if alt.exists():
            whl_csv_path = alt
        else:
            raise FileNotFoundError(f"whl_2025.csv not found in: {data_raw_dir}")

    # Optional baseline comparisons
    baseline_adjusted = _pick_first_existing(
        out_dir / "line_disparity_adjusted_all.csv",
        (out_dir / "line_disparity_adjusted_all.csv").with_name("line_disparity_adjusted_all.csv"),
    )
    baseline_simple = _pick_first_existing(
        out_dir / "line_disparity.csv",
        out_dir / "line_disparity_all.csv",
    )

    return {
        "whl_csv_path": whl_csv_path,
        "out_dir": out_dir,
        "reports_dir": reports_dir,
        "baseline_adjusted": baseline_adjusted if baseline_adjusted else Path(""),
        "baseline_simple": baseline_simple if baseline_simple else Path(""),
    }


def _to_long_offense(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build observation-level offense rows:
      - home offense vs away defense pairing
      - away offense vs home defense pairing
    Each observation has (team, off_line, opp_team, opp_def, xg, toi).
    """
    base = df.copy()

    base["toi"] = pd.to_numeric(base["toi"], errors="coerce")
    base["home_xg"] = pd.to_numeric(base["home_xg"], errors="coerce")
    base["away_xg"] = pd.to_numeric(base["away_xg"], errors="coerce")

    home = pd.DataFrame(
        {
            "team": base["home_team"].astype(str),
            "off_line": base["home_off_line"].astype(str),
            "opp_team": base["away_team"].astype(str),
            "opp_def": base["away_def_pairing"].astype(str),
            "xg": base["home_xg"],
            "toi": base["toi"],
        }
    )
    away = pd.DataFrame(
        {
            "team": base["away_team"].astype(str),
            "off_line": base["away_off_line"].astype(str),
            "opp_team": base["home_team"].astype(str),
            "opp_def": base["home_def_pairing"].astype(str),
            "xg": base["away_xg"],
            "toi": base["toi"],
        }
    )

    obs = pd.concat([home, away], ignore_index=True)

    # filter to even-strength top-2 lines only
    obs = obs[obs["off_line"].isin(ALLOWED_OFF_LINES)].copy()
    obs = obs[obs["opp_def"].isin(ALLOWED_DEF_PAIR)].copy()

    # remove invalid exposure
    obs = obs[obs["toi"].notna() & (obs["toi"] > 0)].copy()
    obs = obs[obs["xg"].notna() & (obs["xg"] >= 0)].copy()

    obs["is_first"] = (obs["off_line"] == "first_off").astype(int)
    obs["opp_def_second"] = (obs["opp_def"] == "second_def").astype(int)

    return obs


def _fit_poisson_glm(obs: pd.DataFrame, teams: list[str]) -> Tuple[sm.GLM, sm.GLMResults, dict]:
    """
    Poisson GLM with log link and log(toi) offset:

      log(E[xg]) = const
                  + team_offense_FE
                  + opp_team_FE
                  + opp_def_second
                  + is_first (baseline team first-vs-second)
                  + is_first * team_offense_FE  (team-specific line deltas)
                  + offset(log(toi))

    This estimates per-team first-vs-second effect without hand-tuned weights.
    """
    obs = obs.copy().reset_index(drop=True)
    obs["team"] = pd.Categorical(obs["team"].astype(str), categories=teams)
    obs["opp_team"] = pd.Categorical(obs["opp_team"].astype(str), categories=teams)

    xg = pd.to_numeric(obs["xg"], errors="coerce").astype(np.float64)
    toi = pd.to_numeric(obs["toi"], errors="coerce").astype(np.float64)

    is_first = (obs["off_line"].astype(str) == "first_off").astype(np.float64)
    opp_def_second = (obs["opp_def"].astype(str) == "second_def").astype(np.float64)

    n = len(obs)
    X = pd.DataFrame(
        {
            "const": np.ones(n, dtype=np.float64),
            "opp_def_second": opp_def_second.to_numpy(dtype=np.float64),
            "is_first": is_first.to_numpy(dtype=np.float64),
        },
        index=obs.index,
    )

    # Offense team fixed effects (drop first as baseline)
    team_d = pd.get_dummies(obs["team"], prefix="team", drop_first=True).astype(np.float64)

    # Opponent team fixed effects (drop first as baseline)
    opp_d = pd.get_dummies(obs["opp_team"], prefix="opp", drop_first=True).astype(np.float64)

    X = pd.concat([X, team_d, opp_d], axis=1)

    # interactions: is_first * team_dummies
    is_first_arr = is_first.to_numpy(dtype=np.float64)
    for col in team_d.columns:
        X[f"first_x_{col}"] = team_d[col].to_numpy(dtype=np.float64) * is_first_arr

    offset = np.log(np.clip(toi.to_numpy(dtype=np.float64), 1e-6, None))
    X_num = X.astype(np.float64)

    mask = np.isfinite(xg.to_numpy()) & np.isfinite(offset)
    mask &= np.isfinite(X_num.to_numpy()).all(axis=1)

    if not np.all(mask):
        X_num = X_num.loc[mask].copy()
        xg = xg.loc[mask].copy()
        offset = offset[mask]

    model = sm.GLM(xg.to_numpy(dtype=np.float64), X_num, family=sm.families.Poisson(), offset=offset)
    res = model.fit(cov_type="HC0")  # robust SE

    null_dev = getattr(res, "null_deviance", None)
    dev = float(res.deviance)
    pseudo_r2 = None
    if null_dev is not None and null_dev > 0:
        pseudo_r2 = float(1.0 - dev / float(null_dev))

    pearson_chi2 = float(res.pearson_chi2) if hasattr(res, "pearson_chi2") else None
    df_resid = float(res.df_resid) if hasattr(res, "df_resid") else None
    dispersion = None
    if pearson_chi2 is not None and df_resid and df_resid > 0:
        dispersion = float(pearson_chi2 / df_resid)

    diag = {
        "n_obs": int(len(X_num)),
        "teams": int(len(teams)),
        "aic": float(res.aic) if hasattr(res, "aic") else None,
        "deviance": dev,
        "null_deviance": float(null_dev) if null_dev is not None else None,
        "pseudo_r2_deviance": pseudo_r2,
        "pearson_chi2": pearson_chi2,
        "df_resid": df_resid,
        "dispersion": dispersion,
        "baseline_team": teams[0],
    }
    return model, res, diag


def _get_coef(res: sm.GLMResults, name: str) -> float:
    return float(res.params[name]) if name in res.params.index else 0.0


def _predict_team_line_xg60(
    res: sm.GLMResults,
    teams: list[str],
    weights: pd.DataFrame,
    team: str,
    is_first: int,
) -> float:
    """
    Compute xG/60 for a given team+line by averaging predicted xG rate over empirical joint
    distribution of (opp_team, opp_def) using TOI weights.

    In this GLM, exp(eta) = expected xG per second (because offset is log(toi_seconds)).
    So xG60 = 3600 * sum_w exp(eta(team,line,opp,def)).
    """
    baseline = teams[0]

    beta0 = _get_coef(res, "const")
    beta_def2 = _get_coef(res, "opp_def_second")
    beta_first_base = _get_coef(res, "is_first")

    team_eff = 0.0 if team == baseline else _get_coef(res, f"team_{team}")

    # team-specific first-line delta (interaction)
    if team == baseline:
        first_delta = beta_first_base
    else:
        first_delta = beta_first_base + _get_coef(res, f"first_x_team_{team}")

    line_delta = first_delta if is_first == 1 else 0.0

    total = 0.0
    for _, r in weights.iterrows():
        opp_team = r["opp_team"]
        opp_def = r["opp_def"]
        w = float(r["w"])

        opp_eff = 0.0 if opp_team == baseline else _get_coef(res, f"opp_{opp_team}")
        def_eff = beta_def2 if opp_def == "second_def" else 0.0

        eta = beta0 + team_eff + opp_eff + def_eff + line_delta
        total += w * np.exp(eta)

    xg60 = 3600.0 * total
    return float(xg60)


def main() -> int:
    project_root = _resolve_project_root()
    p = _get_paths(project_root)

    whl_csv_path = p["whl_csv_path"]
    out_dir = p["out_dir"]
    reports_dir = p["reports_dir"]
    baseline_adjusted = p["baseline_adjusted"]
    baseline_simple = p["baseline_simple"]

    df = pd.read_csv(whl_csv_path, usecols=USECOLS)
    obs = _to_long_offense(df)

    teams = sorted(set(obs["team"].astype(str).unique().tolist()))
    if len(teams) != 32:
        # still proceed, but DoD will fail
        pass

    # Fit GLM
    _, res, diag = _fit_poisson_glm(obs, teams)

    # Empirical joint weights of (opp_team, opp_def) using TOI
    wtbl = (
        obs.groupby(["opp_team", "opp_def"], as_index=False)
        .agg(toi_sum=("toi", "sum"))
        .copy()
    )
    tot = float(wtbl["toi_sum"].sum())
    wtbl["w"] = wtbl["toi_sum"] / tot if tot > 0 else 0.0
    wtbl = wtbl[["opp_team", "opp_def", "w"]].copy()

    # Predict regression-based xG60 and ratio
    rows = []
    for team in teams:
        xg60_first = _predict_team_line_xg60(res, teams, wtbl, team, is_first=1)
        xg60_second = _predict_team_line_xg60(res, teams, wtbl, team, is_first=0)
        ratio = (xg60_first / xg60_second) if xg60_second > 0 else np.nan
        rows.append(
            {
                "team": team,
                "first_off_xg60_reg": xg60_first,
                "second_off_xg60_reg": xg60_second,
                "disparity_ratio_reg": ratio,
            }
        )

    out_all = pd.DataFrame(rows).sort_values("disparity_ratio_reg", ascending=False).reset_index(drop=True)
    out_all["reg_rank"] = np.arange(1, len(out_all) + 1)

    out_top10 = out_all.head(10).copy()

    out_all_path = out_dir / "line_disparity_regression_all.csv"
    out_top10_path = out_dir / "line_disparity_regression_top10.csv"
    out_all.to_csv(out_all_path, index=False)
    out_top10.to_csv(out_top10_path, index=False)

    diag["weights_joint_opp_team_def_sum"] = float(wtbl["w"].sum())
    diag["weights_preview"] = wtbl.sort_values("w", ascending=False).head(10).to_dict(orient="records")

    diag_path = reports_dir / "line_disparity_regression_diagnostics.json"
    diag_path.write_text(json.dumps(diag, indent=2), encoding="utf-8")

    # Rationale text (for 1d)
    rationale = (
        "We estimated first-line vs second-line offensive disparity using a Poisson GLM with a log link and a log(TOI) offset, "
        "so xG is modeled as a rate rather than raw totals. The regression includes offense-team fixed effects and opponent-team "
        "fixed effects, plus an opponent defense-pairing indicator (first_def vs second_def), to control for matchup difficulty "
        "without any hand-tuned weights. Team-specific first-line advantages are identified via interactions between team and the "
        "first-line indicator. We convert fitted coefficients to xG/60 and report a regression-based first/second ratio per team "
        "by averaging predictions over the empirical distribution of opponent team and defense pairing."
    )
    rat_path = reports_dir / "line_disparity_regression_rationale.txt"
    rat_path.write_text(rationale.strip() + "\n", encoding="utf-8")

    # Optional comparison vs adjusted baseline
    compare_path = reports_dir / "line_disparity_compare_adjusted_vs_regression.csv"
    overlap = None
    if baseline_adjusted and baseline_adjusted.exists():
        base = pd.read_csv(baseline_adjusted)
        # expected column in your pipeline: disparity_ratio_adj
        if "disparity_ratio_adj" in base.columns:
            base = base[["team", "disparity_ratio_adj"]].copy()
            base = base.sort_values("disparity_ratio_adj", ascending=False).reset_index(drop=True)
            base["adjusted_rank"] = np.arange(1, len(base) + 1)

            merged = out_all.merge(base, on="team", how="left")
            merged.to_csv(compare_path, index=False)

            top10_reg = set(out_top10["team"].tolist())
            top10_adj = set(base.head(10)["team"].tolist())
            overlap = len(top10_reg.intersection(top10_adj))

    teams_count = len(out_all)
    nan_count = int(out_all["disparity_ratio_reg"].isna().sum())

    print("=== STEP15 LINE DISPARITY (REGRESSION: opponent controls, no hand-tuned weights) ===")
    print(f"input:     {whl_csv_path.resolve()}")
    if baseline_adjusted and baseline_adjusted.exists():
        print(f"baseline:  {baseline_adjusted.resolve()} (for comparison)")
    elif baseline_simple and baseline_simple.exists():
        print(f"baseline:  {baseline_simple.resolve()} (for reference)")
    else:
        print("baseline:  (none found)")

    print(f"output:    {out_all_path.resolve()}")
    print(f"output:    {out_top10_path.resolve()}")
    print(f"diag:      {diag_path.resolve()}")
    print(f"rationale: {rat_path.resolve()}")
    if compare_path.exists():
        print(f"compare:   {compare_path.resolve()}")

    print(f"\nmodel: Poisson GLM + log(TOI) offset | n_obs={diag['n_obs']} | AIC={diag['aic']:.2f} | deviance={diag['deviance']:.2f} | dispersion={diag.get('dispersion')}")
    print(f"teams in output: {teams_count} (expected 32) | NaN ratios: {nan_count}")

    print("\n--- TOP 10 by disparity_ratio_reg ---")
    show = out_top10[["team", "disparity_ratio_reg", "first_off_xg60_reg", "second_off_xg60_reg", "reg_rank"]].copy()
    print(show.to_string(index=False))

    if overlap is not None:
        print(f"\nadjusted vs regression TOP10 overlap: {overlap}/10")

    # DoD
    dod_ok = (teams_count == 32) and (nan_count == 0) and (len(out_top10) == 10)
    print("\n=== DoD (PASS CRITERIA) ===")
    print(f"teams==32: {teams_count == 32}")
    print(f"no NaN ratios: {nan_count == 0}")
    print(f"top10 rows==10: {len(out_top10) == 10}")
    print(f"\nOVERALL: {'PASS' if dod_ok else 'FAIL'}")
    return 0 if dod_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
