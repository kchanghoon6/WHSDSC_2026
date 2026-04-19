"""
Calibrated Round 1 home-win probabilities (v3).

Builds on earlier probability models and applies a calibration step before producing the submission CSV.
Writes out/round1_homewin_probs_v3.csv and overwrites out/round1_homewin_probs.csv.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import skellam


REQ_GAMES_COLS = ["game_id", "home_team", "away_team", "home_goals", "away_goals", "toi"]
REQ_MATCHUP_COLS = {"home_team", "away_team"}


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


def _canon(s: str) -> str:
    return str(s).strip().lower().replace(" ", "_")


def _skellam_home_win_prob(lam_home: float, lam_away: float) -> float:
    # P(H>A) + 0.5*P(tie)
    cdf0 = skellam.cdf(0, lam_home, lam_away)
    pmf0 = skellam.pmf(0, lam_home, lam_away)
    p = float((1.0 - cdf0) + 0.5 * pmf0)
    return max(0.0, min(1.0, p))


def _safe_log(x: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    return np.log(np.clip(x, eps, None))


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

    params_path = reports_dir / "poisson_glm_v3_offset_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Missing v3 params: {params_path} (run STEP12 fisrt)")

    seed = int(rc.get("seed", 2026))
    return {
        "rc": rc,
        "seed": seed,
        "games_path": games_path,
        "matchups_path": matchups_path,
        "out_dir": out_dir,
        "reports_dir": reports_dir,
        "params_path": params_path,
    }


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


def _compute_lambdas(
        intercept: float,
        home_beta: float,
        atk: dict,
        dfa: dict,
        home_team: np.ndarray,
        away_team: np.ndarray,
        toi_seconds: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    # if toi_seconds is None -> assume regulation 60min (1 hour exposure), i.e., log_exposure=0
    if toi_seconds is None:
        log_exp = 0.0
        lam_home = np.exp(intercept + home_beta + np.vectorize(atk.get)(home_team) + np.vectorize(dfa.get)(away_team) + log_exp)
        lam_away = np.exp(intercept + 0.0 + np.vectorize(atk.get)(away_team) + np.vectorize(dfa.get)(home_team) + log_exp)
        return lam_home.astype(float), lam_away.astype(float)
    
    toi_hours = np.clip(toi_seconds.astype(float) / 3600.0, 1e-9, None)
    log_exp = np.log(toi_hours)
    lam_home = np.exp(intercept + home_beta + np.vectorize(atk.get)(home_team) + np.vectorize(dfa.get)(away_team) + log_exp)
    lam_away = np.exp(intercept + 0.0 + np.vectorize(atk.get)(away_team) + np.vectorize(dfa.get)(home_team) + log_exp)
    return lam_home.astype(float), lam_away.astype(float)


def _platt_fit(x_logit: np.ndarray, y: np.ndarray) -> sm.GLM:
    X = sm.add_constant(x_logit.astype(float))
    m = sm.GLM(y.astype(float), X, family=sm.families.Binomial())
    return m.fit()


def _platt_predict(fit: sm.GLM, x_logit: np.ndarray) -> np.ndarray:
    X = sm.add_constant(x_logit.astype(float))
    return fit.predict(X)


def _metrics(p: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    eps = 1e-15
    brier = float(np.mean((p - y) ** 2))
    logloss = float(-np.mean(y * _safe_log(p, eps) + (1 - y) * _safe_log(1 - p, eps)))
    return brier, logloss


def main() -> int:
    project_root = _resolve_project_root()
    paths = _get_paths(project_root)
    np.random.seed(paths["seed"])

    # load v3 params
    params = json.loads(Path(paths["params_path"]).read_text(encoding="utf-8"))
    intercept = float(params["intercept"])
    home_beta = float(params["home_adv_beta_log"])
    atk = {str(k): float(v) for k, v in params["attack_coef"].items()}
    dfa = {str(k): float(v) for k, v in params["defense_allow_coef"].items()}

    # season games (for calibration decision)
    games = pd.read_csv(paths["games_path"], usecols=REQ_GAMES_COLS)
    for c in ["home_goals", "away_goals", "toi"]:
        games[c] = pd.to_numeric(games[c], errors="coerce")
    if games[["home_goals", "away_goals", "toi"]].isna().any().any():
        raise ValueError("Found NaN in home_goals/away_goals/toi in games_2025.csv")
    
    y = (games["home_goals"].to_numpy() > games["away_goals"].to_numpy()).astype(int)
    ties = int((games["home_goals"] == games["away_goals"]).sum())  # likely 0

    lam_home, lam_away = _compute_lambdas(
        intercept, home_beta, atk, dfa,
        games["home_team"].astype(str).to_numpy(),
        games["away_team"].astype(str).to_numpy(),
        games["toi"].to_numpy(dtype=float),
    )
    p_raw = np.array([_skellam_home_win_prob(a, b) for a, b in zip(lam_home, lam_away)], dtype=float)

    # Platt scaling feature: logit(p_raw)
    p_clip = np.clip(p_raw, 1e-6, 1 - 1e-6)
    x_logit = np.log(p_clip / (1 - p_clip))

    # 5-fold CV to decide whether to use calibrated probabilities
    n = len(games)
    idx = np.arange(n)
    np.random.shuffle(idx)
    k = 5
    folds = np.array_split(idx, k)

    raw_briers, raw_lls = [], []
    cal_briers, cal_lls = [], []

    for i in range(k):
        te = folds[i]
        tr = np.setdiff1d(idx, te)

        b_raw, ll_raw = _metrics(p_raw[te], y[te])
        raw_briers.append(b_raw)
        raw_lls.append(ll_raw)

        fit = _platt_fit(x_logit[tr], y[tr])
        p_te = _platt_predict(fit, x_logit[te])
        b_cal, ll_cal = _metrics(p_te, y[te])
        cal_briers.append(b_cal)
        cal_lls.append(ll_cal)

    cv = {
        "kfold": k,
        "ties": ties,
        "raw_brier_mean": float(np.mean(raw_briers)),
        "raw_logloss_mean": float(np.mean(raw_lls)),
        "cal_brier_mean": float(np.mean(cal_briers)),
        "cal_logloss_mean": float(np.mean(cal_lls)),
    }

    # Decision use calibrated only if it improves BOTH brier and logloss by a tiny margin
    use_cal = (cv["cal_brier_mean"] + 1e-6 < cv["raw_brier_mean"]) and (cv["cal_logloss_mean"] + 1e-6 < cv["raw_logloss_mean"])

    # Fit final calibrator on full data (if chosen)
    cal_params = None
    if use_cal:
        fit_full = _platt_fit(x_logit, y)
        cal_params = {"const": float(fit_full.params[0]), "slope": float(fit_full.params[1])}
    else:
        fit_full = None

    # Round 1 matchups
    matchups = _read_matchups(paths["matchups_path"])
    matchups["home_team"] = matchups["home_team"].astype(str)
    matchups["away_team"] = matchups["away_team"].astype(str)

    unknown = []
    for i, r in matchups.iterrows():
        ht, at = r["home_team"], r["away_team"]
        if ht not in atk or at not in atk:
            unknown.append((r.get("game_id", f"row_{i}"), ht, at))
    if unknown:
        msg = "\n".join([f"{gid}: {ht} vs {at}" for gid, ht, at in unknown[:10]])
        raise ValueError("Some matchup teams not in v3 coefficient map.\nFirst examples\n" + msg)
    
    lam_h_r1, lam_a_r1 = _compute_lambdas(
        intercept, home_beta, atk, dfa,
        matchups["home_team"].to_numpy(),
        matchups["away_team"].to_numpy(),
        None,  # regulation baseline
    )
    p_raw_r1 = np.array([_skellam_home_win_prob(a, b) for a, b in zip(lam_h_r1, lam_a_r1)], dtype=float)
    
    if use_cal:
        p_clip_r1 = np.clip(p_raw_r1, 1e-6, 1 - 1e6)
        x_r1 = np.log(p_clip_r1 / (1 - p_clip_r1))
        p_cal_r1 = _platt_predict(fit_full, x_r1)
        p_submit = p_cal_r1
        method_tag = "poisson_glm_offset_v3_platt_calibrated"
    else:
        p_cal_r1 = np.full_like(p_raw_r1, np.nan, dtype=float)
        p_submit = p_raw_r1
        method_tag = "poisson_glm_offset_v3_raw"

    out = pd.DataFrame(
        {
            "game_id": matchups["game_id"] if "game_id" in matchups.columns else [f"game_{i+1}" for i in range(len(matchups))],
            "home_team": matchups["home_team"],
            "away_team": matchups["away_team"],
            "p_home_win": p_submit,
            "p_home_win_raw": p_raw_r1,
            "p_home_win_calibrated": p_cal_r1,
            "lambda_home": lam_h_r1,
            "lambda_away": lam_a_r1,
            "method_tag": method_tag,
        }
    )

    out_dir = Path(paths["out_dir"])
    reports_dir = Path(paths["reports_dir"])

    out_v3 = out_dir / "round1_homewin_probs_v3.csv"
    out_submit = out_dir / "round1_homewin_probs.csv"
    out.to_csv(out_v3, index=False)

    # submission file uses p_home_win column + core fields
    submit = out[["game_id", "home_team", "away_team", "p_home_win", "lambda_home", "lambda_away", "method_tag"]].copy()
    submit.to_csv(out_submit, index=False)
    
    report = {
        "model_tag": params["model_tag"],
        "cv": cv,
        "use_calibration": bool(use_cal),
        "calibration_params": cal_params,
    }
    (reports_dir / "round1_v3_calibration_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    note = (
        "We evaluated Platt scaling (logit calibration) using 5-fold CV on 2025 season games.\n"
        f"Calibration used: {use_cal}\n"
        f"Raw CV brier/logloss: {cv['raw_brier_mean']:.6f} / {cv['raw_logloss_mean']:.6f}\n"
        f"Cal CV brier/logloss: {cv['cal_brier_mean']:.6f} / {cv['cal_logloss_mean']:.6f}\n"
        "Round 1 probabilities use the chosen method and assume regulation-length exposure (60 minutes).\n"
    )
    (reports_dir / "round1_v3_method_note.txt").write_text(note, encoding="utf-8")

    # DoD checks
    nrows = len(submit)
    in_range = bool(((submit["p_home_win"] >= 0) & (submit["p_home_win"] <= 1)).all())
    no_nan = bool(submit["p_home_win"].notna().all())
    row_ok = (nrows == 16)

    print("=== STEP13 ROUND 1 HOME WIN PROBS (V3: Offset Poisson + optional calibration) ===")
    print(f"params:  {Path(paths['params_path']).resolve()}")
    print(f"matchups: {Path(paths['matchups_path']).resolve()}")
    print(f"games:   {Path(paths['games_path']).resolve()}")
    print(f"output(v3):  {out_v3.resolve()}")
    print(f"output(submit): {out_submit.resolve()}")
    print(f"cal_report: { (reports_dir / 'round1_v3_calibration_report.json').resolve()}")
    print(f"home_adv_beta(log)={home_beta:.6f} | exp(beta)={np.exp(home_beta):.6f} | intercept={intercept:.6f}")
    print(f"cv_raw(brier/logloss)={cv['raw_brier_mean']:.6f}/{cv['raw_logloss_mean']:.6f} | "
        f"cv_cal(brier/logloss)={cv['cal_brier_mean']:.6f}/{cv['cal_logloss_mean']:.6f} | use_cal={use_cal}")
    print(f"rows: {nrows} (expected 16) | in_range: {in_range} | no_nan: {no_nan}")

    print("\n--- CSV (copy/paste) ---")
    print(submit.to_csv(index=False).strip())

    overall = row_ok and in_range and no_nan
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
