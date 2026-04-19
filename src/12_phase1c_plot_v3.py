"""
Phase 1C visualization (v3).

Final plotting variant used after calibration/regression steps.
Writes figure files udner out/phase1c_*.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


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


def main() -> int:
    project_root = _resolve_project_root()
    rc = _read_run_config(project_root)
    paths = rc.get("paths", {})

    out_dir = Path(paths.get("out_dir", project_root / "out")).resolve()
    reports_dir = Path(paths.get("reports_dir", project_root / "out")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    strength_path = out_dir / "power_rankings_v3_offset.csv"
    if not strength_path.exists():
        raise FileNotFoundError(f"Missing: {strength_path} (run STEP12)")
    
    # prefer adjusted disparity if exists
    disp_path = out_dir / "line_disparity_adjusted_all.csv"
    disp_col = "disparity_ratio_adj"
    if not disp_path.exists():
        disp_path = out_dir / "line_disparity.csv"
        disp_col = "disparity_ratio"
        if not disp_path.exists():
            raise FileNotFoundError("Missing line disparity file (run STEP07/08).")
    
    strength = pd.read_csv(strength_path)
    disp = pd.read_csv(disp_path)

    # expected cols
    if "team" not in strength.columns or "strength_score" not in strength.columns:
        raise ValueError("power_rankings_v3_offset.csv must include columns: team, strength_score")
    if "team" not in disp.columns or disp_col not in disp.columns:
        raise ValueError(f"disparity file must include columns: team, {disp_col}")
    
    m =pd.merge(
        disp[["team", disp_col]].copy(),
        strength[["team", "strength_score"]].copy(),
        on="team",
        how="inner",
    )
    if len(m) != 32:
        raise ValueError(f"Joined rows should be 32, got {len(m)}")
    
    # mean-center strength for interpretability
    m["strength_centered"] = m["strength_score"] - m["strength_score"].mean()

    rho, p = spearmanr(m[disp_col].to_numpy(), m["strength_score"].to_numpy())

    # label subset: top/bottom by disparity and strength + one residual outlier
    m["disp_rank"] = m[disp_col].rank(ascending=False, method="min")
    m["str_rank"] = m["strength_score"].rank(ascending=False, method="min")

    x = m[disp_col].to_numpy()
    y = m["strength_centered"].to_numpy()
    A = np.vstack([x, np.ones_like(x)]).T
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    resid = np.abs(y - yhat)
    outlier_team = m.iloc[int(np.argmax(resid))]["team"]

    label_mask = (
        (m["disp_rank"] <= 5)
        | (m["disp_rank"] >= 28)
        | (m["str_rank"] <= 5)
        | (m["str_rank"] >= 28)
        | (m["team"] == outlier_team)
    )

    plt.figure(figsize=(12, 8))
    plt.scatter(m[disp_col], m["strength_centered"])

    plt.axvline(1.0, linestyle="--")
    plt.axhline(0.0, linestyle="--")

    for _, r in m[label_mask].iterrows():
        plt.text(r[disp_col], r["strength_centered"], str(r["team"]), fontsize=10)

    plt.title(
        "Phase 1c - Line disparity vs Team strength (v3 offset)\n"
        f"Disparity: {disp_path.name} | Spearman rho={rho:.3f}, p={p:.3f} | N=32"
    )
    plt.xlabel(f"Line disparity (adjusted if available): {disp_col}  (ratio; 1.0 = equal)")
    plt.ylabel("Team strength (v3) centered: strength_score - means(strength_score)")

    plt.figtext(
        0.5,
        0.01,
        "All teams are plotted; labels show extremes by disparity/strength plus the largest residual outlier."
        "Strength is mean-centered so y=0 represents league-average strength (v3 score is relative by constructing).",
        ha="center",
        fontsize=10,
    )

    out_plot = out_dir / "phase1c_line_disparity_vs_strength_v3.png"
    out_submit = out_dir / "WHSDSC_2026.png"
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_plot, dpi=150)
    plt.savefig(out_submit, dpi=150)
    plt.close()

    b1 = out_plot.stat().st_size
    b2 = out_submit.stat().st_size

    interp = (
        f"Using v3 (TOI-offset) Poisson attack/defense strength and "
        f"{disp_path.name} disparity, Spearman rho={rho:.3f} (p={p:.3f}) suggests little monotinic association."
        f"Strength is mean-centered for interpretability (y=0 = league.average)."
        f"We label extremes and the largest residual outlier to highlight teams that do not follow the overall trend.\n"
    )
    (reports_dir / "phase1c_interpretation_v3.txt").write_text(interp, encoding="utf-8")

    print("=== STEP14 PHASE 1C PLOT (V3 OFFSET) ===")
    print(f"strength: {strength_path.resolve()}")
    print(f"disparity: {disp_path.resolve()} (using {disp_col})")
    print(f"joined rows: {len(m)} (expected 32)")
    print(f"saved: {out_plot.resolve()} | bytes={b1}")
    print(f"saved: {out_submit.resolve()} | bytes={b2}")
    print(f"saved: {(reports_dir / 'phase1c_interpretation_v3.txt').resolve()}")
    print(f"spearman_rho={rho:.6f} | p={p:.6f}")
    print(f"labels_shown: {int(label_mask.sum())} teams")
    ok = (len(m) == 32) and (b1 < 5_000_000) and (b2 < 5_000_000)
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
