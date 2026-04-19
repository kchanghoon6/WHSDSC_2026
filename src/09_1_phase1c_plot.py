"""
Phase 1C visualization (v1).

Creates the primary plot(s) used in the phase 1C write-up using previously generated CSV outputs.
Writes figure files under out/phase1c_*.png.
"""

from __future__ import annotations

import json
from io import BytesIO
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


def _sanitize_name(s: str) -> str:
    return "".join(ch for ch in str(s).strip().replace(" ", "_") if ch.isalnum() or ch in ("_", "-"))


def _get_team_tag(rc: dict) -> str:
    # Try a few plausible keys; fallback is deterministic
    for key in ["submission_team_name", "team_name", "TEAM_NAME"]:
        if key in rc and rc[key]:
            return _sanitize_name(rc[key])
    sub = rc.get("submission", {})
    if isinstance(sub, dict) and sub.get("team_name"):
        return _sanitize_name(sub["team_name"])
    return "WHSDSC_2025"


def _save_png_under_limit(fig, path: Path, max_bytes: int = 5 * 1024 * 1024) -> int:
    """
    Save fig as PNG under max_bytes by reducing DPI if needed.
    Returns final file size in bytes,
    """
    dpi = 220
    min_dpi = 90

    while True:
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        size = buf.tell()
        if size <= max_bytes or dpi <= min_dpi:
            path.write_bytes(buf.getvalue())
            return path.stat().st_size
        dpi = max(min_dpi, int(dpi * 0.85))

    
def main() -> int:
    project_root = _resolve_project_root()
    rc = _read_run_config(project_root)

    paths = rc.get("paths", {})
    out_dir = Path(paths.get("out_dir", project_root / "out")).resolve()
    reports_dir = Path(paths.get("reports_dir", project_root / "reports")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    strength_path = out_dir / "power_rankings_v2_model.csv"
    if not strength_path.exists():
        raise FileNotFoundError(f"Missing: {strength_path} (run STEP05)")
    
    # Prefer adjusted disparity if present
    disp_adjusted = out_dir / "line_disparity_adjusted_all.csv"
    disp_baseline = out_dir / "line_disparity.csv"

    if disp_adjusted.exists():
        disp_path = disp_adjusted
        disp_col = "disparity_ratio_adj"
        disp_tag = "adjusted"
    elif disp_baseline.exists():
        disp_path = disp_baseline
        disp_col = "disparity_ratio"
        disp_tag=  "baseline"
    else:
        raise FileNotFoundError("No disparity file found. Expected out/line_disparity_adjusted_all.csv or out/line_disparity.csv")
    
    strength = pd.read_csv(strength_path)
    disp = pd.read_csv(disp_path)

    if "team" not in strength.columns or "strength_score" not in strength.columns:
        raise ValueError("power_ranking_v2_model.csv must contain columns: team, strength_score")

    if "team" not in disp.columns or disp_col not in disp.columns:
        raise ValueError(f"{disp_path.name} must contain columns: team, {disp_col}")
    
    s = strength[["team", "strength_score"]].copy()
    d = disp[["team", disp_col]].copy()

    merged = s.merge(d, on="team", how="inner")
    join_rows = len(merged)

    s_teams = set(s["team"].astype(str))
    d_teams = set(d["team"].astype(str))
    only_s = sorted(list(s_teams - d_teams))
    only_d = sorted(list(d_teams - s_teams))

    x = pd.to_numeric(merged[disp_col], errors="coerce")
    y = pd.to_numeric(merged["strength_score"], errors="coerce")
    ok = x.notna() & y.notna()
    rho, pval = spearmanr(x[ok], y[ok]) if ok.sum() >= 3 else (np.nan, np.nan)

    # Identify a couple of notable teams for interpretaion
    merged2 = merged.copy()
    merged2["x"] = x
    merged2["y"] = y
    merged2 = merged2.dropna(subset=["x", "y"]).reset_index(drop=True)

    top_disp = merged2.sort_values("x", ascending=False).head(1).iloc[0]
    top_strength = merged2.sort_values("y", ascending=False).head(1).iloc[0]

    # Simple linear fit for "outlier" via residual (explainable)
    if len(merged2) >= 3:
        coef = np.polyfit(merged2["x"].to_numpy(), merged2["y"].to_numpy(), 1)
        yhat = coef[0] * merged2["x"].to_numpy() + coef[1]
        resid = merged2["y"].to_numpy() - yhat
        idx = int(np.argmax(np.abs(resid)))
        outlier = merged2.iloc[idx]
    else:
        outlier = top_disp

    # Plot
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    ax.scatter(merged2["x"], merged2["y"])

    # Optional reference lines (no explicit colors)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.axhline(1.0, linestyle="--", linewidth=1)

    # Label every point (small font to reduce clutter)
    for _, r in merged2.iterrows():
        ax.annotate(str(r["team"]), (r["x"], r["y"]), fontsize=7, xytext=(3, 2), textcoords="offset points")
    
    ax.set_xlabel(f"Line disparity ({disp_tag}): {disp_col}")
    ax.set_ylabel("Team strength (v2) strength_score (attack - defense_allow)")
    title = "Phase 1c - Line disparity vs Team strength"
    subtitle = f"Disparity source: {disp_path.name} | Spearman rho={rho:.3f}, p={pval:.3g} | N={len(merged2)}"
    ax.set_title(title + "\n" + subtitle)

    caption = (
        "Each point is a team. X uses first-line vs second-line xG/60 ratio (adjusted if available)."
        "Y uses Poisson GML attack/defense strength_score from Phase 1a (v2)."
    )
    plt.figtext(0.5, 0.01, caption, ha="center", fontsize=9)

    # Save output:
    work_png = out_dir / "phase1c_line_disparity_vs_strength.png"
    size_work = _save_png_under_limit(plt.gcf(), work_png)

    team_tag = _get_team_tag(rc)
    submit_png = out_dir / f"{team_tag}.png"
    submit_png.write_bytes(work_png.read_bytes())
    size_submit = submit_png.stat().st_size

    interp_lines = []
    interp_lines.append(
        f"We joined team strength (v2 Poisson GLM strength_score) with line disparity ({disp_tag}) for {join_rows} teams."
    )
    interp_lines.append(
        f"The association is Spearman rho={rho:.3f} (p={pval:.3g}), which indicates a {'positive' if rho > 0 else 'negative' if rho < 0 else 'near-zero'} monotonic relationship."
    )
    interp_lines.append(
        f"The largest line disparity is {top_disp['team']} with {disp_col}={top_disp['x']:.3f}, suggesting a bigger gap between first- and second-line xG/60."
    )
    interp_lines.append(
        f"The strongest team by v2 strength_score is {top_strength['team']} with strength_score={top_strength['y']:.3f}."
    )
    interp_lines.append(
        f"One notable deviation from the overall trend is {outlier['team']}, which sits far from the best-fit line given its disparity and strength."
    )
    interp_lines.append(
        "We chose disparity ratio on the scatterplot makes it easy to identify which teams drive the pattern and which teams behave unusually."
    )
    interp_lines.append(
        "Labeling teams on teh scatterplot makes it easy to identify which teams drive the pattern and which teams behave unusually."
    )

    interpretion_path = reports_dir / "phase1c_interpretation.txt"
    interpretion_path.write_text("\n".join(interp_lines) + "\n", encoding="utf-8")

    print("=== STEP09 PHASE 1C PLOT ===")
    print(f"strength: {strength_path.resolve()}")
    print(f"disparity: {disp_path.resolve()} (using {disp_col})")
    print(f"joined rows: {join_rows} (expected 32)")
    if join_rows != 32:
        print(f"teams_only in strength: {only_s[:10]}{' ...' if len(only_s) > 10 else ''}")
        print(f"teams_only in disparity: {only_d[:10]}{' ...' if len(only_d) > 10 else ''}")
    print(f"saved: {work_png.resolve()} | bytes={size_work}")
    print(f"saved: {submit_png.resolve()} | bytes={size_submit}")
    print(f"saved: {interpretion_path.resolve()}")
    print(f"spearman_rho={rho:.6f} | p={pval:.6g}")

    # DoD
    dod_ok = (join_rows == 32) and work_png.exists() and (size_work <= 5 * 1024 * 1024)
    print(f"\nOVERALL: {'PASS' if dod_ok else 'FAIL'}")
    return 0 if dod_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
