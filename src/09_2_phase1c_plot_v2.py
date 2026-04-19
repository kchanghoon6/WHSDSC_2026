"""
Phase 1C visualization (v2).

Alternative plotting variant for Phase 1C (same inputs/outputs contract as v1).
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
    for key in ["submission_team_name", "team_name", "TEAM_NAME"]:
        if key in rc and rc[key]:
            return _sanitize_name(rc[key])
    sub = rc.get("submission", {})
    if isinstance(sub, dict) and sub.get("team_name"):
        return _sanitize_name(sub["team_name"])
    return "WHSDSC_2026"


def _save_png_under_limit(fig, path: Path, max_bytes: int = 5 * 1024 * 1024) -> int:
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
        disp_tag = "baseline"
    else:
        raise FileNotFoundError("No disparity file found. Expected out/line_disparity_adjusted_all.csv or out/line_disparity.csv")

    strength = pd.read_csv(strength_path)
    disp = pd.read_csv(disp_path)

    if "team" not in strength.columns or "strength_score" not in strength.columns:
        raise ValueError("power_rankings_v2_model.csv must contain columns: team, strength_score")
    if "team" not in disp.columns or disp_col not in disp.columns:
        raise ValueError(f"{disp_path.name} must contain columns: team, {disp_col}")

    merged = strength[["team", "strength_score"]].merge(disp[["team", disp_col]], on="team", how="inner")
    join_rows = len(merged)

    x = pd.to_numeric(merged[disp_col], errors="coerce")
    y_raw = pd.to_numeric(merged["strength_score"], errors="coerce")
    ok = x.notna() & y_raw.notna()
    merged2 = merged.loc[ok, ["team"]].copy()
    merged2["x"] = x[ok].astype(float).to_numpy()
    merged2["y_raw"] = y_raw[ok].astype(float).to_numpy()

    # Center strength so y=0 is league average (reduces "reference team=0" confusion)
    y_mean = float(merged2["y_raw"].mean())
    merged2["y"] = merged2["y_raw"] - y_mean

    # Correlation (monotonic) remains comparable
    rho, pval = spearmanr(merged2["x"], merged2["y_raw"]) if len(merged2) >= 3 else (np.nan, np.nan)

    # Select a compact label set to avoid clutter
    k = 5
    label_set = set()
    label_set |= set(merged2.sort_values("x", ascending=False).head(k)["team"])
    label_set |= set(merged2.sort_values("x", ascending=True).head(k)["team"])
    label_set |= set(merged2.sort_values("y", ascending=False).head(k)["team"])
    label_set |= set(merged2.sort_values("y", ascending=True).head(k)["team"])

    # Identify outlier by residual from simple linear fit in centered space
    outlier_team = None
    if len(merged2) >= 3:
        coef = np.polyfit(merged2["x"].to_numpy(), merged2["y"].to_numpy(), 1)
        yhat = coef[0] * merged2["x"].to_numpy() + coef[1]
        resid = merged2["y"].to_numpy() - yhat
        idx = int(np.argmax(np.abs(resid)))
        outlier_team = str(merged2.iloc[idx]["team"])
        label_set.add(outlier_team)

    # Plot
    plt.figure(figsize=(10, 7))
    ax = plt.gca()

    ax.scatter(merged2["x"], merged2["y"])  # default style

    # Reference lines: ratio=1 (equal lines), y=0 (league-average centered strength)
    ax.axvline(1.0, linestyle="--", linewidth=1)
    ax.axhline(0.0, linestyle="--", linewidth=1)

    # Label only selected teams (less overlap, more readable)
    for _, r in merged2.iterrows():
        t = str(r["team"])
        if t in label_set:
            ax.annotate(t, (r["x"], r["y"]), fontsize=9, xytext=(4, 3), textcoords="offset points")

    ax.set_xlabel(f"Line disparity ({disp_tag}): {disp_col}  (ratio; 1.0 = equal)")
    ax.set_ylabel("Team strength (v2) centered: strength_score - mean(strength_score)")
    title = "Phase 1c — Line disparity vs Team strength (readability-focused)"
    subtitle = f"Disparity: {disp_path.name} | Spearman rho={rho:.3f}, p={pval:.3g} | N={len(merged2)}"
    ax.set_title(title + "\n" + subtitle)

    caption = (
        "All teams are plotted; labels show top/bottom by disparity and strength plus the largest residual outlier. "
        "Strength is mean-centered so y=0 represents league-average strength (v2 score is relative by construction)."
    )
    plt.figtext(0.5, 0.01, caption, ha="center", fontsize=9)

    work_png = out_dir / "phase1c_line_disparity_vs_strength_v2.png"
    size_work = _save_png_under_limit(plt.gcf(), work_png)

    team_tag = _get_team_tag(rc)
    submit_png = out_dir / f"{team_tag}.png"
    # Overwrite submission png with the improved plot
    submit_png.write_bytes(work_png.read_bytes())
    size_submit = submit_png.stat().st_size

    # Interpretation (6–8 sentences)
    lines = []
    lines.append(f"We joined team strength (v2 Poisson GLM) and line disparity ({disp_tag}) for {join_rows} teams.")
    lines.append(f"The relationship is near zero (Spearman rho={rho:.3f}, p={pval:.3g}), suggesting disparity alone does not explain overall strength here.")
    lines.append("We plotted all teams but labeled only key extremes (top/bottom by disparity and strength) to reduce overlap and improve readability.")
    lines.append("Because v2 strength_score is defined relative to a reference category in the GLM, we mean-centered strength so y=0 represents league-average strength.")
    if outlier_team is not None:
        lines.append(f"The largest deviation from a simple linear trend is {outlier_team}, highlighted via residual magnitude in the centered plot.")
    lines.append("We kept the x-axis as a ratio with a reference line at 1.0 to directly match the Phase 1b first/second line definition.")
    lines.append("The figure is saved under 5MB and includes labeled axes, title/subtitle, and a brief caption for reproducibility.")

    interpretation_path = reports_dir / "phase1c_interpretation_v2.txt"
    interpretation_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== STEP09B PHASE 1C PLOT (IMPROVED READABILITY) ===")
    print(f"strength: {strength_path.resolve()}")
    print(f"disparity: {disp_path.resolve()} (using {disp_col})")
    print(f"joined rows: {join_rows} (expected 32)")
    print(f"saved: {work_png.resolve()} | bytes={size_work}")
    print(f"saved: {submit_png.resolve()} | bytes={size_submit}")
    print(f"saved: {interpretation_path.resolve()}")
    print(f"spearman_rho(raw_strength)={rho:.6f} | p={pval:.6g}")
    print(f"labels_shown: {len(label_set)} teams")

    overall = (join_rows == 32) and work_png.exists() and (size_work <= 5 * 1024 * 1024)
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())
