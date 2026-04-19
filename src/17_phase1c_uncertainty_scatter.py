"""
Uncertainty scatter plot for Phase 1C.

Creates a scatter plot that highlights both point estimates and uncertainty for key metrics.
WRites figure files under out/phase1c_*.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr


REQ_STRENGTH_COLS = {"team", "strength_score"}
REQ_CI_COLS = {"team", "disparity_ratio_point", "ci95_lo", "ci95_hi"}


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


def _get_paths(project_root: Path) -> dict:
    rc = _read_run_config(project_root)
    paths = rc.get("paths", {})

    out_dir = Path(paths.get("out_dir", project_root / "out")).resolve()
    reports_dir = Path(paths.get("reports_dir", project_root / "reports")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    strength_path = out_dir / "power_rankings_v3_offset.csv"
    if not strength_path.exists():
        alt = out_dir / "power_rankings_v3_offset.csv"
        if alt.exists():
            strength_path = alt
        else:
            raise FileNotFoundError(f"strength file not found: {strength_path}")
        
    ci_path = out_dir / "line_disparity_regression_ci_all.csv"
    if not ci_path.exists():
        raise FileNotFoundError(f"CI file not found: {ci_path}")
    
    return {
        "out_dir": out_dir,
        "reports_dir": reports_dir,
        "strength_path": strength_path,
        "ci_path": ci_path,
    }


def _require_cols(df: pd.DataFrame, required: set[str], name: str) -> None:
    cols = {c.strip() for c in df.columns}
    missing = sorted(list(required - cols))
    if missing:
        raise KeyError(f"{name}: missing required columns={missing} | available={sorted(list(cols))}")
    

def _select_labels(df: pd.DataFrame, target_n: int = 18) -> set[str]:
    df2 = df.copy()

    def top_teams(col: str, n: int, ascending: bool) -> list[str]:
        d = df2[["team_canon", col]].dropna().copy()
        d = d.sort_values([col, "team_canon"], ascending=[ascending, True])
        return d["team_canon"].head(n).tolist()
    
    picks: list[str] = []
    picks += top_teams("disparity_ratio_point", 6, ascending=False)
    picks += top_teams("disparity_ratio_point", 4, ascending=True)
    picks += top_teams("strength_index", 6, ascending=False)
    picks += top_teams("strength_index", 4, ascending=True)
    picks += top_teams("ci_width", 6, ascending=False)


    seen = []
    seen_set = set()
    for t in picks:
        if t not in seen_set:
            seen.append(t)
            seen_set.add(t)
        
    if len(seen) <= target_n:
        return set(seen)
    
    keep = seen[:target_n]
    return set(keep)


def _write_notes(
    out_path: Path,
    n_rows: int,
    rho: float,
    p: float,
    n_labels: int,
    ci_sig_gt1: int,
    ci_cross_1: int,
) -> None:
    lines = []
    lines.append(
        f"We joined team strength (offset Poisson model) with regression-based line disparity CI for {n_rows} teams."
    )
    lines.append(
        "The scatter plot encodes uncertainty by marker size (wider 95% CI -> larger marker) and selectively labels a subset of teams."
    )
    lines.append(
        f"Spearman correlation between disparity_ratio_point and strength_index (higher = stronger) is rho={rho:.3f} (p={p:.3f}), suggesting little monotonic association in this sample."
    )
    lines.append(
        f"Confidence-interval screening shows {ci_sig_gt1} teams with ci95_lo > 1.0 (first line reliably above second line) and {ci_cross_1} teams whose CI crosses 1.0."
    )
    lines.append(
        f"Labels shown: {n_labels} teams selected from extremes of disparity, strength, and CI width to aid readability without clutter."
    )
    lines.append(
        "Interpretation should emphasize (1) which teams have reliably high disparity (ci95_lo>1), and (2) that the strength-disparity relationshipo appears weak under this definition."
    )
    lines.append(
        "All steps are reproducible from the stored CSV outputs; the plot is a visualization of already-computed metrics rather than an additional fitted model."
    )
    out_path.write_text("\n".join(lines).strip() + "\n", encoding='utf-8')


def main() -> int:
    project_root = _resolve_project_root()
    paths = _get_paths(project_root)

    strength = pd.read_csv(paths["strength_path"])
    ci = pd.read_csv(paths["ci_path"])

    _require_cols(strength, REQ_STRENGTH_COLS, "strength")
    _require_cols(ci, REQ_CI_COLS, "ci_all")

    strength = strength.copy()
    strength["team"] = strength["team"].astype(str)
    strength["team_canon"] = strength["team"].map(_canon)
    strength["strength_score"] = pd.to_numeric(strength["strength_score"], errors="coerce")

    ci = ci.copy()
    ci["team"] = ci["team"].astype(str)
    ci["team_canon"] = ci["team"].map(_canon)
    ci["disparity_ratio_point"] = pd.to_numeric(ci["disparity_ratio_point"], errors="coerce")
    ci["ci95_lo"] = pd.to_numeric(ci["ci95_lo"], errors="coerce")
    ci["ci95_hi"] = pd.to_numeric(ci["ci95_hi"], errors="coerce")
    ci["ci_width"] = (ci["ci95_hi"] - ci["ci95_lo"]).clip(lower=0.0)

    merged = pd.merge(
        strength[["team", "team_canon", "strength_score"]],
        ci[["team_canon", "disparity_ratio_point", "ci95_lo", "ci95_hi", "ci_width"]],
        on="team_canon",
        how="inner",
    )

    merged = merged.copy()
    merged["strength_index"] = -pd.to_numeric(merged["strength_score"], errors="coerce")

    n_rows = int(len(merged))
    joined_ok = (n_rows == 32)

    rho, pval = spearmanr(
        merged["disparity_ratio_point"].to_numpy(dtype=float),
        merged["strength_index"].to_numpy(dtype=float),
        nan_policy="omit",
    )
    rho = float(rho) if np.isfinite(rho) else float("nan")
    pval = float(pval) if np.isfinite(pval) else float("nan")

    labels = _select_labels(merged, target_n=18)
    merged["label"] = merged["team_canon"].isin(labels)
    
    lo_gt1 = int((merged["ci95_lo"] > 1.0).sum())
    crosses_1 = int(((merged["ci95_lo"] <= 1.0) & (merged["ci95_hi"] >= 1.0)).sum())

    out_dir = paths["out_dir"]
    reports_dir = paths["reports_dir"]

    join_csv = reports_dir / "phase1c_uncertainty_join.csv"
    merged.sort_values(["strength_index", "team_canon"], ascending=[False, True]).to_csv(join_csv, index=False)

    x = merged["disparity_ratio_point"].to_numpy(dtype=float)
    y = merged["strength_index"].to_numpy(dtype=float)

    w = merged["ci_width"].to_numpy(dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    w_norm = (w - w.min()) / (w.max() - w.min() + 1e-12)
    sizes = 30.0 + 220.0 * w_norm

    fig = plt.figure(figsize=(10.5, 7.0), dpi=160)
    ax = plt.gca()

    ax.scatter(x, y, s=sizes, alpha=0.75, edgecolors="none")

    ax.axvline(1.0, linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Line disparity (first/second), point estimate")
    ax.set_ylabel("Team strength index (higher = stronger)")
    ax.set_title("Phase 1c: Line Disparity vs Team Strength (marker size = CI width)")

    caption = (
        f"Spearman rho={rho:.3f} (p={pval:.3f}) | "
        f"ci95_lo>1.0 teams={lo_gt1} | CI crosses 1.0 teams={crosses_1}"
    )
    ax.text(0.01, -0.12, caption, transform=ax.transAxes, fontsize=9, alpha=0.9)

    for _, r in merged.loc[merged["label"]].iterrows():
        ax.annotate(
            r["team"],
            (float(r["disparity_ratio_point"]), float(r["strength_index"])),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=8,
            alpha=0.9,
        )

    ax.grid(True, alpha=0.25)
    plt.tight_layout()

    out_plot = out_dir / "phase1c_strength_vs_disparity_uncertainty.png"
    fig.savefig(out_plot, bbox_inches="tight")
    plt.close(fig)

    submit_plot = out_dir / "WHSDSC_2026.png"
    submit_plot.write_bytes(out_plot.read_bytes())

    notes_path = reports_dir / "phase1c_uncertainty_notes.txt"
    _write_notes(
        notes_path,
        n_rows=n_rows,
        rho=rho,
        p=pval,
        n_labels=int(len(labels)),
        ci_sig_gt1=lo_gt1,
        ci_cross_1=crosses_1,
    )

    size_bytes = int(out_plot.stat().st_size)
    within_5mb = (size_bytes <= 5 * 1024 * 1024)

    print("=== STEP19 PHASE 1C UNCERTAINTY SCATTER (CI-aware) ===")
    print(f"strength: {paths['strength_path'].resolve()}")
    print(f"ci_all:   {paths['ci_path'].resolve()}")
    print(f"joined rows: {n_rows} (expected 32) | joined_ok: {joined_ok}")
    print(f"saved: {out_plot.resolve()} | bytes={size_bytes} | <=5MB: {within_5mb}")
    print(f"saved: {submit_plot.resolve()} | bytes={int(submit_plot.stat().st_size)}")
    print(f"saved: {notes_path.resolve()}")
    print(f"saved: {join_csv.resolve()}")
    print(f"spearman_rho={rho:.6f} | p={pval:.6f}")
    print(f"labels_shown: {int(len(labels))} teams")
    print(f"CI vs 1.0: lo>1={lo_gt1} | crosses_1={crosses_1}")

    overall = joined_ok and within_5mb and np.isfinite(rho) and np.isfinite(pval)
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())