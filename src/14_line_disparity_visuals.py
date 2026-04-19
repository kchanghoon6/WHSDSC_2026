"""
Line disparity visuals.

Generates plots/tables for communicating line-disparity results (baseline/adjusted/regression).
Writes figure files under out/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Tuple

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


def _pick_first_existing(*paths: Path) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _get_paths(project_root: Path) -> Dict[str, Path]:
    rc = _read_run_config(project_root)
    paths = rc.get("paths", {})

    out_dir = Path(paths.get("out_dir", project_root / "out")).resolve()
    reports_dir = Path(paths.get("reports_dir", project_root / "reports")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    adjusted = out_dir / "line_disparity_adjusted_all.csv"
    regression = out_dir / "line_disparity_regression_all.csv"

    if not adjusted.exists():
        raise FileNotFoundError(f"missing: {adjusted} (run STEP08 first)")
    if not regression.exists():
        raise FileNotFoundError(f"missing: {regression} (run STEP15 first)")

    strength = _pick_first_existing(
        out_dir / "power_rankings_v3_offset.csv",
        out_dir / "power_rankings_v2_model.csv",
        out_dir / "power_rankings_v1.csv",
        out_dir / "power_rankings.csv",
    )
    if strength is None:
        raise FileNotFoundError("missing power rankings csv (expected v3_offset or v2_model or v1)")

    return {
        "out_dir": out_dir,
        "reports_dir": reports_dir,
        "adjusted": adjusted,
        "regression": regression,
        "strength": strength,
    }


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _compute_ranks(df: pd.DataFrame, col: str, rank_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp = tmp.sort_values(col, ascending=False).reset_index(drop=True)
    tmp[rank_col] = np.arange(1, len(tmp) + 1)
    return tmp


def _save_png(fig: plt.Figure, path: Path) -> int:
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path.stat().st_size


def main() -> int:
    project_root = _resolve_project_root()
    p = _get_paths(project_root)

    out_dir = p["out_dir"]
    reports_dir = p["reports_dir"]

    adj = pd.read_csv(p["adjusted"])
    reg = pd.read_csv(p["regression"])
    strength = pd.read_csv(p["strength"])

    if "team" not in adj.columns or "disparity_ratio_adj" not in adj.columns:
        raise ValueError("adjusted file must contain columns: team, disparity_ratio_adj")
    if "team" not in reg.columns or "disparity_ratio_reg" not in reg.columns:
        raise ValueError("regression file must contain columns: team, disparity_ratio_reg")

    if "team" not in strength.columns:
        raise ValueError("strength file must contain column: team")
    if "strength_score" not in strength.columns:
        raise ValueError("strength file must contain column: strength_score")

    adj = adj[["team", "disparity_ratio_adj"]].copy()
    reg = reg[["team", "disparity_ratio_reg"]].copy()
    strength = strength[["team", "strength_score"]].copy()

    adj["disparity_ratio_adj"] = _safe_numeric(adj["disparity_ratio_adj"])
    reg["disparity_ratio_reg"] = _safe_numeric(reg["disparity_ratio_reg"])
    strength["strength_score"] = _safe_numeric(strength["strength_score"])

    m = adj.merge(reg, on="team", how="inner")
    m = m.merge(strength, on="team", how="inner")
    if len(m) != 32:
        raise ValueError(f"joined rows must be 32, got {len(m)}")

    m = _compute_ranks(m, "disparity_ratio_adj", "adjusted_rank")
    m = _compute_ranks(m, "disparity_ratio_reg", "reg_rank")

    merged = m.copy()
    merged["ratio_diff"] = merged["disparity_ratio_reg"] - merged["disparity_ratio_adj"]
    merged["rank_delta"] = merged["reg_rank"] - merged["adjusted_rank"]
    merged["abs_rank_delta"] = merged["rank_delta"].abs()

    rho_ratio, p_ratio = spearmanr(merged["disparity_ratio_adj"], merged["disparity_ratio_reg"])
    rho_phase, p_phase = spearmanr(merged["disparity_ratio_reg"], merged["strength_score"])

    top10_adj = set(merged.sort_values("disparity_ratio_adj", ascending=False).head(10)["team"].tolist())
    top10_reg = set(merged.sort_values("disparity_ratio_reg", ascending=False).head(10)["team"].tolist())
    overlap = len(top10_adj.intersection(top10_reg))

    # Plot 1: adjusted vs regression agreement
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.scatter(merged["disparity_ratio_adj"], merged["disparity_ratio_reg"])
    lo = float(min(merged["disparity_ratio_adj"].min(), merged["disparity_ratio_reg"].min()))
    hi = float(max(merged["disparity_ratio_adj"].max(), merged["disparity_ratio_reg"].max()))
    pad = 0.02 * (hi - lo) if hi > lo else 0.02
    ax1.plot([lo - pad, hi + pad], [lo - pad, hi + pad], linestyle="--")
    ax1.set_xlabel("Adjusted disparity ratio (STEP08)")
    ax1.set_ylabel("Regression disparity ratio (STEP15)")
    ax1.set_title(f"Adjusted vs Regression disparity (Spearman rho={rho_ratio:.3f}, p={p_ratio:.3g} | top10 overlap={overlap}/10)")

    # label: largest abs diff + any abs_rank_delta>=3
    lab = merged.sort_values("ratio_diff", key=lambda s: s.abs(), ascending=False).head(1)
    movers = merged[merged["abs_rank_delta"] >= 3].copy()
    label_df = pd.concat([lab, movers], ignore_index=True).drop_duplicates(subset=["team"]).head(12)

    for _, r in label_df.iterrows():
        ax1.annotate(str(r["team"]), (float(r["disparity_ratio_adj"]), float(r["disparity_ratio_reg"])), xytext=(4, 4), textcoords="offset points")

    p1 = out_dir / "step16_disparity_adjusted_vs_regression.png"
    b1 = _save_png(fig1, p1)

    # Plot 2: rank delta bar
    fig2 = plt.figure(figsize=(10, 5.5))
    ax2 = fig2.add_subplot(111)
    rd = merged.sort_values("abs_rank_delta", ascending=False).head(20).copy()
    ax2.bar(rd["team"].astype(str).tolist(), rd["rank_delta"].astype(int).tolist())
    ax2.axhline(0, linestyle="--")
    ax2.set_ylabel("rank_delta = reg_rank - adjusted_rank")
    ax2.set_title("Largest rank movers (Regression vs Adjusted)")
    ax2.tick_params(axis="x", rotation=45, labelsize=9)

    p2 = out_dir / "step16_rank_delta_top20.png"
    b2 = _save_png(fig2, p2)

    # Plot 3: Phase1c using regression disparity (no overwrite of submission png)
    y = merged["strength_score"].astype(float)
    y_center = y - float(y.mean())

    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    ax3.scatter(merged["disparity_ratio_reg"], y_center)
    ax3.axvline(1.0, linestyle="--")
    ax3.axhline(0.0, linestyle="--")
    ax3.set_xlabel("Line disparity (regression): disparity_ratio_reg (1.0 = equal)")
    ax3.set_ylabel("Team strength (centered): strength_score - mean(strength_score)")
    ax3.set_title(f"Phase1c alternative (regression disparity) | Spearman rho={rho_phase:.3f}, p={p_phase:.3g} | N=32")

    # label extremes by x and y + one largest residual to a simple linear fit
    x = merged["disparity_ratio_reg"].astype(float).to_numpy()
    yy = y_center.astype(float).to_numpy()
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
    resid = yy - (A @ coef)

    idx = set()
    idx.update(merged.sort_values("disparity_ratio_reg", ascending=False).head(3).index.tolist())
    idx.update(merged.sort_values("disparity_ratio_reg", ascending=True).head(3).index.tolist())
    idx.update(merged.sort_values("strength_score", ascending=False).head(3).index.tolist())
    idx.update(merged.sort_values("strength_score", ascending=True).head(3).index.tolist())
    idx.add(int(np.argmax(np.abs(resid))))

    for i in sorted(idx):
        r = merged.iloc[i]
        ax3.annotate(str(r["team"]), (float(r["disparity_ratio_reg"]), float((r["strength_score"] - y.mean()))), xytext=(4, 4), textcoords="offset points")

    p3 = out_dir / "step16_phase1c_regression_disparity.png"
    b3 = _save_png(fig3, p3)

    summary = {
        "files_used": {
            "adjusted": str(p["adjusted"]),
            "regression": str(p["regression"]),
            "strength": str(p["strength"]),
        },
        "metrics": {
            "spearman_ratio_adj_vs_reg": float(rho_ratio),
            "spearman_ratio_adj_vs_reg_p": float(p_ratio),
            "top10_overlap_adj_vs_reg": int(overlap),
            "spearman_strength_vs_reg_disparity": float(rho_phase),
            "spearman_strength_vs_reg_disparity_p": float(p_phase),
        },
        "outputs": {
            "adjusted_vs_regression_png": {"path": str(p1), "bytes": int(b1)},
            "rank_delta_png": {"path": str(p2), "bytes": int(b2)},
            "phase1c_regression_png": {"path": str(p3), "bytes": int(b3)},
        },
        "largest_ratio_diff": merged.sort_values("ratio_diff", key=lambda s: s.abs(), ascending=False).head(5)[
            ["team", "disparity_ratio_adj", "disparity_ratio_reg", "ratio_diff", "adjusted_rank", "reg_rank", "rank_delta"]
        ].to_dict(orient="records"),
    }

    out_json = reports_dir / "step16_line_disparity_visuals_summary.json"
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    notes = (
        "STEP16 created three plots: (1) adjusted vs regression agreement, (2) largest rank movers, and "
        "(3) a Phase1c alternative using regression-based disparity. If (1) points lie close to y=x and "
        "top10 overlap is high, the line-disparity conclusion is robust to the choice of adjustment method. "
        "If (3) Spearman remains near zero with a large p-value, it supports the conclusion that line-disparity "
        "is not strongly associated with team strength in this dataset under the chosen definitions."
    )
    (reports_dir / "step16_notes.txt").write_text(notes.strip() + "\n", encoding="utf-8")

    # DoD: png < 5MB
    ok = all(sz < 5_000_000 for sz in [b1, b2, b3])

    print("=== STEP16 LINE DISPARITY VISUALS ===")
    print(f"adjusted:   {Path(p['adjusted']).resolve()}")
    print(f"regression: {Path(p['regression']).resolve()}")
    print(f"strength:   {Path(p['strength']).resolve()}")
    print(f"saved: {p1.resolve()} | bytes={b1}")
    print(f"saved: {p2.resolve()} | bytes={b2}")
    print(f"saved: {p3.resolve()} | bytes={b3}")
    print(f"summary: {out_json.resolve()}")
    print(f"notes:   {(reports_dir / 'step16_notes.txt').resolve()}")
    print(f"\nDoD: all_png_under_5MB={ok}")
    print(f"OVERALL: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
