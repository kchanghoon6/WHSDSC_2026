"""
Visualize bootstrap confidence intervals.

Reads teh CI tables produced in previous step and generates publication-ready uncertainty plots.
WRites figure files under out/.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def _get_paths(project_root: Path) -> dict:
    rc = _read_run_config(project_root)
    paths = rc.get("paths", {})

    out_dir = Path(paths.get("out_dir", project_root / "out")).resolve()
    reports_dir = Path(paths.get("reports_dir", project_root / "reports")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    ci_all = out_dir / "line_disparity_regression_ci_all.csv"
    if not ci_all.exists():
        raise FileNotFoundError(f"Missing CI file: {ci_all} (run STEP17 first)")
    
    diag = reports_dir / "line_disparity_regression_bootstrap_ci_diagnostics.json"

    forest_png = out_dir / "line_disparity_regression_ci_forest.png"
    prob_png = out_dir / "line_disparity_regression_prob_gt1.png"
    submit_copy = out_dir / "WHSDSC_2026.png"

    summary_txt = reports_dir / "line_disparity_regression_ci_summary.txt"

    return {
        "ci_all": ci_all,
        "diag": diag,
        "forest_png": forest_png,
        "prob_png": prob_png,
        "submit_copy": submit_copy,
        "summary_txt": summary_txt,
        "out_dir": out_dir,
        "reports_dir": reports_dir
    }


def _pick_columns(df: pd.DataFrame) -> dict:
    cols = set(df.columns)

    if "team" not in cols:
        raise KeyError(f"Missing 'team' column. Available={list(df.columns)}")
    
    if "disparity_ratio_point" in cols:
        point = "disparity_ratio_point"
    elif "disparity_ratio_reg" in cols:
        point = "disparity_ratio_reg"
    else:
        raise KeyError(f"Missing point estimate column. Available={list(df.columns)}")
    
    ci_pairs = [
        ("ci95_lo", "ci95_hi"),
        ("ci_low", "ci_high"),
        ("ci_lo", "ci_hi"),
        ("ci_lower", "ci_upper"),
        ("ci_l", "ci_u"),
    ]
    lo = hi = None
    for a, b in ci_pairs:
        if a in cols and b in cols:
            lo, hi = a, b
            break

    if lo is None or hi is None:
        if "ci" in cols:
            s = df["ci"].astype(str)
            lo_v = []
            hi_v = []
            for v in s:
                v2 = v.replace("–", "-").replace("—", '-')
                parts = v2.split("-")
                if len(parts) != 2:
                    lo_v.append(np.nan)
                    hi_v.append(np.nan)
                    continue
                lo_v.append(float(parts[0].strip()))
                hi_v.append(float(parts[1].strip()))
                lo, hi = "_ci_low_parsed", "_ci_high_parsed"
            else:
                raise KeyError(
                    f"Missing CI columns (low/high) or 'ci' string column. Available={list(df.columns)}"
                )
            
    boot_mean = "disparity_ratio_boot_mean" if "disparity_ratio_boot_mean" in cols else None
    boot_sd = "disparity_ratio_boot_sd" if "disparity_ratio_boot_sd" in cols else None

    return {"point": point, "lo": lo, "hi": hi, "boot_mean": boot_mean, "boot_sd": boot_sd}
    

def _safe_read_json(path: Path) -> dict | None:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return None
    return None


def _norm_prob_gt1(mu: float, sd: float) -> float:
    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
        return float("nan")
    z = (1.0 - mu) / (sd * math.sqrt(2.0))
    p = 0.5 * math.erfc(z)
    return float(max(0.0, min(1.0, p)))


def _save_forest_plot(df: pd.DataFrame, point: str, lo: str, hi: str, title: str, subtitle: str, out_path: Path) -> None:
    d = df.copy()
    d = d.sort_values(point, ascending=False).reset_index(drop=True)

    y = np.arange(len(d))
    x = d[point].to_numpy(dtype=float)
    lo_v = d[lo].to_numpy(dtype=float)
    hi_v = d[hi].to_numpy(dtype=float)
    xerr = np.vstack([x - lo_v, hi_v - x])

    fig_h = max(7.5, 0.33 * len(d) + 1.8)
    fig = plt.figure(figsize=(11.5, fig_h))
    ax = fig.add_subplot(111)

    ax.errorbar(x, y, xerr=xerr, fmt='o', capsize=3, linewidth=1)
    ax.axvline(1.0, linestyle='--', linewidth=1)

    ax.set_yticks(y)
    ax.set_yticklabels(d["team"].astype(str).tolist())
    ax.invert_yaxis()

    ax.set_xlabel("Disparity ratio (first-line xG/60 ÷ second-line xG/60)")
    ax.set_ylabel("Team")
    ax.set_title(title + "\n" + subtitle)

    cap = (
        "Each point is the regression-based disparity ratio with 95% CI (bootstrap summary). "
        "Values > 1.0 indicate higher first-line offensive production after opponent-defense controls."
    )
    fig.text(0.01, 0.01, cap, ha="left", va="bottom", fontsize=10)

    fig.tight_layout(rect=[0, 0.03, 1, 0.98])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _save_prob_plot(df: pd.DataFrame, out_path: Path) -> None:
    d = df.copy().sort_values("p_gt1_norm_approx", ascending=False).reset_index(drop=True)

    fig = plt.figure(figsize=(12, 6.5))
    ax = fig.add_subplot(111)
    ax.bar(d["team"].astype(str).tolist(), d["p_gt1_norm_approx"].to_numpy(dtype=float))
    ax.axhline(0.5, linestyle="--", linewidth=1)
    ax.set_title("Approx P(disparity ratio > 1.0) from bootstrap mean/sd (Normal approximation)")
    ax.set_xlabel("Team")
    ax.set_ylabel("Probability")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    project_root = _resolve_project_root()
    paths = _get_paths(project_root)

    ci_all = pd.read_csv(paths["ci_all"])
    col = _pick_columns(ci_all)

    for k in ["point", "lo", "hi", "boot_mean", "boot_sd"]:
        v = col.get(k)
        if v is not None and v in ci_all.columns:
            ci_all[v] = pd.to_numeric(ci_all[v], errors="coerce")

    ci_all["team"] = ci_all["team"].astype(str)

    n = len(ci_all)
    row_ok = (n == 32)
    no_nan_ci = bool(ci_all[[col["point"], col["lo"], col["hi"]]].notna().all().all())

    sig_gt1 = int((ci_all[col["lo"]] > 1.0).sum())
    sig_lt1 = int((ci_all[col["hi"]] < 1.0).sum())
    cross_1 = int(n - sig_gt1 - sig_lt1)

    diag = _safe_read_json(paths["diag"])
    diag_line = ""
    if diag and isinstance(diag, dict):
        meta = diag.get("meta", {})
        nb = meta.get("n_boot")
        ns = meta.get("n_success")
        seed = meta.get("seed")
        if nb is not None and ns is not None:
            diag_line = f" | bootstrap={ns}/{nb}"
        if seed is not None:
            diag_line += f" | seed={seed}"
        
    title = "STEP18 - Regression line disparity with 95% CI (forest plot)"
    subtitle = f"Source: {paths['ci_all'].name}{diag_line} | N={n}"
    _save_forest_plot(ci_all, col["point"], col['lo'], col["hi"], title, subtitle, paths["forest_png"])

    paths["submit_copy"].write_bytes(paths["forest_png"].read_bytes())

    size_forest = paths["forest_png"].stat().st_size
    size_copy = paths["submit_copy"].stat().st_size

    prob_ok = False
    size_prob = 0
    if col["boot_mean"] and col["boot_sd"]:
        mu = ci_all[col["boot_mean"]].to_numpy(dtype=float)
        sd = ci_all[col["boot_sd"]].to_numpy(dtype=float)
        p = [_norm_prob_gt1(float(m), float(s)) for m, s in zip(mu, sd)]
        ci_all["p_gt1_norm_approx"] = p
        if np.isfinite(ci_all["p_gt1_norm_approx"]).all():
            _save_prob_plot(ci_all, paths["prob_png"])
            size_prob = paths["prob_png"].stat().st_size
            prob_ok = True

    top10 = ci_all.sort_values(col["point"], ascending=False).head(10).copy()
    lines = []
    lines.append("Regression-based line disparity CI summary (STEP18 visuals)")
    lines.append(f"ci_file: {paths['ci_all'].resolve()}")
    lines.append(f"rows: {n} (expected 32) | ci_complete: {no_nan_ci}")
    lines.append(f"CI vs 1.0: lo>1={sig_gt1} | hi<1={sig_lt1} | crosses_1={cross_1}")
    if prob_ok:
        lines.append("Stability proxy: P(disparity>1.0) via Normal approximation from bootstrap mean/sd.")
    lines.append("")
    lines.append("Top10 by point estimate (with 95% CI):")
    for _, r in top10.iterrows():
        lines.append(
            f"- {r['team']}: {float(r[col['point']]):.6f} (95% CI {float(r[col['lo']]):.3f}-{float(r[col['hi']]):.3f})"
        )
    paths["summary_txt"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    under_5mb = (size_forest <= 5_000_000) and (size_copy <= 5_000_000) and (not prob_ok or size_prob <= 5_000_000)

    print("=== STEP18 LINE DISPARITY CI VISUALS ===")
    print(f"ci_all:    {paths['ci_all'].resolve()}")
    print(f"rows: {n} (expected 32) | ci_complete: {no_nan_ci}")
    print(f"CI vs 1.0: lo>1={sig_gt1} | hi<1={sig_lt1} | crosses_1={cross_1}")
    print(f"saved: {paths['forest_png'].resolve()} | bytes={size_forest}")
    print(f"saved: {paths['submit_copy'].resolve()} | bytes={size_copy}")
    if prob_ok:
        print(f"saved: {paths['prob_png'].resolve()} | bytes={size_prob}")
    print(f"saved: {paths['summary_txt'].resolve()}")

    overall = row_ok and no_nan_ci and under_5mb
    print(f"\nOVERALL: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 2


if __name__ == "__main__":
    raise SystemExit(main())