#!/usr/bin/env python3
"""Create static scientific figures from actual SIMPLE validation results.

Requires matplotlib and numpy. No solver is run, no missing data is synthesized,
and all six plotted cases must have complete, converged output before any figure
is written. A completed plot is not an assertion that the full suite passed.

  python scripts/plot_simple_validation.py
  python scripts/plot_simple_validation.py --pdf --dpi 300
  python scripts/plot_simple_validation.py --suite-root output/simple_validation_full --output validation/figures

Outputs: accuracy_by_resolution, duct16_section, adaptive_axial_error,
adaptive_convergence (.png and optional .pdf), plus plot_manifest.json with
source hashes, actual values, definitions, and the recorded suite status.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


REPO = Path(__file__).resolve().parents[1]
CASES = ("uniform8", "uniform16", "adaptive8", "adaptive16", "duct8", "duct16")
np = None
plt = None


class PlotError(RuntimeError):
    pass


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path):
    def reject(token):
        raise ValueError(f"nonfinite JSON constant {token}")
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject)
    except (OSError, ValueError) as error:
        raise PlotError(f"Cannot read {path}: {error}") from error


def read_table(path, required):
    try:
        with path.open(newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = set(required) - set(reader.fieldnames or [])
            if missing:
                raise PlotError(f"{path}: missing columns {sorted(missing)}")
            rows = list(reader)
        if not rows:
            raise PlotError(f"{path}: empty data table")
        table = {column: np.asarray([float(row[column]) for row in rows], dtype=float) for column in required}
        if not all(np.isfinite(values).all() for values in table.values()):
            raise PlotError(f"{path}: nonfinite numeric data")
        return table
    except (OSError, ValueError, TypeError) as error:
        raise PlotError(f"Cannot read {path}: {error}") from error


def metric_value(metrics, key, source):
    value = metrics.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PlotError(f"{source}: missing or nonfinite metric {key}")
    return float(value)


def close(actual, expected, label, rtol=1e-8, atol=1e-14):
    if not math.isclose(float(actual), float(expected), rel_tol=rtol, abs_tol=atol):
        raise PlotError(f"Inconsistent results for {label}: CSV={actual}, metrics={expected}")


def load_cases(root):
    leaves = ("metrics.json", "case.json", "solution.csv", "sections.csv", "history.csv")
    required = [root / name / leaf for name in CASES for leaf in leaves]
    required.append(root / "suite_results.json")
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise PlotError("Required validation outputs are missing; finish the full suite first:\n  " + "\n  ".join(missing))
    cases = {}
    for name in CASES:
        directory = root / name
        metrics = read_json(directory / "metrics.json")
        config = read_json(directory / "case.json")
        if metrics.get("converged") is not True:
            raise PlotError(f"{name}: solver is not marked converged; refusing to present this as a final accuracy figure")
        ny = 16 if name.endswith("16") else 8
        for key, expected in (("ny", ny), ("adaptive", name.startswith("adaptive")),
                              ("periodic_z", not name.startswith("duct")), ("periodic_x", True)):
            if metrics.get(key) != expected or config.get(key) != expected:
                raise PlotError(f"{name}: {key} must equal {expected} in both case.json and metrics.json")
        for key in ("velocity_relative_l2", "flow_rate_relative_error", "volume_flux", "volume_flux_exact",
                    "momentum_relative_l2", "continuity_relative_linf", "anderson_depth"):
            metric_value(metrics, key, name)
        if metrics["velocity_relative_l2"] < 0 or metrics["flow_rate_relative_error"] < 0:
            raise PlotError(f"{name}: error metrics cannot be negative")
        if metrics["anderson_depth"] != config.get("anderson_depth"):
            raise PlotError(f"{name}: inconsistent recorded Anderson configuration")
        solution = read_table(directory / "solution.csv",
                              ("id", "level", "x", "y", "z", "h", "volume", "u", "v", "w", "u_exact"))
        if len(solution["id"]) != metrics.get("cells") or len(set(solution["id"])) != len(solution["id"]):
            raise PlotError(f"{name}: cell count/IDs do not agree with metrics")
        if np.any(solution["volume"] <= 0) or np.any(solution["h"] <= 0):
            raise PlotError(f"{name}: nonpositive cell volume or size")
        squared_error = (solution["u"] - solution["u_exact"])**2 + solution["v"]**2 + solution["w"]**2
        exact_norm = np.sum(solution["u_exact"]**2 * solution["volume"])
        if exact_norm <= 0:
            raise PlotError(f"{name}: zero reference-velocity norm")
        recomputed_l2 = np.sqrt(np.sum(squared_error * solution["volume"]) / exact_norm)
        close(recomputed_l2, metrics["velocity_relative_l2"], f"{name} velocity L2")
        sections = read_table(directory / "sections.csv", ("x", "volume_flux", "area", "exact_volume_flux"))
        if np.any(sections["area"] <= 0) or np.any(sections["exact_volume_flux"] == 0):
            raise PlotError(f"{name}: invalid cross-section area or reference flux")
        if len(set(sections["x"])) != len(sections["x"]):
            raise PlotError(f"{name}: duplicate axial section coordinates")
        close(np.mean(sections["volume_flux"]), metrics["volume_flux"], f"{name} true face-flux Q")
        if not np.allclose(sections["exact_volume_flux"], metrics["volume_flux_exact"], rtol=1e-12, atol=1e-15):
            raise PlotError(f"{name}: sections do not share the reported exact Q")
        q_error = abs(np.mean(sections["volume_flux"]) - metrics["volume_flux_exact"]) / abs(metrics["volume_flux_exact"])
        close(q_error, metrics["flow_rate_relative_error"], f"{name} true face-flux error")
        history = read_table(directory / "history.csv",
                             ("iteration", "momentum_relative_l2", "continuity_relative_linf",
                              "nonorth_face_defect_relative_linf"))
        if np.any(np.diff(history["iteration"]) != 1) or history["iteration"][0] != 1:
            raise PlotError(f"{name}: history iterations must be contiguous starting at 1")
        if history["iteration"][-1] != metrics.get("iterations"):
            raise PlotError(f"{name}: history is not complete through the reported final iteration")
        for key in ("momentum_relative_l2", "continuity_relative_linf", "nonorth_face_defect_relative_linf"):
            if np.any(history[key] < 0):
                raise PlotError(f"{name}: negative residual in {key}")
            close(history[key][-1], metric_value(metrics, key, name), f"{name} final {key}")
        bounds = {a: (float(np.min(solution[a] - solution["h"] / 2)),
                      float(np.max(solution[a] + solution["h"] / 2))) for a in "xyz"}
        cases[name] = {"metrics": metrics, "config": config, "solution": solution,
                       "sections": sections, "history": history, "bounds": bounds,
                       "squared_velocity_error": squared_error}
    return cases, required


def choose_scale(axis, values):
    """Do not replace actual zeros by invented positive residuals."""
    values = np.concatenate([np.ravel(v) for v in values])
    if np.any(values == 0):
        axis.set_yscale("symlog", linthresh=1e-14)
        return "symlog, linear region below 1e-14; exact zeros retained"
    axis.set_yscale("log")
    return "log"


def save_figure(fig, stem, args, output):
    paths = []
    for suffix in (["png", "pdf"] if args.pdf else ["png"]):
        path = args.output / f"{stem}.{suffix}"
        fig.savefig(path, dpi=args.dpi, facecolor="white", bbox_inches="tight")
        paths.append(str(path))
    plt.close(fig)
    output[stem] = {"files": paths}
    return output[stem]


def plot_accuracy(cases, args, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.3), constrained_layout=True)
    groups = (("uniform", "Parallel plates: uniform", "#0072B2", "o"),
              ("adaptive", "Parallel plates: octree", "#009E73", "s"),
              ("duct", "Four-wall square duct", "#D55E00", "^"))
    panels = (("velocity_relative_l2", "Cell velocity relative $L_2$ error (%)"),
              ("flow_rate_relative_error", "Conservative face-flux $Q$ relative error (%)"))
    values = []
    for ax, (metric, ylabel) in zip(axes, panels):
        curves = []
        for prefix, label, color, marker in groups:
            data = np.array([100 * cases[f"{prefix}{ny}"]["metrics"][metric] for ny in (8, 16)])
            curves.append(data)
            ax.plot((8, 16), data, marker=marker, color=color, label=label, linewidth=1.7, markersize=6)
        values.append(choose_scale(ax, curves))
        ax.set_xticks((8, 16))
        ax.set_xlabel("Base cells across channel height ($N_y$)")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="both", alpha=0.22)
    axes[0].legend(loc="best", fontsize=8.5)
    fig.suptitle("Accuracy of developed laminar flow", fontsize=13)
    fig.supxlabel("Octree cases refine part of x to 2× the base resolution. Lines connect measured points; no convergence order is fitted.", fontsize=8)
    record = save_figure(fig, "accuracy_by_resolution", args, output)
    record["y_scales"] = values
    record["definitions"] = {"velocity": "volume-weighted vector velocity L2 error at cell centers",
                              "flow": "absolute relative error of actual conservative axial face flux; not cell-velocity quadrature"}
    record["values"] = {name: {key: cases[name]["metrics"][key] for key, _ in panels} for name in CASES}


def duct_section(case):
    data = case["solution"]
    lo, hi = case["bounds"]["x"]
    x_values = np.unique(data["x"])
    selected = float(x_values[np.argmin(abs(x_values - (lo + hi) / 2))])
    mask = data["x"] == selected
    y, z = np.unique(data["y"][mask]), np.unique(data["z"][mask])
    if mask.sum() != len(y) * len(z):
        raise PlotError("duct16: selected transverse cell-center plane does not form a complete tensor grid")
    h = data["h"][mask]
    if not np.allclose(h, h[0], rtol=0, atol=1e-14):
        raise PlotError("duct16: unexpected nonuniform section spacing")
    u, exact = np.full((len(y), len(z)), np.nan), np.full((len(y), len(z)), np.nan)
    for index in np.flatnonzero(mask):
        j, k = np.searchsorted(y, data["y"][index]), np.searchsorted(z, data["z"][index])
        if np.isfinite(u[j, k]):
            raise PlotError("duct16: duplicate y/z cell in selected section")
        u[j, k], exact[j, k] = data["u"][index], data["u_exact"][index]
    if not np.isfinite(u).all() or not np.isfinite(exact).all():
        raise PlotError("duct16: incomplete section")
    y_edges = np.r_[y - h[0] / 2, y[-1] + h[0] / 2]
    z_edges = np.r_[z - h[0] / 2, z[-1] + h[0] / 2]
    u_reference = float(np.max(np.abs(exact)))
    if u_reference <= 0:
        raise PlotError("duct16: zero reference velocity")
    return {"x": selected, "y_edges": y_edges, "z_edges": z_edges, "u": u, "exact": exact,
            "normalized_error_percent": 100 * (u - exact) / u_reference, "u_reference": u_reference}


def plot_duct(case, section, args, output):
    fig, axes = plt.subplots(1, 3, figsize=(12.3, 3.9), constrained_layout=True)
    y0, y1 = case["bounds"]["y"]
    z0, z1 = case["bounds"]["z"]
    yy = (section["y_edges"] - y0) / (y1 - y0)
    zz = (section["z_edges"] - z0) / (z1 - z0)
    top = max(float(np.max(section["u"])), float(np.max(section["exact"])))
    for ax, field, title in zip(axes[:2], ("u", "exact"), ("Numerical axial velocity", "Analytic duct velocity")):
        velocity_image = ax.pcolormesh(zz, yy, section[field], cmap="viridis", vmin=0, vmax=top, shading="flat")
        ax.set_title(title, fontsize=10)
    magnitude = float(np.max(np.abs(section["normalized_error_percent"])))
    # This only sets a nondegenerate color scale when the actual array is zero.
    color_limit = magnitude if magnitude > 0 else np.finfo(float).eps
    error_image = axes[2].pcolormesh(zz, yy, section["normalized_error_percent"], cmap="RdBu_r",
                                    vmin=-color_limit, vmax=color_limit, shading="flat")
    axes[2].set_title("Signed error / sampled exact $U_{max}$", fontsize=10)
    for ax in axes:
        ax.set(xlabel="$z/W$", ylabel="$y/H$", aspect="equal")
    fig.colorbar(velocity_image, ax=list(axes[:2]), label="Axial velocity (solver units)", shrink=0.85)
    fig.colorbar(error_image, ax=axes[2], label="Normalized error (%)", shrink=0.85)
    fig.suptitle(f"Four-wall duct, $N_y=N_z=16$, cell-center slice x={section['x']:.6g}", fontsize=12)
    record = save_figure(fig, "duct16_section", args, output)
    record.update(selected_x=section["x"], reference_velocity=section["u_reference"],
                  maximum_normalized_error_percent=magnitude,
                  error_definition="100*(u-u_exact)/max(abs(u_exact)) on the same sampled cross-section")


def axial_profile(case, name):
    data = case["solution"]
    positions = np.unique(data["x"])
    errors = []
    area_expected = (case["bounds"]["y"][1] - case["bounds"]["y"][0]) * (case["bounds"]["z"][1] - case["bounds"]["z"][0])
    for x in positions:
        mask = data["x"] == x
        close(np.sum(data["h"][mask]**2), area_expected, f"{name} section coverage at x={x}")
        denominator = np.sum(data["u_exact"][mask]**2 * data["volume"][mask])
        if denominator <= 0:
            raise PlotError(f"{name}: zero exact norm at x={x}")
        errors.append(100 * np.sqrt(np.sum(case["squared_velocity_error"][mask] * data["volume"][mask]) / denominator))
    refined = data["level"] > np.min(data["level"])
    if not refined.any():
        raise PlotError(f"{name}: no refined cells in an adaptive case")
    region = (float(np.min(data["x"][refined] - data["h"][refined] / 2)),
              float(np.max(data["x"][refined] + data["h"][refined] / 2)))
    return positions, np.asarray(errors), region


def plot_axial(cases, profiles, args, output):
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.1), constrained_layout=True)
    error_curves = []
    for name, color in (("adaptive8", "#0072B2"), ("adaptive16", "#D55E00")):
        case, (x, errors, region) = cases[name], profiles[name]
        lo, hi = case["bounds"]["x"]
        axes[0].plot((x - lo) / (hi - lo), errors, label=name, color=color, linewidth=1.4)
        sections = case["sections"]
        order = np.argsort(sections["x"])
        relative_q = 100 * (sections["volume_flux"] - sections["exact_volume_flux"]) / abs(sections["exact_volume_flux"])
        axes[1].plot((sections["x"][order] - lo) / (hi - lo), relative_q[order], label=name, color=color, linewidth=1.4)
        error_curves.append(errors)
    lo, hi = cases["adaptive8"]["bounds"]["x"]
    region = profiles["adaptive8"][2]
    if not np.allclose(region, profiles["adaptive16"][2], rtol=0, atol=1e-12):
        raise PlotError("Adaptive cases have different refinement extents; cannot use a shared refinement band")
    for ax in axes:
        ax.axvspan((region[0] - lo) / (hi - lo), (region[1] - lo) / (hi - lo), color="0.8", alpha=0.25,
                   label="Locally refined region")
        ax.set_xlabel("Axial position $x/L_x$")
        ax.grid(True, which="both", alpha=0.22)
    y_scale = choose_scale(axes[0], error_curves)
    axes[0].set_ylabel("Cross-section velocity relative $L_2$ error (%)")
    axes[1].set_ylabel("Signed conservative $Q$ relative error (%)")
    axes[0].legend(fontsize=8.5)
    fig.suptitle("Octree interface accuracy along the channel", fontsize=12)
    record = save_figure(fig, "adaptive_axial_error", args, output)
    record.update(velocity_y_scale=y_scale, refined_x_interval=list(region),
                  velocity_error="Volume-weighted vector velocity L2 error within each actual cell-center x slice",
                  flux_error="Signed relative error from sections.csv authority flux, independent of cell velocity sampling")


def plot_convergence(cases, args, output):
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), constrained_layout=True)
    fields = (("momentum_relative_l2", "Momentum residual (relative $L_2$)"),
              ("continuity_relative_linf", r"Continuity residual (relative $L_\infty$)"),
              ("nonorth_face_defect_relative_linf", r"Deferred face-flux defect (relative $L_\infty$)"))
    scales = {}
    for ax, (field, label) in zip(axes, fields):
        curves = []
        for name, color in (("adaptive8", "#0072B2"), ("adaptive16", "#D55E00")):
            case = cases[name]
            history = case["history"]
            depth = int(case["metrics"]["anderson_depth"])
            ax.plot(history["iteration"], history[field], label=f"{name}, Anderson depth {depth}", color=color, linewidth=1.1)
            curves.append(history[field])
        tolerance = cases["adaptive8"]["config"]["tolerance"]
        if tolerance != cases["adaptive16"]["config"]["tolerance"]:
            raise PlotError("Adaptive cases have different stopping tolerances")
        ax.axhline(tolerance, color="0.35", linestyle=":", label=f"Stopping tolerance {tolerance:g}")
        scales[field] = choose_scale(ax, curves)
        ax.set(xlabel="SIMPLE outer iteration", ylabel=label)
        ax.grid(True, which="both", alpha=0.22)
    axes[0].legend(fontsize=7.5)
    fig.suptitle("Adaptive convergence; acceleration settings are stated separately from grid resolution", fontsize=11)
    record = save_figure(fig, "adaptive_convergence", args, output)
    record.update(y_scales=scales, acceleration={name: cases[name]["metrics"]["anderson_depth"] for name in ("adaptive8", "adaptive16")},
                  interpretation="Iteration counts reflect both mesh resolution and acceleration configuration; they do not isolate a mesh effect")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite-root", type=Path, default=REPO / "output/simple_validation_full")
    parser.add_argument("--output", type=Path, default=REPO / "validation/figures")
    parser.add_argument("--pdf", action="store_true", help="also write vector PDF files")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()
    if args.dpi < 72:
        parser.error("--dpi must be at least 72")
    global np, plt
    try:
        import numpy as numpy_module
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot_module
        np, plt = numpy_module, pyplot_module
    except ImportError as error:
        print(f"Plotting dependency missing: {error}. Install numpy and matplotlib in this Python environment.", file=sys.stderr)
        return 2
    args.suite_root, args.output = args.suite_root.resolve(), args.output.resolve()
    try:
        cases, sources = load_cases(args.suite_root)
        section = duct_section(cases["duct16"])
        profiles = {name: axial_profile(cases[name], name) for name in ("adaptive8", "adaptive16")}
        if not np.allclose(profiles["adaptive8"][2], profiles["adaptive16"][2], rtol=0, atol=1e-12):
            raise PlotError("Adaptive refinement extents differ")
        if cases["adaptive8"]["config"]["tolerance"] != cases["adaptive16"]["config"]["tolerance"]:
            raise PlotError("Adaptive stopping tolerances differ")
        suite = read_json(args.suite_root / "suite_results.json")
        manifest = {"created_utc": datetime.now(timezone.utc).isoformat(), "suite_root": str(args.suite_root),
                    "source_suite_passed": suite.get("passed"), "source_full_pass": suite.get("full_pass"),
                    "source_failed_or_missing_cases": suite.get("failed_or_missing_cases"),
                    "source_full_suite_cases_not_run": suite.get("full_suite_cases_not_run"),
                    "scope": "Plots of six completed cases, not a new validation verdict or a full-suite completion claim",
                    "source_sha256": {str(path): digest(path) for path in sources},
                    "script_sha256": digest(Path(__file__)), "figures": {}}
        args.output.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                             "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
        plot_accuracy(cases, args, manifest["figures"])
        plot_duct(cases["duct16"], section, args, manifest["figures"])
        plot_axial(cases, profiles, args, manifest["figures"])
        plot_convergence(cases, args, manifest["figures"])
        for figure in manifest["figures"].values():
            figure["sha256"] = {path: digest(Path(path)) for path in figure["files"]}
        manifest_path = args.output / "plot_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(json.dumps({"figures": {name: record["files"] for name, record in manifest["figures"].items()},
                          "manifest": str(manifest_path), "source_full_pass": manifest["source_full_pass"]}, indent=2))
        return 0
    except (PlotError, OSError, ValueError, KeyError) as error:
        print(f"Cannot generate validation figures: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
