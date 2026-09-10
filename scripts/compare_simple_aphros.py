#!/usr/bin/env python3
"""Compare native-octree SIMPLE CSV dumps with instrumented Aphros dumps.

Examples:
  python scripts/compare_simple_aphros.py --iteration 1
  python scripts/compare_simple_aphros.py --iteration 2 --atol 1e-9 --rtol 1e-6
  python scripts/compare_simple_aphros.py --final --output comparison.json

Iteration N is compared with Aphros simple_(N-1)_b0. Cell and face IDs need
not agree. Fluxes are normalized to the positive coordinate-axis direction;
duplicate periodic faces in Aphros must agree before they can be deduplicated.
Exit codes: 0 = all requested comparisons passed, 1 = numerical/coverage
mismatch, 2 = missing files, invalid schemas, or other incomplete comparison.
This checks numerical agreement, not whether either solver is physically valid.
"""

import argparse
import csv
import json
import math
from pathlib import Path
import sys


class ComparisonError(RuntimeError):
    pass


def read_csv(path, required):
    if not path.is_file():
        raise ComparisonError(f"Required CSV does not exist: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        missing = sorted(set(required) - set(fields))
        if missing:
            raise ComparisonError(f"{path}: missing columns {missing}; found {fields}")
        if len(set(fields)) != len(fields):
            raise ComparisonError(f"{path}: duplicate column names")
        rows = []
        for line, record in enumerate(reader, 2):
            try:
                row = {key: float(record[key]) for key in fields}
            except (ValueError, TypeError) as error:
                raise ComparisonError(f"{path}:{line}: invalid numeric CSV row: {error}") from error
            if not all(math.isfinite(value) for value in row.values()):
                raise ComparisonError(f"{path}:{line}: nonfinite value")
            rows.append(row)
    if not rows:
        raise ComparisonError(f"{path}: no data rows")
    return rows, fields


def read_json(path):
    if not path.is_file():
        raise ComparisonError(f"Required JSON does not exist: {path}")
    try:
        with path.open(encoding="utf-8-sig") as stream:
            return json.load(stream)
    except (ValueError, OSError) as error:
        raise ComparisonError(f"Cannot read {path}: {error}") from error


def quantize(x, tolerance):
    return int(round(x / tolerance))


def cell_index(rows, tolerance, label):
    result = {}
    for row in rows:
        key = tuple(quantize(row[axis], tolerance) for axis in "xyz")
        if key in result:
            raise ComparisonError(f"{label}: duplicate cell coordinate {tuple(row[a] for a in 'xyz')}")
        result[key] = row
    return result


def sample_coordinates(index, keys):
    result = []
    for key in sorted(keys)[:8]:
        row = index[key]
        item = {axis: row[axis] for axis in "xyz"}
        if "axis" in row:
            item["axis"] = int(row["axis"])
        result.append(item)
    return result


def coverage(ours, baseline):
    matched = set(ours) & set(baseline)
    ours_only = set(ours) - set(baseline)
    baseline_only = set(baseline) - set(ours)
    return {
        "ours_unique": len(ours), "baseline_unique": len(baseline),
        "matched": len(matched), "ours_coverage": len(matched) / len(ours),
        "baseline_coverage": len(matched) / len(baseline),
        "ours_only": len(ours_only), "baseline_only": len(baseline_only),
        "ours_only_examples": sample_coordinates(ours, ours_only),
        "baseline_only_examples": sample_coordinates(baseline, baseline_only),
        "passed": not ours_only and not baseline_only,
    }, sorted(matched)


def within(a, b, atol, rtol):
    return abs(a - b) <= atol + rtol * abs(b)


def face_index(rows, bounds, periodic, coord_tol, atol, rtol, flux_columns, label):
    result = {}
    duplicates = 0
    duplicate_maxdiff = {name: 0.0 for name in ["area"] + flux_columns}
    for original in rows:
        row = original.copy()
        axis_id = int(row["axis"])
        if row["axis"] != axis_id or axis_id not in range(3):
            raise ComparisonError(f"{label}: invalid face axis {row['axis']}")
        sign = row.get("sign", 1.0)
        if sign not in (-1.0, 1.0) or row["area"] <= 0:
            raise ComparisonError(f"{label}: invalid face sign or area")
        for column in flux_columns:
            row[column] *= sign
        raw = tuple(row[a] for a in "xyz")
        for dimension, axis in enumerate("xyz"):
            lo, hi = bounds[dimension]
            if row[axis] < lo - coord_tol or row[axis] > hi + coord_tol:
                raise ComparisonError(f"{label}: face {raw} is outside our mesh bounds {bounds}")
            if axis in periodic and (abs(row[axis] - hi) <= coord_tol or abs(row[axis] - lo) <= coord_tol):
                row[axis] = lo
        key = (axis_id,) + tuple(quantize(row[a], coord_tol) for a in "xyz")
        row["_raw_coordinate"] = raw
        if key in result:
            other = result[key]
            # Only periodic image pairs may collapse; identical duplicate CSV
            # records, different mesh levels, and MPI overlap are not ignored.
            image_pair = any(
                axis in periodic and abs(abs(raw[d] - other["_raw_coordinate"][d]) -
                                        (bounds[d][1] - bounds[d][0])) <= coord_tol
                for d, axis in enumerate("xyz"))
            if not image_pair:
                raise ComparisonError(f"{label}: non-periodic duplicate face at {raw}, axis={axis_id}")
            for column in ["area"] + flux_columns:
                difference = abs(row[column] - other[column])
                duplicate_maxdiff[column] = max(duplicate_maxdiff[column], difference)
                if not within(row[column], other[column], atol, rtol):
                    raise ComparisonError(
                        f"{label}: periodic duplicate disagreement at {raw}, axis={axis_id}, "
                        f"{column}: {row[column]} versus {other[column]}")
            duplicates += 1
        else:
            result[key] = row
    return result, {"input_rows": len(rows), "unique_faces": len(result),
                    "periodic_duplicates_removed": duplicates,
                    "periodic_duplicate_max_abs_difference": duplicate_maxdiff}


def summarize(name, ours, reference, labels, atol, rtol):
    if not ours or len(ours) != len(reference) or len(ours) != len(labels):
        raise ComparisonError(f"{name}: empty or inconsistent comparison vectors")
    differences = [abs(a - b) for a, b in zip(ours, reference)]
    worst = max(range(len(ours)), key=differences.__getitem__)
    error_norm = math.sqrt(math.fsum(value * value for value in differences))
    reference_norm = math.sqrt(math.fsum(value * value for value in reference))
    tolerance_ratios = [d / (atol + rtol * abs(b)) if atol + rtol * abs(b) > 0
                        else (0.0 if d == 0 else math.inf)
                        for d, b in zip(differences, reference)]
    return {
        "count": len(ours), "max_abs_difference": differences[worst],
        "rms_difference": error_norm / math.sqrt(len(ours)),
        "relative_l2": error_norm / reference_norm if reference_norm > 0 else None,
        "reference_l2_norm": reference_norm,
        "zero_reference_note": None if reference_norm > 0 else "relative error undefined; absolute tolerance is used",
        "max_tolerance_ratio": max(tolerance_ratios) if math.isfinite(max(tolerance_ratios)) else None,
        "violations": sum(not within(a, b, atol, rtol) for a, b in zip(ours, reference)),
        "worst_location": labels[worst], "ours_at_worst": ours[worst],
        "baseline_at_worst": reference[worst],
        "passed": all(within(a, b, atol, rtol) for a, b in zip(ours, reference)),
    }


def matrix_entries(path, ids):
    rows, _ = read_csv(path, ["row", "column", "value"])
    result = []
    seen = set()
    for entry in rows:
        r, c = int(entry["row"]), int(entry["column"])
        if entry["row"] != r or entry["column"] != c or r not in ids or c not in ids:
            raise ComparisonError(f"{path}: matrix index does not match cell IDs")
        if (r, c) in seen:
            raise ComparisonError(f"{path}: duplicate matrix entry {(r, c)}")
        seen.add((r, c))
        result.append((r, c, entry["value"]))
    return result


def compare(args):
    ours_dir, baseline_dir = args.ours.resolve(), args.aphros.resolve()
    config = read_json(ours_dir / "case.json")
    periodic = set(filter(None, args.periodic_axes.split(","))) if args.periodic_axes is not None else (
        ({"x"} if config.get("periodic_x", True) else set()) |
        ({"z"} if config.get("periodic_z", True) else set()))
    periodic.discard("")
    if periodic - set("xyz"):
        raise ComparisonError("--periodic-axes must be a comma-separated subset of x,y,z")
    is_pinned = bool(config.get("periodic_x", True))
    report = {
        "ours": str(ours_dir), "aphros": str(baseline_dir),
        "mode": "final" if args.final else "iteration", "atol": args.atol, "rtol": args.rtol,
        "coordinate_tolerance": args.coordinate_tolerance, "periodic_axes": sorted(periodic),
        "comparisons": {}, "not_compared": [],
        "normalizations": ["match cell xyz; match face xyz+axis", "face flux uses positive-axis orientation",
                           "periodic endpoint duplicates verified before deduplication",
                           "pressure gauge removed using each solver's volume-weighted mean"],
    }
    if args.final:
        metrics = read_json(ours_dir / "metrics.json")
        final_iteration = int(metrics["iterations"])
        report["ours_final_iteration"] = final_iteration
        report["ours_solver_converged"] = bool(metrics.get("converged", False))
        stage_dir = ours_dir / f"iter_{final_iteration}"
        our_cell_path = ours_dir / "solution.csv"
        base_prefix = "simple_final_b0"
        our_required = ["id", "x", "y", "z", "h", "volume", "u", "v", "w", "p"]
        base_required = ["x", "y", "z", "volume", "u", "v", "w", "p"]
        report["not_compared"].append({"fields": "predictor, momentum, pressure correction",
                                        "reason": "Aphros final CSV contains only final state"})
    else:
        report["ours_iteration"] = args.iteration
        report["aphros_iteration"] = args.iteration - 1
        stage_dir = ours_dir / f"iter_{args.iteration}"
        our_cell_path = stage_dir / "cells.csv"
        base_prefix = f"simple_{args.iteration - 1}_b0"
        our_required = ["id", "x", "y", "z", "h", "volume", "u", "v", "w", "p", "aP",
                        "rhs_u", "rhs_v", "rhs_w", "u_star", "v_star", "w_star", "pressure_rhs", "pressure_correction"]
        base_required = ["x", "y", "z", "volume", "u", "v", "w", "p", "p_previous", "pcorr",
                         "pcorr_rhs", "pcorr_diag", "diag_u", "diag_v", "diag_w",
                         "delta_rhs_u", "delta_rhs_v", "delta_rhs_w", "u_star", "v_star", "w_star"]
    our_rows, our_fields = read_csv(our_cell_path, our_required)
    base_cell_path = baseline_dir / f"{base_prefix}_cells.csv"
    base_rows, base_fields = read_csv(base_cell_path, base_required)
    oi = cell_index(our_rows, args.coordinate_tolerance, str(our_cell_path))
    bi = cell_index(base_rows, args.coordinate_tolerance, str(base_cell_path))
    report["cell_coverage"], keys = coverage(oi, bi)
    if not keys:
        raise ComparisonError("No matching cell coordinates; these dumps are not on the same grid")
    labels = [{a: oi[k][a] for a in "xyz"} for k in keys]
    ids = {int(row["id"]): row for row in our_rows}
    if len(ids) != len(our_rows) or any(row["id"] != int(row["id"]) for row in our_rows):
        raise ComparisonError("Our cell IDs must be unique integers")
    if is_pinned and 0 not in ids:
        raise ComparisonError("Periodic pressure comparison requires gauge cell id=0")
    pressure_keys = [k for k in keys if not is_pinned or int(oi[k]["id"]) != 0]
    if is_pinned:
        report["normalizations"].append("exclude our gauge cell id=0 from pressure RHS and diagonal comparisons")
        report["pressure_gauge_cell"] = {a: ids[0][a] for a in "xyz"}

    def add(name, a, b, where=labels):
        report["comparisons"][name] = summarize(name, a, b, where, args.atol, args.rtol)

    def columns(name, a, b, selected=keys):
        add(name, [oi[k][a] for k in selected], [bi[k][b] for k in selected],
            [{axis: oi[k][axis] for axis in "xyz"} for k in selected])

    def centered(name, left, right, left_index=oi, right_index=bi):
        # Means use the complete domain, never only the matched subset.
        lm = math.fsum(r[left] * r["volume"] for r in left_index.values()) / math.fsum(r["volume"] for r in left_index.values())
        rm = math.fsum(r[right] * r["volume"] for r in right_index.values()) / math.fsum(r["volume"] for r in right_index.values())
        add(name, [left_index[k][left] - lm for k in keys], [right_index[k][right] - rm for k in keys])
        report["comparisons"][name]["removed_means"] = {"ours": lm, "aphros": rm}

    columns("cell.volume", "volume", "volume")
    for component_name in "uvw":
        columns(f"cell.{component_name}", component_name, component_name)
    centered("cell.p_gauge_removed", "p", "p")
    if not is_pinned:
        columns("cell.p_absolute", "p", "p")

    if not args.final:
        for component_name in "uvw":
            columns(f"cell.{component_name}_star", f"{component_name}_star", f"{component_name}_star")
            columns(f"momentum.diag_{component_name}", "aP", f"diag_{component_name}")
        centered("pressure.correction_gauge_removed", "pressure_correction", "pcorr")
        if not is_pinned:
            columns("pressure.correction_absolute", "pressure_correction", "pcorr")
        columns("pressure.rhs_excluding_gauge", "pressure_rhs", "pcorr_rhs", pressure_keys)
        pressure_entries = matrix_entries(stage_dir / "pressure_matrix.csv", ids)
        pressure_diagonal = {r: value for r, c, value in pressure_entries if r == c}
        if set(pressure_diagonal) != set(ids):
            raise ComparisonError("Pressure matrix does not contain exactly one diagonal per cell")
        add("pressure.diag_excluding_gauge", [pressure_diagonal[int(oi[k]["id"])] for k in pressure_keys],
            [bi[k]["pcorr_diag"] for k in pressure_keys],
            [{a: oi[k][a] for a in "xyz"} for k in pressure_keys])
        previous_path = ours_dir / f"iter_{args.iteration - 1}" / "cells.csv"
        if not args.skip_delta_rhs or previous_path.is_file():
            previous_rows, _ = read_csv(previous_path, ["x", "y", "z", "volume", "u", "v", "w", "p"])
            old = cell_index(previous_rows, args.coordinate_tolerance, str(previous_path))
            report["previous_cell_coverage"], _ = coverage(oi, old)
            if set(old) != set(oi):
                raise ComparisonError("Previous iteration mesh differs; delta RHS cannot be compared without remapping")
            centered("cell.p_previous_gauge_removed", "p", "p_previous", old, bi)
            if not args.skip_delta_rhs:
                entries = matrix_entries(stage_dir / "momentum_matrix.csv", ids)
                for d in "uvw":
                    old_by_id = {int(oi[k]["id"]): old[k][d] for k in oi}
                    au = {cell_id: 0.0 for cell_id in ids}
                    for r, c, value in entries:
                        au[r] += value * old_by_id[c]
                    add(f"momentum.delta_rhs_{d}",
                        [oi[k][f"rhs_{d}"] - au[int(oi[k]["id"])] for k in keys],
                        [bi[k][f"delta_rhs_{d}"] for k in keys])
                report["normalizations"].append("Aphros delta RHS compared with our absolute RHS - A_relaxed*u_previous")
        if args.skip_delta_rhs:
            report["not_compared"].append({"fields": "momentum.delta_rhs_u/v/w", "reason": "explicit --skip-delta-rhs"})
            if not previous_path.is_file():
                report["not_compared"].append({"fields": "cell.p_previous", "reason": f"previous dump unavailable: {previous_path}"})
    bounds = [(min(r[a] - 0.5 * r["h"] for r in our_rows),
               max(r[a] + 0.5 * r["h"] for r in our_rows)) for a in "xyz"]
    our_face_path = stage_dir / "faces.csv"
    base_face_path = baseline_dir / f"{base_prefix}_faces.csv"
    our_flux_columns = ["flux"] if args.final else ["flux", "predicted_flux"]
    base_flux_columns = ["flux"] if args.final else ["corrected_flux", "predicted_flux"]
    our_face_rows, our_face_fields = read_csv(our_face_path, ["x", "y", "z", "axis", "sign", "area"] + our_flux_columns)
    base_face_rows, base_face_fields = read_csv(base_face_path, ["x", "y", "z", "axis", "area"] + base_flux_columns)
    wall_coverage_passed = True
    if args.embedded_walls:
        if not config.get("embedded_geometry") or config.get("adaptive", False):
            raise ComparisonError("--embedded-walls requires a same-grid, uniform embedded case")
        if not {"owner", "neighbor", "boundary"} <= set(our_face_fields):
            raise ComparisonError("Embedded face dump lacks owner/neighbor/boundary")
        walls = [r for r in our_face_rows if r["neighbor"] < 0]
        if not walls or any(r["boundary"] != 1 or r["owner"] not in ids for r in walls):
            raise ComparisonError("Expected stationary no-slip embedded walls with valid owners")
        if len({r["owner"] for r in walls}) != len(walls):
            raise ComparisonError("Expected one embedded wall per cut cell")
        if any(r[col] != 0 for r in walls for col in our_flux_columns):
            raise ComparisonError("Stationary embedded-wall predicted/corrected flux must be exactly zero")
        wall_path = baseline_dir / "tube_b0_geometry_walls.csv"
        base_walls, _ = read_csv(wall_path, ["i", "j", "k", "x", "y", "z", "area"])
        wi = cell_index(walls, args.coordinate_tolerance, "our embedded walls")
        wbi = cell_index(base_walls, args.coordinate_tolerance, str(wall_path))
        report["embedded_wall_coverage"], wall_keys = coverage(wi, wbi)
        wall_coverage_passed = report["embedded_wall_coverage"]["passed"]
        for k in wall_keys:
            owner = ids[int(wi[k]["owner"])]
            if any(abs(owner[a] / owner["h"] - .5 - wbi[k][ijk]) > 1e-8 for a, ijk in zip("xyz", "ijk")):
                raise ComparisonError("Embedded wall matched a different owner; translated/adaptive dumps require remapping")
        add("wall.area", [wi[k]["area"] for k in wall_keys], [wbi[k]["area"] for k in wall_keys],
            [{a: wi[k][a] for a in "xyz"} for k in wall_keys])
        report["embedded_wall_boundary"] = {"count": len(walls), "exact_zero_flux": True,
            "baseline_geometry": str(wall_path), "policy": "Validate wall geometry/owners and zero flux separately; Aphros face CSV contains Cartesian faces only"}
        our_face_rows = [r for r in our_face_rows if r["neighbor"] >= 0]
        if any(r["boundary"] != 0 for r in our_face_rows):
            raise ComparisonError("Unexpected non-internal Cartesian face in embedded benchmark")
    of, od = face_index(our_face_rows, bounds, periodic, args.coordinate_tolerance, args.atol, args.rtol, our_flux_columns, str(our_face_path))
    bf, bd = face_index(base_face_rows, bounds, periodic, args.coordinate_tolerance, args.atol, args.rtol, base_flux_columns, str(base_face_path))
    report["face_coverage"], face_keys = coverage(of, bf)
    report["face_deduplication"] = {"ours": od, "aphros": bd}
    face_labels = [{**{a: of[k][a] for a in "xyz"}, "axis": int(of[k]["axis"])} for k in face_keys]
    for name, a, b in [("face.area", "area", "area"), ("face.corrected_flux", "flux", "flux" if args.final else "corrected_flux")]:
        add(name, [of[k][a] for k in face_keys], [bf[k][b] for k in face_keys], face_labels)
    if not args.final:
        add("face.predicted_flux", [of[k]["predicted_flux"] for k in face_keys], [bf[k]["predicted_flux"] for k in face_keys], face_labels)
    # Extra columns are listed, not silently treated as matched numerical fields.
    report["schema_audit"] = {
        "ours_cell_columns": our_fields, "aphros_cell_columns": base_fields,
        "ours_face_columns": our_face_fields, "aphros_face_columns": base_face_fields,
        "ours_extra_cell_columns": sorted(set(our_fields) - set(our_required)),
        "aphros_extra_cell_columns": sorted(set(base_fields) - set(base_required)),
        "ours_extra_face_columns": sorted(set(our_face_fields) - set(["x", "y", "z", "axis", "sign", "area"] + our_flux_columns)),
        "aphros_extra_face_columns": sorted(set(base_face_fields) - set(["x", "y", "z", "axis", "area"] + base_flux_columns)),
        "extra_column_policy": "listed for inspection; IDs/topology/exact-solution columns have no baseline numeric counterpart",
    }
    report["files"] = {"ours_cells": str(our_cell_path), "ours_faces": str(our_face_path),
                       "aphros_cells": str(base_cell_path), "aphros_faces": str(base_face_path)}
    report["failed_comparisons"] = [name for name, data in report["comparisons"].items() if not data["passed"]]
    report["passed"] = (not report["failed_comparisons"] and report["cell_coverage"]["passed"]
                        and report["face_coverage"]["passed"]
                        and wall_coverage_passed
                        and (not args.final or report["ours_solver_converged"]))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ours", type=Path, default=Path(__file__).resolve().parents[1] / "output" / "simple_uniform8")
    parser.add_argument("--aphros", "--baseline", dest="aphros", type=Path,
                        default=Path("D:/Dropbox/Agent-simulation/simple-baseline/periodic_3d_n8"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--iteration", type=int, default=1)
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--atol", type=float, default=1e-9)
    parser.add_argument("--rtol", type=float, default=1e-6)
    parser.add_argument("--coordinate-tolerance", type=float, default=1e-10)
    parser.add_argument("--periodic-axes", help="comma-separated axes; default follows case.json (x/z or z)")
    parser.add_argument("--skip-delta-rhs", action="store_true", help="explicitly omit delta RHS if previous iteration was not dumped")
    parser.add_argument("--embedded-walls", action="store_true", help="Validate same-grid cut walls separately against Aphros geometry and require exact zero wall flux")
    parser.add_argument("--output", type=Path, help="also save the JSON report to this file")
    args = parser.parse_args()
    if args.iteration < 1 or not all(math.isfinite(x) and x >= 0 for x in [args.atol, args.rtol]) or not (
            math.isfinite(args.coordinate_tolerance) and args.coordinate_tolerance > 0):
        parser.error("iteration must be >=1, tolerances finite and nonnegative, coordinate tolerance positive")
    try:
        report = compare(args)
        code = 0 if report["passed"] else 1
    except (ComparisonError, OSError, KeyError, ValueError) as error:
        report = {"passed": False, "incomplete": True, "error": str(error)}
        code = 2
    text = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return code


if __name__ == "__main__":
    sys.exit(main())
