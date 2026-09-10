#!/usr/bin/env python3
"""Run and audit the fixed SIMPLE validation suite (Python standard library).

Default: quick suite uniform8,perturb8,pressure8,duct8. --full adds seven cases.
Examples:
  python scripts/validate_simple.py --output output/simple_validation
  python scripts/validate_simple.py --full --output output/simple_validation_full
  python scripts/validate_simple.py --cases adaptive8,adaptive16 --output output/adaptive_validation
  python scripts/validate_simple.py --analyze-only --output output/simple_validation

Analyze-only accepts results produced by this harness: case.json, its recorded
configuration, executable SHA256, and output hashes must agree. An explicit
--allow-different-binary permits auditing a recorded older binary but cannot
produce a strict current-binary quick_pass/full_pass. It does not waive missing
provenance, changed output files, or a different case configuration.

No numerical threshold is changed in response to a failed run. Existing case
directories can be rerun, but old outputs cannot make a failed process pass.
Return 0 only when every requested case passes; otherwise return 1 (CLI errors 2).
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
QUICK = ("uniform8", "perturb8", "pressure8", "duct8")
FULL = QUICK + ("uniform16", "adaptive8", "adaptive16", "duct16", "convective8", "accelerated8", "adaptive_convective8")
BASELINES = {
    "uniform8": ("periodic_3d_n8", (1, 2, "final")),
    "perturb8": ("perturb_3d_n8", (1, 2, "final")),
    "duct8": ("duct_3d_n8", ("final",)),
    "duct16": ("duct_3d_n16", ("final",)),
    "accelerated8": ("periodic_3d_n8", ("final",)),
}
COMMON_LIMITS = {
    "momentum_relative_l2": 1e-8,
    "continuity_relative_linf": 1e-8,
    "nonorth_defect_relative_linf": 1e-8,
    "nonorth_face_defect_relative_linf": 1e-8,
    "flux_change_relative_linf": 1e-8,
    "pressure_change_relative_linf": 1e-8,
    "cross_section_flux_relative_spread": 1e-7,
}


class ValidationError(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_sha(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path):
    if not path.is_file():
        raise ValidationError(f"Missing required file: {path}")
    try:
        def reject_constant(token):
            raise ValueError(f"nonfinite JSON constant {token}")
        return json.loads(path.read_text(encoding="utf-8-sig"), parse_constant=reject_constant)
    except (ValueError, OSError) as error:
        raise ValidationError(f"Cannot read {path}: {error}") from error


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replacement keeps suite progress readable if a long run is stopped.
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_provenance():
    result = {"repository": str(REPO)}
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                                capture_output=True, text=True, check=True)
        status = subprocess.run(["git", "status", "--porcelain=v1"], cwd=REPO,
                                capture_output=True, text=True, check=True)
        diff = subprocess.run(["git", "diff", "HEAD", "--binary"], cwd=REPO,
                              capture_output=True, check=True)
        result.update(commit=commit.stdout.strip(), dirty=bool(status.stdout.strip()),
                      status_porcelain=status.stdout.splitlines(),
                      tracked_diff_sha256=hashlib.sha256(diff.stdout).hexdigest())
    except (OSError, subprocess.CalledProcessError) as error:
        result["error"] = str(error)
    return result


def make_config(name, directory):
    if name not in FULL:
        raise ValidationError(f"Unknown case {name}; choose from {','.join(FULL)}")
    config = {
        "ny": 16 if name.endswith("16") else 8,
        "adaptive": name.startswith("adaptive"), "periodic_x": name != "pressure8",
        "periodic_z": not name.startswith("duct"), "quadratic_interfaces": True,
        "rho": 1.0, "nu": 0.01, "alpha_u": 0.7, "alpha_p": 0.3,
        "tolerance": 1e-8, "linear_tolerance": 1e-11, "nonorth_iterations": 8,
        "pressure_solver": "auto",
        "max_iterations": 5000, "convection": name in ("convective8", "adaptive_convective8"),
        "anderson_depth": 5 if name in ("uniform16", "adaptive8", "adaptive16", "duct16", "accelerated8") else 0,
        "flux_relaxation_memory": False, "initial_perturbation": 0.01 if name == "perturb8" else 0.0,
        "pressure_in": 1.0, "pressure_out": 0.0,
        "force": [0.0, 0.0, 0.0] if name == "pressure8" else [1.0, 0.0, 0.0],
        "dump_iterations": [0, 1, 2, 3, 10, 100], "output": str(directory.resolve()),
    }
    return config


def limits_for(name):
    limits = dict(COMMON_LIMITS)
    if name.startswith("adaptive"):
        limits.update(velocity_relative_l2=0.005, flow_rate_relative_error=0.005,
                      wall_shear_relative_error=0.01)
    elif name == "duct8":
        limits.update(velocity_relative_l2=0.01, flow_rate_relative_error=0.015)
    elif name == "duct16":
        limits.update(velocity_relative_l2=0.005, flow_rate_relative_error=0.005,
                      wall_shear_relative_error=0.01)
    else:
        # Pressure8 deliberately uses the same fixed 1e-6 velocity standard.
        limits.update(velocity_relative_l2=1e-6,
                      flow_rate_relative_error=0.0021 if name == "uniform16" else 0.008)
    if name == "pressure8":
        limits["pressure_absolute_l2"] = 1e-6
    return limits


def baseline_requirements(name, root):
    if name not in BASELINES:
        return []
    folder, stages = BASELINES[name]
    paths = []
    for stage in stages:
        prefix = "simple_final_b0" if stage == "final" else f"simple_{stage - 1}_b0"
        paths.extend(root / folder / f"{prefix}_{kind}.csv" for kind in ("cells", "faces"))
    return paths


def expected_artifacts(directory, metrics, name):
    iteration = metrics.get("iterations")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise ValidationError("metrics.json does not contain a positive integer iterations")
    files = [directory / leaf for leaf in ("case.json", "metrics.json", "operator_checks.json", "solution.csv", "sections.csv", "history.csv", "native_fields.bin")]
    dump_stages = {0, 1, 2, 3, 10, 100, iteration}
    dump_stages = {stage for stage in dump_stages if stage <= iteration}
    for stage in sorted(dump_stages):
        files.extend(directory / f"iter_{stage}" / leaf for leaf in ("cells.csv", "faces.csv"))
    if name in ("uniform8", "perturb8"):
        for stage in (1, 2):
            files.extend(directory / f"iter_{stage}" / leaf for leaf in ("momentum_matrix.csv", "pressure_matrix.csv"))
    return files


def check_metrics(metrics, config, name):
    checks = {}
    checks["solver_converged"] = {"value": metrics.get("converged"), "required": True,
                                  "passed": metrics.get("converged") is True}
    for key, threshold in limits_for(name).items():
        value = metrics.get(key)
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
        checks[key] = {"value": value if valid else None, "operator": "<", "threshold": threshold,
                       "passed": valid and value < threshold,
                       "error": None if valid else "missing, nonnumeric, negative, or nonfinite metric"}
    for key in ("ny", "adaptive", "periodic_x", "periodic_z", "quadratic_interfaces", "anderson_depth", "convection", "rho", "nu", "alpha_u", "alpha_p"):
        checks[f"config.{key}"] = {"value": metrics.get(key), "required": config[key],
                                   "passed": key in metrics and metrics[key] == config[key]}
    coarse_fine = metrics.get("coarse_fine_faces")
    valid_count = isinstance(coarse_fine, int) and not isinstance(coarse_fine, bool)
    checks["native_coarse_fine_topology"] = {
        "value": coarse_fine, "required": ">0" if config["adaptive"] else "0",
        "passed": valid_count and (coarse_fine > 0 if config["adaptive"] else coarse_fine == 0)}
    checks["native_mesh_backend"] = {"value": metrics.get("mesh_backend"),
        "required": "HADeviceGrid<Tile>", "passed": "HADeviceGrid<Tile>" in str(metrics.get("mesh_backend", ""))}
    checks["config.pressure_solver"] = {"value": metrics.get("pressure_solver_requested"),
        "required": config["pressure_solver"],
        "passed": metrics.get("pressure_solver_requested") == config["pressure_solver"]}
    expected_backend = "ldlt" if not config["convection"] and metrics.get("cells", 0) >= 100000 else "cg"
    checks["pressure_linear_solver"] = {"value": metrics.get("pressure_linear_solver"),
        "required": expected_backend, "passed": metrics.get("pressure_linear_solver") == expected_backend}
    if config["anderson_depth"]:
        accepted = metrics.get("anderson_accepted_steps")
        checks["acceleration_exercised"] = {"value": accepted, "required": ">0",
            "passed": isinstance(accepted, int) and not isinstance(accepted, bool) and accepted > 0}
    return checks


def check_operator_report(report, metrics, config):
    """Require the real operator-report schema, independently inspect its gates.

    Replicated low-order quadratic flux errors are characterization data and
    have no pass threshold. The production quadratic reconstruction has exact
    polynomial gates. An inapplicable uniform-grid module is explicitly checked
    to have zero active cells rather than presented as an exercised interface.
    """
    checks = {}

    def lookup(path):
        value = report
        for key in path.split("."):
            if not isinstance(value, dict) or key not in value:
                raise ValidationError(f"operator_checks.json missing required metric: {path}")
            value = value[key]
        return value

    def number(path, minimum=0):
        value = lookup(path)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < minimum:
            raise ValidationError(f"operator_checks.json invalid numeric metric: {path}={value}")
        return value

    def count(path):
        value = number(path)
        if not isinstance(value, int):
            raise ValidationError(f"operator_checks.json requires integer count: {path}")
        return value

    def limit(path, threshold):
        value = number(path)
        checks[path] = {"value": value, "operator": "<", "threshold": threshold, "passed": value < threshold}

    def equal(path, expected):
        value = lookup(path)
        checks[path] = {"value": value, "required": expected, "passed": value == expected and type(value) is type(expected)}

    def norms(path, expected_count=None):
        data = {key: number(f"{path}.{key}") for key in ("L1", "L2", "Linf", "area")}
        data["count"] = count(f"{path}.count")
        if expected_count is not None and data["count"] != expected_count:
            raise ValidationError(f"operator_checks.json {path}.count does not match face-class coverage")
        if (data["count"] == 0 and any(data[key] != 0 for key in ("L1", "L2", "Linf", "area"))) or (data["count"] > 0 and data["area"] <= 0):
            raise ValidationError(f"operator_checks.json inconsistent empty/nonempty norm: {path}")
        # Area-weighted nonnegative error norms must obey L1 <= L2 <= Linf.
        if data["L1"] > data["L2"] * (1 + 1e-10) + 1e-30 or data["L2"] > data["Linf"] * (1 + 1e-10) + 1e-30:
            raise ValidationError(f"operator_checks.json inconsistent norm ordering: {path}")
        return data

    equal("passed", True)
    equal("failures", [])
    for field in ("cells", "faces", "coarse_fine_faces"):
        count(field)
        equal(field, metrics[field])
    equal("periodic_z", config["periodic_z"])
    periodic_count = count("periodic_faces_checked")
    checks["periodic_faces_exercised"] = {"value": periodic_count, "required": ">0", "passed": periodic_count > 0}
    limit("first_geometric_moment_relative_linf", 1e-11)
    limit("random_flux_global_conservation_relative_error", 1e-13)
    for field in ("constant", "affine_axis_0", "affine_axis_1", "affine_axis_2", "mixed_affine"):
        scale = math.sqrt(0.71**2 + 1.31**2 + 2.17**2) if field == "mixed_affine" else 1.0
        for metric, tolerance in (("least_squares_gradient_linf", 1e-10),
                                  ("face_interpolation_linf", 1e-11), ("compact_normal_gradient_linf", 1e-10)):
            limit(f"affine.{field}.{metric}", tolerance * scale)
    pressure = "pressure_correction_identity"
    limit(f"{pressure}.Lp_minus_D_orthogonal_flux_relative_linf", 1e-12)
    limit(f"{pressure}.corrected_divergence_identity_relative_linf", 1e-12)
    limit(f"{pressure}.matrix_relative_symmetry_error", 1e-13)
    boundary_count = count(f"{pressure}.homogeneous_pressure_boundary_faces")
    checks["pressure_boundary_type"] = {"value": boundary_count,
        "required": "0" if config["periodic_x"] else ">0",
        "passed": boundary_count == 0 if config["periodic_x"] else boundary_count > 0}
    if boundary_count == 0:
        limit(f"{pressure}.row_sum_linf", 1e-12)
    else:
        number(f"{pressure}.row_sum_linf")
    energy = number(f"{pressure}.random_pressure_energy", minimum=-1e-12)
    checks["pressure_nonnegative_energy"] = {"value": energy, "operator": ">=", "threshold": -1e-12, "passed": energy >= -1e-12}
    quadratic_fields = ("y_times_H_minus_y", "x_times_y", "x_squared_plus_y_squared_plus_z_squared", "full_mixed_quadratic")
    for field in quadratic_fields:
        for gradient in ("exact_cell_gradients", "least_squares_cell_gradients"):
            for face_class in ("uniform", "coarse_fine"):
                expected_count = metrics["coarse_fine_faces"] if face_class == "coarse_fine" else None
                for metric in ("normal_derivative_error", "unit_diffusivity_integrated_flux_error"):
                    norms(f"quadratic_flux_errors.{field}.{gradient}.{face_class}.{metric}", expected_count)
    module = "quadratic_reconstruction_module"
    active = count(f"{module}.active_cells")
    equal(f"{module}.applicable", config["adaptive"])
    checks["quadratic_active_cells"] = {"value": active, "required": ">0" if config["adaptive"] else "0",
                                        "passed": active > 0 if config["adaptive"] else active == 0}
    count(f"{module}.minimum_stencil_samples")
    count(f"{module}.maximum_stencil_samples")
    number(f"{module}.maximum_weighted_design_condition_number")
    limit(f"{module}.production_cell_values_channel_gradient_linf", 1e-10)
    limit(f"{module}.production_cell_values_channel_interpolation_linf", 1e-11)
    for field in quadratic_fields:
        prefix = f"{module}.fields.{field}"
        for metric, tolerance in (("reconstructed_gradient_linf", 1e-10),
                                  ("reconstructed_hessian_linf", 1e-9), ("two_sided_taylor_face_value_linf", 1e-11)):
            limit(f"{prefix}.{metric}", tolerance)
        norms(f"{prefix}.coarse_fine_normal_derivative_error", metrics["coarse_fine_faces"])
        norms(f"{prefix}.coarse_fine_unit_diffusivity_flux_error", metrics["coarse_fine_faces"])
        limit(f"{prefix}.coarse_fine_normal_derivative_error.Linf", 1e-10)
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks,
            "quadratic_truncation_policy": "Low-order quadratic flux norms require complete valid structure/coverage; no accuracy threshold is imposed on characterization data",
            "scope": lookup("scope")}


def compare_baseline(name, directory, args):
    if name not in BASELINES:
        return {"required": False, "reason": "Analytic/reference-convergence test; no identical-grid Aphros CSV configured for this case",
                "checks": [], "passed": True}
    folder, stages = BASELINES[name]
    baseline_dir = args.baseline_root / folder
    axes = "x" if name.startswith("duct") else "x,z"
    output = {"required": True, "directory": str(baseline_dir), "checks": []}
    for stage in stages:
        final = stage == "final"
        target = directory / ("comparison_final.json" if final else f"comparison_iter{stage}.json")
        command = [sys.executable, str(REPO / "scripts" / "compare_simple_aphros.py"),
                   "--ours", str(directory), "--baseline", str(baseline_dir),
                   "--atol", "1e-8" if final else "1e-11", "--rtol", "1e-6" if final else "1e-8",
                   "--periodic-axes", axes, "--output", str(target)]
        command.extend(["--final"] if final else ["--iteration", str(stage)])
        start = time.perf_counter()
        process = subprocess.run(command, cwd=REPO, capture_output=True, text=True,
                                 encoding="utf-8", errors="replace")
        record = {"stage": stage, "command": command, "returncode": process.returncode,
                  "elapsed_seconds": time.perf_counter() - start, "report_file": str(target)}
        try:
            comparison = json.loads(process.stdout)
            record["report"] = comparison
            record["passed"] = process.returncode == 0 and comparison.get("passed") is True
        except ValueError:
            record.update(passed=False, error="Comparison script did not return valid JSON",
                          stdout=process.stdout[-4000:])
        if process.stderr:
            record["stderr"] = process.stderr[-4000:]
        output["checks"].append(record)
    output["passed"] = all(check["passed"] for check in output["checks"])
    return output


def execute_case(name, config, args, executable_sha, source):
    directory = Path(config["output"])
    directory.mkdir(parents=True, exist_ok=True)
    if sha256(args.exe) != executable_sha:
        raise ValidationError("Executable changed since suite start; refusing to mix binaries")
    case_path = directory / "run_case.json"
    write_json(case_path, config)
    metrics_path = directory / "metrics.json"
    previous_mtime = metrics_path.stat().st_mtime_ns if metrics_path.exists() else None
    manifest = {"schema_version": 1, "case": name, "configuration": config,
                "configuration_sha256": config_sha(config), "executable": str(args.exe),
                "executable_sha256": executable_sha, "source": source,
                "started_utc": utc_now(), "status": "running", "artifacts_sha256": {}}
    manifest_path = directory / "run_manifest.json"
    write_json(manifest_path, manifest)
    command = [str(args.exe), str(case_path)]
    started = time.perf_counter()
    try:
        with (directory / "solver.log").open("w", encoding="utf-8") as log:
            process = subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=args.timeout if args.timeout > 0 else None)
        manifest["returncode"] = process.returncode
    except subprocess.TimeoutExpired:
        manifest.update(returncode=None, error=f"Solver exceeded {args.timeout} seconds; child process terminated")
    except OSError as error:
        manifest.update(returncode=None, error=str(error))
    manifest.update(command=command, elapsed_seconds=time.perf_counter() - started,
                    finished_utc=utc_now(), status="finished")
    manifest["executable_sha256_after_run"] = sha256(args.exe)
    manifest["metrics_freshly_written"] = metrics_path.is_file() and metrics_path.stat().st_mtime_ns != previous_mtime
    if manifest["returncode"] == 0 and manifest["metrics_freshly_written"]:
        try:
            metrics = read_json(metrics_path)
            for artifact in expected_artifacts(directory, metrics, name):
                if artifact.is_file():
                    manifest["artifacts_sha256"][str(artifact.relative_to(directory))] = sha256(artifact)
                else:
                    manifest.setdefault("missing_artifacts", []).append(str(artifact))
        except ValidationError as error:
            manifest["error"] = str(error)
    write_json(manifest_path, manifest)
    return manifest


def audit_case(name, config, args, executable_sha, source):
    directory = Path(config["output"])
    record = {"case": name, "directory": str(directory), "expected_configuration": config,
              "fixed_metric_limits": limits_for(name), "passed": False, "errors": []}
    if name == "convective8":
        record["scope_note"] = "Convection enabled on developed straight flow; this does not independently validate a nonzero steady convective derivative"
    if name == "duct8":
        record["scope_note"] = "Coarse-grid characterization: 1% velocity / 1.5% flow gates; duct16 has tighter target gates"
    if name == "accelerated8":
        record["scope_note"] = "Stokes-only Anderson depth 5; final solution must match the same unaccelerated Aphros reference used by uniform8; iteration states need not coincide"
    required = baseline_requirements(name, args.baseline_root)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        record.update(status="missing_baseline", missing_baseline_files=missing)
        record["errors"].append("Required Aphros data missing. Follow validation/aphros/README.md and its build_baseline.ps1/run_baseline.ps1; no baseline checks were skipped.")
        return record
    record["baseline_input_sha256"] = {str(path): sha256(path) for path in required}
    try:
        if args.analyze_only:
            manifest = read_json(directory / "run_manifest.json")
        else:
            print(f"Running {name}; solver log: {directory / 'solver.log'}", flush=True)
            manifest = execute_case(name, config, args, executable_sha, source)
        record["execution"] = manifest
        if manifest.get("case") != name or manifest.get("configuration") != config or manifest.get("configuration_sha256") != config_sha(config):
            raise ValidationError("Recorded case/configuration does not match this fixed suite. Rerun the case through this harness.")
        recorded_sha = manifest.get("executable_sha256")
        if not isinstance(recorded_sha, str) or len(recorded_sha) != 64:
            raise ValidationError("Missing recorded executable SHA256; existing unrecorded results cannot be adopted by analyze-only")
        changed = recorded_sha != executable_sha
        record["current_binary_matches_run"] = not changed
        if changed and not (args.analyze_only and args.allow_different_binary):
            raise ValidationError("Executable SHA256 differs from the recorded run. Rerun, or explicitly use --analyze-only --allow-different-binary to audit that older binary.")
        if changed:
            record["binary_override"] = "Explicitly allowed historical binary; not a strict validation of the current executable"
        if manifest.get("executable_sha256_after_run") != recorded_sha:
            raise ValidationError("Executable changed during the recorded solver run")
        if manifest.get("status") != "finished" or manifest.get("returncode") != 0:
            raise ValidationError(f"Solver did not finish successfully: returncode={manifest.get('returncode')}; {manifest.get('error', 'see solver.log')}")
        if manifest.get("metrics_freshly_written") is not True:
            raise ValidationError("Run has no proof that metrics.json was freshly written; stale output cannot pass")
        actual_config = read_json(directory / "case.json")
        if actual_config != config:
            raise ValidationError("Solver-written case.json differs from the fixed requested configuration")
        metrics = read_json(directory / "metrics.json")
        recorded_artifacts = manifest.get("artifacts_sha256", {})
        for path in expected_artifacts(directory, metrics, name):
            relative = str(path.relative_to(directory))
            if not path.is_file():
                raise ValidationError(f"Missing required solver artifact: {path}")
            if recorded_artifacts.get(relative) != sha256(path):
                raise ValidationError(f"Artifact has changed or its hash was not recorded: {path}")
        record["metrics"] = metrics
        record["metric_checks"] = check_metrics(metrics, config, name)
        if config["anderson_depth"]:
            # Supplemental diagnostic files are hashed at audit time; the core
            # numerical outputs above must retain their recorded run-time hashes.
            record["acceleration_diagnostic_sha256_at_audit"] = {
                leaf: sha256(directory / leaf) for leaf in ("acceleration.csv", "dump_semantics.json")}
        with (directory / "history.csv").open(newline="", encoding="utf-8") as stream:
            last = None
            for last in csv.DictReader(stream):
                pass
        if last is None or int(last["iteration"]) != metrics["iterations"]:
            raise ValidationError("Last history iteration does not match final metrics")
        change = float(last["velocity_change_relative_linf"])
        record["metric_checks"]["history.velocity_change_relative_linf"] = {
            "value": change, "operator": "<", "threshold": config["tolerance"],
            "passed": math.isfinite(change) and 0 <= change < config["tolerance"]}
        operators = read_json(directory / "operator_checks.json")
        record["operator_checks"] = check_operator_report(operators, metrics, config)
        record["baseline_comparison"] = compare_baseline(name, directory, args)
        record["failed_metric_checks"] = [key for key, check in record["metric_checks"].items() if not check["passed"]]
        record["passed"] = not record["failed_metric_checks"] and record["operator_checks"]["passed"] and record["baseline_comparison"]["passed"]
        record["status"] = "passed" if record["passed"] else "failed_checks"
    except (ValidationError, OSError, ValueError, KeyError) as error:
        record["status"] = "incomplete_or_failed_execution"
        record["errors"].append(str(error))
    return record


def update_summary(suite, selected):
    completed = {item["case"]: item for item in suite["cases"]}
    suite["requested_cases_not_finished"] = [name for name in selected if name not in completed]
    suite["failed_or_missing_cases"] = [name for name, item in completed.items() if not item["passed"]]
    suite["full_suite_cases_not_run"] = [name for name in FULL if name not in completed]
    strict = lambda name: name in completed and completed[name]["passed"] and completed[name].get("current_binary_matches_run") is True
    suite["refinement_checks"] = {}
    for coarse, fine, keys in (
            ("uniform8", "uniform16", ("flow_rate_relative_error",)),
            ("adaptive8", "adaptive16", ("velocity_relative_l2", "flow_rate_relative_error")),
            ("duct8", "duct16", ("velocity_relative_l2", "flow_rate_relative_error"))):
        for key in keys:
            label = f"{coarse}->{fine}.{key}"
            check = {"coarse_case": coarse, "fine_case": fine, "metric": key,
                     "required": "fine error < coarse error", "passed": None}
            if coarse in completed and fine in completed:
                a = completed[coarse].get("metrics", {}).get(key)
                b = completed[fine].get("metrics", {}).get(key)
                valid = all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and math.isfinite(v) and v >= 0 for v in (a, b))
                check.update(coarse_error=a, fine_error=b, passed=valid and b < a)
            suite["refinement_checks"][label] = check
    suite["failed_refinement_checks"] = [key for key, value in suite["refinement_checks"].items()
                                          if value["passed"] is False]
    suite["passed"] = not suite["errors"] and not suite["requested_cases_not_finished"] and not suite["failed_or_missing_cases"] and not suite["failed_refinement_checks"]
    suite["strict_current_binary_pass"] = suite["passed"] and all(strict(name) for name in selected)
    suite["quick_pass"] = not suite["errors"] and all(strict(name) for name in QUICK)
    suite["full_pass"] = not suite["errors"] and all(strict(name) for name in FULL) and all(
        check["passed"] is True for check in suite["refinement_checks"].values())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exe", type=Path, default=REPO / "build/windows/x64/release/simple_channel.exe")
    parser.add_argument("--output", type=Path, default=REPO / "output/simple_validation")
    parser.add_argument("--baseline-root", type=Path, default=Path("D:/Dropbox/Agent-simulation/simple-baseline"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--quick", action="store_true", help="default four-case suite")
    mode.add_argument("--full", action="store_true", help="all eleven defined cases")
    parser.add_argument("--cases", help="explicit comma-separated case names; reports a custom subset")
    parser.add_argument("--analyze-only", action="store_true")
    parser.add_argument("--allow-different-binary", action="store_true", help="explicit historical-binary audit override; analyze-only required")
    parser.add_argument("--timeout", type=float, default=0, help="per-case solver timeout in seconds; 0 has no limit")
    args = parser.parse_args()
    if args.allow_different_binary and not args.analyze_only:
        parser.error("--allow-different-binary requires --analyze-only")
    if not math.isfinite(args.timeout) or args.timeout < 0:
        parser.error("--timeout must be finite and nonnegative")
    if args.cases and (args.quick or args.full):
        parser.error("choose --cases or --quick/--full, not both")
    selected = tuple(name.strip() for name in args.cases.split(",")) if args.cases else (FULL if args.full else QUICK)
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(FULL):
        parser.error(f"--cases must contain distinct names from {','.join(FULL)}")
    args.exe, args.output, args.baseline_root = args.exe.resolve(), args.output.resolve(), args.baseline_root.resolve()
    suite = {"schema_version": 1, "started_utc": utc_now(), "mode": "analyze_only" if args.analyze_only else "execute",
             "scope": "custom" if args.cases else ("full" if args.full else "quick"),
             "requested_cases": list(selected), "quick_case_definition": list(QUICK), "full_case_definition": list(FULL),
             "executable": str(args.exe), "baseline_root": str(args.baseline_root),
             "allow_different_binary": args.allow_different_binary, "source_at_audit": git_provenance(),
             "threshold_policy": "Fixed strict '<' gates; no automatic tolerance relaxation",
             "scope_limitations": ["Straight laminar channels/ducts only; no curved cut-cell walls or turbulence validation",
                                   "A quick/subset pass never implies full-suite pass",
                                   "Executable hashes identify the run binary; source Git state is recorded, not a proof of its build inputs"],
             "cases": [], "errors": []}
    result_path = args.output / "suite_results.json"
    try:
        if not args.exe.is_file():
            raise ValidationError(f"SIMPLE executable missing: {args.exe}; build target simple_channel or pass --exe")
        if not args.baseline_root.is_dir():
            raise ValidationError(f"Aphros baseline root missing: {args.baseline_root}. Follow {REPO / 'validation/aphros/README.md'} and its build_baseline.ps1/run_baseline.ps1. Required comparisons cannot be skipped.")
        executable_sha = sha256(args.exe)
        suite["executable_sha256"] = executable_sha
        update_summary(suite, selected)
        write_json(result_path, suite)
        for name in selected:
            config = make_config(name, args.output / name)
            case = audit_case(name, config, args, executable_sha, suite["source_at_audit"])
            suite["cases"].append(case)
            update_summary(suite, selected)
            write_json(result_path, suite)
            print(f"{name}: {case['status']}", flush=True)
    except KeyboardInterrupt:
        suite["errors"].append("Interrupted; remaining requested cases were not completed")
    except (ValidationError, OSError, ValueError, KeyError) as error:
        suite["errors"].append(str(error))
    suite["finished_utc"] = utc_now()
    update_summary(suite, selected)
    write_json(result_path, suite)
    print(json.dumps({key: suite[key] for key in ("scope", "passed", "strict_current_binary_pass", "quick_pass", "full_pass",
                      "failed_or_missing_cases", "failed_refinement_checks", "requested_cases_not_finished", "full_suite_cases_not_run", "errors")}, indent=2))
    print(f"Full evidence: {result_path}")
    return 0 if suite["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
