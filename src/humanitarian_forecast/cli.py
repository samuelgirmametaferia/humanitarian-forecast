from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from humanitarian_forecast.core.model_store import discover_models, read_info
from humanitarian_forecast.core.paths import PATHS
from humanitarian_forecast.core.registry import all_systems, get_system
from humanitarian_forecast.core.runner import run_module
from humanitarian_forecast.workflows.production import ProductionConfig, print_plan, run_full_workflow


def _default_ethiopia_db() -> Path:
    return Path(os.environ.get("HUMANITARIAN_ETHIOPIA_DB", "/Users/sam/Documents/wfp/database.sqlite"))


def _workflow_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--ethiopia-db", type=Path, default=_default_ethiopia_db())
    parser.add_argument("--ucdp", type=Path, default=PATHS.raw / "ged261-csv.zip")
    parser.add_argument("--reliefweb", type=Path, default=PATHS.data / "reliefweb_full.jsonl.gz")
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true", help="Allow overwriting an explicitly selected version directory.")
    parser.add_argument("--from-step", help="Start at a workflow step name or numbered key such as 4:risk.train.production.")
    parser.add_argument("--through-step", help="Stop after a workflow step name or numbered key.")
    parser.add_argument("--base-version", default="v3")
    parser.add_argument("--ethiopia-version", default="v5")
    parser.add_argument("--location-version", default="v10")
    parser.add_argument("--location-ethiopia-version", default="v1")
    parser.add_argument("--base-epochs", type=int, default=20)
    parser.add_argument("--ethiopia-epochs", type=int, default=20)
    parser.add_argument("--candidate-epochs", type=int, default=12)
    parser.add_argument("--candidate-ethiopia-epochs", type=int, default=4)


def _config(args: argparse.Namespace, *, dry_run: bool) -> ProductionConfig:
    kwargs = dict(
        ethiopia_db=args.ethiopia_db,
        ucdp=args.ucdp,
        reliefweb=args.reliefweb,
        start_year=args.start_year,
        skip_download=args.skip_download,
        force=args.force,
        from_step=args.from_step,
        through_step=args.through_step,
        base_version=args.base_version,
        ethiopia_version=args.ethiopia_version,
        location_version=args.location_version,
        location_ethiopia_version=args.location_ethiopia_version,
        base_epochs=args.base_epochs,
        ethiopia_epochs=args.ethiopia_epochs,
        candidate_epochs=args.candidate_epochs,
        candidate_ethiopia_epochs=args.candidate_ethiopia_epochs,
        dry_run=dry_run,
    )
    if args.end_year is not None:
        kwargs["end_year"] = args.end_year
    return ProductionConfig(**kwargs)


def _headline(info: dict) -> str:
    metrics = info.get("metrics", {})
    headline = metrics.get("headline", {}) if isinstance(metrics, dict) else {}
    if isinstance(headline, dict):
        median = headline.get("median_error_km", headline.get("top1_median_error_km"))
        mean = headline.get("mean_error_km", headline.get("top1_mean_error_km"))
        if median is not None:
            return f"median={float(median):.2f} km" + (f", mean={float(mean):.2f} km" if mean is not None else "")
        ap = headline.get("average_precision")
        if ap is not None:
            return f"AP={float(ap):.4f}"
    return "no standalone holdout headline"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Unified controller for humanitarian forecasting data, training, evaluation, and inference.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    systems = sub.add_parser("systems", help="List registered subsystems available to the controller.")
    systems.set_defaults(handler=_cmd_systems)

    run = sub.add_parser("run", help="Run any registered subsystem; add future systems in core/registry.py.")
    run.add_argument("system")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("args", nargs="*", help="Arguments passed to the subsystem after a `--` separator.")
    run.set_defaults(handler=_cmd_run)

    workflow = sub.add_parser("workflow", help="Run or inspect composed production workflows.")
    workflow_sub = workflow.add_subparsers(dest="workflow_command", required=True)
    full = workflow_sub.add_parser("full", help="Global all-data training followed by Ethiopia fine-tuning.")
    _workflow_options(full)
    full.add_argument("--dry-run", action="store_true", help="Print every command without executing it.")
    full.set_defaults(handler=_cmd_workflow_full)
    plan = workflow_sub.add_parser("plan", help="Show the full workflow without running commands.")
    _workflow_options(plan)
    plan.set_defaults(handler=_cmd_workflow_plan)

    models = sub.add_parser("models", help="Inspect the versioned model store and info.blt metadata.")
    models_sub = models.add_subparsers(dest="models_command", required=True)
    model_list = models_sub.add_parser("list", help="List all versioned models with metadata.")
    model_list.set_defaults(handler=_cmd_models_list)
    model_info = models_sub.add_parser("info", help="Print one model's info.blt.")
    model_info.add_argument("subsystem", help="For example: location/candidate_ranker or risk/ethiopia")
    model_info.add_argument("version", help="For example: v9")
    model_info.set_defaults(handler=_cmd_models_info)

    return parser


def _cmd_systems(args: argparse.Namespace) -> int:
    del args
    print(f"{'SYSTEM':40} {'CATEGORY':12} {'PROD':4} DESCRIPTION")
    for spec in all_systems():
        print(f"{spec.name:40} {spec.category:12} {'yes' if spec.production else 'no ':4} {spec.description}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    spec = get_system(args.system)
    forwarded = list(args.args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    run_module(spec.module, forwarded, dry_run=args.dry_run)
    return 0


def _cmd_workflow_full(args: argparse.Namespace) -> int:
    config = _config(args, dry_run=args.dry_run)
    run_full_workflow(config)
    return 0


def _cmd_workflow_plan(args: argparse.Namespace) -> int:
    print_plan(_config(args, dry_run=True))
    return 0


def _cmd_models_list(args: argparse.Namespace) -> int:
    del args
    models = discover_models()
    if not models:
        print("No model metadata found.")
        return 0
    print(f"{'MODEL':48} {'STATUS':28} PERFORMANCE")
    for directory, info in models:
        relative = directory.relative_to(PATHS.models)
        print(f"{str(relative):48} {str(info.get('status', 'unknown')):28} {_headline(info)}")
    return 0


def _cmd_models_info(args: argparse.Namespace) -> int:
    directory = PATHS.models / Path(args.subsystem) / args.version
    path = directory / "info.blt"
    if not path.exists():
        raise SystemExit(f"model metadata not found: {path}")
    print(json.dumps(read_info(directory), indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args) or 0)
    except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
