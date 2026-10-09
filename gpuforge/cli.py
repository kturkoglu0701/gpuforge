from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gpuforge import __version__
from gpuforge.config import load_config
from gpuforge.daemon import run_loop
from gpuforge.metrics import collect_snapshot
from gpuforge.platform_linux import require_linux
from gpuforge.policy import PolicyEngine

log = logging.getLogger("gpuforge")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def _print_status_table(title: str, rows: list[tuple[str, str]]) -> None:
    width = max(len(title), max((len(r[0]) for r in rows), default=0))
    print(title)
    print("-" * (width + 20))
    for left, right in rows:
        print(f"  {left:<18} {right}")
    print()


def cmd_status(_: argparse.Namespace) -> int:
    snap = collect_snapshot()
    rows = [
        ("CPU %", f"{snap.cpu_percent:.1f}"),
        ("RAM %", f"{snap.memory_percent:.1f}"),
        ("RAM free (MB)", f"{snap.memory_available_mb:.0f}"),
    ]
    if snap.gpu.available:
        rows.extend(
            [
                ("GPU", snap.gpu.name or "?"),
                ("GPU util %", f"{snap.gpu.utilization_pct or 0:.1f}"),
                (
                    "GPU mem (MB)",
                    f"{snap.gpu.memory_used_mb or 0:.0f} / {snap.gpu.memory_total_mb or 0:.0f}",
                ),
            ]
        )
    else:
        rows.append(("GPU", f"unavailable ({snap.gpu.error or 'no driver'})"))
    _print_status_table("GPUForge status", rows)

    cats: dict[str, int] = {}
    for p in snap.processes:
        if p.category.value == "unknown":
            continue
        cats[p.category.value] = cats.get(p.category.value, 0) + 1
    if cats:
        _print_status_table(
            "Detected IDE / workload processes",
            [(k, str(v)) for k, v in sorted(cats.items())],
        )
    return 0


def cmd_once(args: argparse.Namespace) -> int:
    cfg = load_config(Path(args.config) if args.config else None)
    engine = PolicyEngine(cfg)
    snap = collect_snapshot(user_only=bool(cfg.get("user_only", True)))
    result = engine.evaluate(snap)
    for note in result.notes:
        print(f"rule: {note}")
    for action in result.actions:
        mode = "dry-run" if args.dry_run else "apply"
        print(f"{mode}: {action.kind} pid={action.pid} — {action.detail}")
    if not args.dry_run:
        from gpuforge.actions import apply_action

        for action in result.actions:
            apply_action(action, dry_run=False)
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    require_linux("run")
    run_loop(
        config_path=Path(args.config) if args.config else None,
        once=False,
        dry_run=args.dry_run,
    )
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="gpuforge",
        description=(
            "Self-contained Linux daemon: detect AI IDE usage, prioritize GPU, "
            "use CPU/RAM only when needed"
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="One-shot system + process summary")
    p_status.set_defaults(func=cmd_status)

    p_once = sub.add_parser("once", help="Evaluate policies once")
    p_once.add_argument("--dry-run", action="store_true", help="Print actions without applying")
    p_once.add_argument("--config", type=str, default=None, help="Override config YAML path")
    p_once.set_defaults(func=cmd_once)

    p_run = sub.add_parser("run", help="Background daemon (native Linux)")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--config", type=str, default=None)
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    _setup_logging(args.verbose)
    sys.exit(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    main()
