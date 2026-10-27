"""`mvsr` command line (subplans/00_FOUNDATION.md F0/F5).

    mvsr --version
    mvsr record --cameras wrist[,head] --scenario SMOKE --object <name> [--duration S]
    mvsr inspect-session <session_id|path> [--json]
    mvsr verify-session <session_id|path> [--fast] [--json]
    mvsr register-session <session_id|path> --used-by P0-gate [--gt TYPE]

Exit codes: 0 success, 1 check failed / session aborted, 2 usage or configuration error.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from mvsr.software import version_string


def _add_record(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("record", help="record a session from the rig (D405 [+ D435] + RM65)")
    p.add_argument(
        "--config", type=Path, default=None, help="recorder YAML (default configs/record.yaml)"
    )
    p.add_argument("--cameras", default="wrist", help="comma list of wrist,head (default wrist)")
    p.add_argument("--wrist-fps", type=int, default=None, help="override wrist fps (default 30)")
    p.add_argument("--head-fps", type=int, default=None, help="override head fps (default 15)")
    p.add_argument("--duration", type=float, default=None, help="stop after S seconds of capture")
    p.add_argument(
        "--scenario",
        required=True,
        help="comma list of scenario tags (ADR-0002), e.g. SMOKE or R-STATIC,R-CALIB",
    )
    p.add_argument("--object", dest="object_name", required=True, help="object name")
    p.add_argument("--gt", default=None, help="GT type, e.g. constant_T_base_obj")
    p.add_argument("--notes", default="", help="operator notes")
    p.add_argument("--robot-source", choices=["poll", "udp"], default=None)
    p.add_argument("--preview", action="store_true", help="show the OpenCV preview window")
    p.add_argument("--start-paused", action="store_true", help="begin paused (SPACE resumes)")
    p.add_argument("--data-root", type=Path, default=None, help="override $MVSR_DATA_ROOT")


def _add_session_arg(p: argparse.ArgumentParser) -> None:
    p.add_argument("session", help="session_id under $MVSR_DATA_ROOT/sessions or a directory")
    p.add_argument("--data-root", type=Path, default=None, help="override $MVSR_DATA_ROOT")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mvsr", description="MVStreamRefine V2")
    parser.add_argument("--version", action="version", version=version_string())
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)
    _add_record(sub)

    p = sub.add_parser("inspect-session", help="print a session summary report")
    _add_session_arg(p)
    p.add_argument("--json", action="store_true", help="machine-readable report on stdout")

    p = sub.add_parser("verify-session", help="check session integrity (checksums, ids, counts)")
    _add_session_arg(p)
    p.add_argument("--fast", action="store_true", help="skip image hashing and PNG decoding")
    p.add_argument("--json", action="store_true", help="machine-readable report on stdout")
    p.add_argument("--workers", type=int, default=8, help="threads for hashing/decoding")

    p = sub.add_parser("register-session", help="add a verified session to fixtures/registry.yaml")
    _add_session_arg(p)
    p.add_argument("--used-by", required=True, help="comma list, e.g. P0-gate,I0")
    p.add_argument("--gt", default=None, help="GT type (default: manifest scenario.gt_type)")
    p.add_argument("--registry", type=Path, default=None, help="default fixtures/registry.yaml")
    return parser


def _cmd_record(args: argparse.Namespace) -> int:
    from mvsr.record.config import load_record_config
    from mvsr.record.recorder import RecordRequest, run_recorder

    config = load_record_config(args.config)
    request = RecordRequest.from_cli(args)
    result = run_recorder(config, request)
    print(json.dumps(result.summary(), indent=2, sort_keys=True))
    return 0 if result.status == "COMPLETE" else 1


def _cmd_inspect(args: argparse.Namespace) -> int:
    from mvsr.io.inspect import format_report, inspect_session
    from mvsr.paths import resolve_session_dir

    report = inspect_session(
        resolve_session_dir(args.session, args.data_root and args.data_root.resolve())
    )
    print(json.dumps(report, indent=2, sort_keys=True) if args.json else format_report(report))
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    from mvsr.io.verify import verify_session
    from mvsr.paths import resolve_session_dir

    path = resolve_session_dir(args.session, args.data_root and args.data_root.resolve())
    report = verify_session(path, check_images=not args.fast, workers=args.workers)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(report.format())
    return 0 if report.ok else 1


def _cmd_register(args: argparse.Namespace) -> int:
    from mvsr.io.registry import register_session
    from mvsr.paths import resolve_session_dir

    path = resolve_session_dir(args.session, args.data_root and args.data_root.resolve())
    entry = register_session(
        path, used_by=[u for u in args.used_by.split(",") if u], gt=args.gt, registry=args.registry
    )
    print(json.dumps(entry, indent=2, sort_keys=True))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    handlers = {
        "record": _cmd_record,
        "inspect-session": _cmd_inspect,
        "verify-session": _cmd_verify,
        "register-session": _cmd_register,
    }
    from mvsr.core.errors import MvsrError

    try:
        return handlers[args.command](args)
    except (FileNotFoundError, FileExistsError, ValueError, MvsrError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
