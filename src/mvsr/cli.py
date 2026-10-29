"""`mvsr` command line (subplans/00_FOUNDATION.md F0/F5).

    mvsr --version
    mvsr record --object <name> [--cameras wrist,head] [--scenario R-STATIC] [--gt TYPE]
               # live preview on by default; --no-preview to disable
    mvsr inspect-session <session_id|path> [--json]
    mvsr verify-session <session_id|path> [--fast] [--json] [--p0-gate]
    mvsr register-session <session_id|path> --used-by P0-gate [--gt TYPE]
    mvsr replay <session_id|path> [--camera wrist] [--speed S] [--limit N]        # P1
    mvsr sensors-report <session_id ...> [--square-mm F] [--stride N]           # P1, writes runs/
    mvsr i0-gate [<session_id ...>]                       # P1 gate (default: registry used_by I0)
    mvsr segment <session_id|path> (--box X1 Y1 X2 Y2 | --select)                  # P2 masks
    mvsr masks <session_id|path>
    mvsr observe-report <session_id ...> [--mask-revision R]       # P2 benchmark -> runs/
    mvsr i1-gate [<session_id ...>]                       # P2 gate (default: registry used_by I1)

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
    p.add_argument(
        "--cameras",
        default="wrist,head",
        help="comma list of wrist,head (default wrist,head for the single P0 take)",
    )
    p.add_argument("--wrist-fps", type=int, default=None, help="override wrist fps (default 30)")
    p.add_argument("--head-fps", type=int, default=None, help="override head fps (default 15)")
    p.add_argument(
        "--duration",
        type=float,
        default=None,
        help="optional auto-stop after S seconds; omit and stop with Ctrl+C / Q so length is operator-chosen",
    )
    p.add_argument(
        "--scenario",
        default="R-STATIC",
        help="comma list of scenario tags (ADR-0002); default R-STATIC for the P0 take",
    )
    p.add_argument("--object", dest="object_name", required=True, help="object name")
    p.add_argument("--gt", default=None, help="GT type, e.g. constant_T_base_obj")
    p.add_argument("--notes", default="", help="operator notes")
    p.add_argument("--robot-source", choices=["poll", "udp"], default=None)
    p.add_argument(
        "--preview",
        dest="preview",
        action="store_true",
        default=True,
        help="show the OpenCV preview window (default on; Q/ESC stop, SPACE pause)",
    )
    p.add_argument(
        "--no-preview",
        dest="preview",
        action="store_false",
        help="record without a window (headless / hardware pytest)",
    )
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
    p.add_argument(
        "--p0-gate",
        action="store_true",
        help="also require wrist frames, robot samples, and head frames if head was recorded (no fps/duration targets)",
    )
    p.add_argument("--workers", type=int, default=8, help="threads for hashing/decoding")

    p = sub.add_parser("register-session", help="add a verified session to fixtures/registry.yaml")
    _add_session_arg(p)
    p.add_argument("--used-by", required=True, help="comma list, e.g. P0-gate,I0")
    p.add_argument("--gt", default=None, help="GT type (default: manifest scenario.gt_type)")
    p.add_argument("--registry", type=Path, default=None, help="default fixtures/registry.yaml")
    _add_p1(sub)
    return parser


def _add_p1(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("replay", help="replay a session to SensorFrames (P1) and summarize")
    _add_session_arg(p)
    p.add_argument("--camera", default="wrist", choices=["wrist", "head"])
    p.add_argument(
        "--speed", type=float, default=None, help="pace at S x recorded time (default: fast)"
    )
    p.add_argument("--limit", type=int, default=None, help="stop after N frames")
    p.add_argument("--no-verify", action="store_true", help="skip per-image checksum verification")
    p.add_argument(
        "--config", type=Path, default=None, help="sensors YAML (default configs/sensors.yaml)"
    )
    p.add_argument("--json", action="store_true", help="machine-readable summary on stdout")

    for name, helptext in (
        ("sensors-report", "timing/sync report (+ board check for R-CALIB) written to runs/"),
        ("i0-gate", "I0 gate: sensors report + decision (default: registry sessions used_by I0)"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument(
            "sessions", nargs="*" if name == "i0-gate" else "+", help="session ids or dirs"
        )
        p.add_argument("--data-root", type=Path, default=None, help="override $MVSR_DATA_ROOT")
        p.add_argument(
            "--square-mm", type=float, default=None, help="board square (else notes square_mm=)"
        )
        p.add_argument(
            "--stride", type=int, default=None, help="check every Nth frame for the board"
        )
        p.add_argument("--no-board", action="store_true", help="skip the board check")
        p.add_argument(
            "--config", type=Path, default=None, help="sensors YAML (default configs/sensors.yaml)"
        )
        p.add_argument(
            "--handeye",
            type=Path,
            default=None,
            help="evaluate with this hand-eye file instead of the copy stored in the session",
        )
        p.add_argument("--json", action="store_true", help="print the full JSON report")
    _add_p2(sub)


def _add_p2(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p = sub.add_parser("segment", help="SAM 2.1 masks for a session -> derived/masks (P2)")
    _add_session_arg(p)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument(
        "--box", type=float, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="xyxy prompt (px)"
    )
    g.add_argument(
        "--select", action="store_true", help="draw the box on the prompt frame (needs X11: rig)"
    )
    p.add_argument(
        "--point",
        type=float,
        nargs=3,
        action="append",
        metavar=("X", "Y", "LABEL"),
        help="extra point prompt, label 1=object 0=background",
    )
    p.add_argument(
        "--frame-id", type=int, default=None, help="prompt frame (default: first emitted)"
    )
    p.add_argument("--limit", type=int, default=None, help="segment only N frames")
    p.add_argument(
        "--config",
        type=Path,
        default=None,
        help="observation YAML (default configs/observation.yaml)",
    )

    p = sub.add_parser("masks", help="list complete mask revisions of a session")
    _add_session_arg(p)

    for name, helptext in (
        ("observe-report", "observation benchmark (P2 O6) -> data/runs/"),
        ("i1-gate", "I1 gate: observation benchmark + decision (default: registry used_by I1)"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.add_argument(
            "sessions", nargs="*" if name == "i1-gate" else "+", help="session ids or dirs"
        )
        p.add_argument("--data-root", type=Path, default=None, help="override $MVSR_DATA_ROOT")
        p.add_argument("--mask-revision", default=None, help="default: the only complete revision")
        p.add_argument("--limit", type=int, default=None, help="observe only N frames")
        p.add_argument("--config", type=Path, default=None, help="observation YAML")
        p.add_argument("--json", action="store_true", help="print the full JSON report")


def _cmd_segment(args: argparse.Namespace) -> int:
    from mvsr.observation.config import load_observation_config
    from mvsr.observation.segmentation import MaskPrompt, segment_session
    from mvsr.paths import resolve_session_dir

    path = resolve_session_dir(args.session, args.data_root and args.data_root.resolve())
    box, source = (
        (tuple(args.box), "cli_box")
        if args.box
        else (_select_box(path, args.frame_id), "interactive_roi")
    )
    pts = [(p[0], p[1]) for p in args.point] if args.point else None
    labels = [int(p[2]) for p in args.point] if args.point else None
    prompt = MaskPrompt(
        frame_id=args.frame_id,
        box_xyxy=box,
        points=None if pts is None else tuple(pts),
        labels=None if labels is None else tuple(labels),
        source=source,
    )

    def progress(k: int, n: int) -> None:
        if k == n or k % 100 == 0:
            print(f"  segmented {k}/{n}", file=sys.stderr, flush=True)

    meta = segment_session(
        path, prompt, load_observation_config(args.config), limit=args.limit, progress=progress
    )
    out = {k: meta[k] for k in ("mask_revision", "frames", "by_status", "fps", "wall_s", "runtime")}
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _select_box(path: Path, frame_id: int | None) -> tuple[float, float, float, float]:
    import cv2

    from mvsr.sensors.replay import ReplaySource

    src = ReplaySource(path)
    fid = src.frame_ids[0] if frame_id is None else frame_id
    rgb = src.reader.read_frame("wrist", fid).color_rgb_u8
    title = f"prompt frame {fid}: drag a box, ENTER to accept"
    x, y, w, h = cv2.selectROI(title, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), showCrosshair=True)
    cv2.destroyAllWindows()
    if w <= 0 or h <= 0:
        raise ValueError("no box selected")
    return float(x), float(y), float(x + w), float(y + h)


def _cmd_masks(args: argparse.Namespace) -> int:
    from mvsr.observation.masks import MaskCache, list_revisions
    from mvsr.paths import resolve_session_dir

    path = resolve_session_dir(args.session, args.data_root and args.data_root.resolve())
    for rev in list_revisions(path):
        m = MaskCache(path, rev).revision_meta
        prompt = (m.get("params") or {}).get("prompt")
        print(f"{rev}  frames={m.get('frames')} by_status={m.get('by_status')} prompt={prompt}")
    return 0


def _pick_revision(path: Path, requested: str | None) -> str:
    from mvsr.observation.masks import list_revisions

    if requested:
        return str(requested)
    revs = list_revisions(path)
    if len(revs) != 1:
        raise ValueError(f"{path.name}: pass --mask-revision (complete revisions: {revs})")
    return revs[0]


def _cmd_observe(args: argparse.Namespace) -> int:
    from mvsr.observation.benchmark import i1_decision, run_benchmark, write_run
    from mvsr.observation.config import load_observation_config
    from mvsr.paths import resolve_session_dir

    root = args.data_root and args.data_root.resolve()
    cfg = load_observation_config(args.config)
    sessions = list(args.sessions)
    scen: list[list[str]] = []
    if args.command == "i1-gate" and not sessions:
        from mvsr.io.registry import load_registry

        entries = [e for e in load_registry() if "I1" in e["used_by"]]
        sessions = [e["session_id"] for e in entries]
        scen = [list(e["scenario"]) for e in entries]
        if not sessions:
            print("I1 decision: BLOCKED (no registry sessions used_by I1)", file=sys.stderr)
            return 1
    reports = []
    for s in sessions:
        path = resolve_session_dir(s, root)
        rep = run_benchmark(path, _pick_revision(path, args.mask_revision), cfg, limit=args.limit)
        reports.append(rep)
        if not scen or len(scen) < len(reports):
            scen.append(list((rep["scenario"] or {}).get("tags") or []))
    decision = i1_decision(reports, scen, cfg) if args.command == "i1-gate" else None
    run_dir = write_run(reports, decision, cfg, root)
    if args.json:
        print(
            json.dumps(
                {"sessions": reports, "i1": decision, "run_dir": str(run_dir)},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        from mvsr.observation.report import format_report

        print(format_report(reports, decision, run_dir))
    if args.command == "i1-gate":
        return 1 if decision is None or decision["decision"] == "BLOCKED" else 0
    return 0


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
    report = verify_session(
        path, check_images=not args.fast, workers=args.workers, p0_gate=args.p0_gate
    )
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


def _cmd_replay(args: argparse.Namespace) -> int:
    import hashlib
    import time
    from collections import Counter

    from mvsr.paths import resolve_session_dir
    from mvsr.sensors.config import load_sensors_config
    from mvsr.sensors.replay import ReplaySource

    path = resolve_session_dir(args.session, args.data_root and args.data_root.resolve())
    src = ReplaySource(
        path,
        args.camera,
        config=load_sensors_config(args.config),
        speed=args.speed,
        verify_images=not args.no_verify,
    )
    digest = hashlib.sha256()
    sync: Counter[str] = Counter()
    methods: Counter[str] = Counter()
    n = 0
    t0 = time.monotonic()
    first = last = None
    for f in src:
        n += 1
        sync[f.pose_sync.status.value] += 1
        methods[f.timing.method] += 1
        first = first if first is not None else f.header.timestamp_ns
        last = f.header.timestamp_ns
        digest.update(
            f"{f.header.frame_id}:{f.header.timestamp_ns}:{f.pose_sync.status.value}".encode()
        )
        if f.T_base_cam_prior is not None:
            digest.update(f.T_base_cam_prior.matrix.tobytes())
        if args.limit is not None and n >= args.limit:
            break
    wall = time.monotonic() - t0
    out = {
        "session_id": src.session_id,
        "camera": args.camera,
        "frames": n,
        "sync": dict(sorted(sync.items())),
        "timestamp_methods": dict(sorted(methods.items())),
        "recorded_span_s": None if first is None or last is None else (last - first) / 1e9,
        "wall_s": round(wall, 3),
        "fps": round(n / wall, 2) if wall > 0 else None,
        "output_sha256": digest.hexdigest(),
    }
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


def _cmd_sensors_report(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from mvsr.calib.handeye import load_handeye
    from mvsr.paths import resolve_session_dir
    from mvsr.sensors.config import load_sensors_config
    from mvsr.sensors.report import format_report, i0_decision, session_report, write_run

    root = args.data_root and args.data_root.resolve()
    cfg = load_sensors_config(args.config)
    handeye = load_handeye(args.handeye, expected_sha256=None) if args.handeye else None
    if args.stride is not None:
        cfg = replace(cfg, board=replace(cfg.board, stride=args.stride))
    sessions = list(args.sessions)
    if args.command == "i0-gate" and not sessions:
        from mvsr.io.registry import load_registry

        sessions = [e["session_id"] for e in load_registry() if "I0" in e["used_by"]]
        if not sessions:
            print("I0 decision: BLOCKED (no registry sessions used_by I0)", file=sys.stderr)
            return 1
    reports, bundles = [], {}
    for s in sessions:
        rep, b = session_report(
            resolve_session_dir(s, root),
            cfg,
            square_mm=args.square_mm,
            board=False if args.no_board else None,
            handeye=handeye,
        )
        reports.append(rep)
        if b:
            bundles[rep["session_id"]] = b
    decision = (
        i0_decision(reports, cfg)
        if args.command == "i0-gate" or any(r["board_check"] for r in reports)
        else None
    )
    run_dir = write_run(reports, bundles, decision, cfg, root)
    if args.json:
        print(
            json.dumps(
                {"sessions": reports, "i0": decision, "run_dir": str(run_dir)},
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print(format_report(reports, decision, run_dir))
    if args.command == "i0-gate":
        return 1 if decision is None or decision["decision"] == "BLOCKED" else 0
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
        "replay": _cmd_replay,
        "sensors-report": _cmd_sensors_report,
        "i0-gate": _cmd_sensors_report,
        "segment": _cmd_segment,
        "masks": _cmd_masks,
        "observe-report": _cmd_observe,
        "i1-gate": _cmd_observe,
    }
    from mvsr.core.errors import MvsrError

    try:
        return handlers[args.command](args)
    except (FileNotFoundError, FileExistsError, ValueError, MvsrError) as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
