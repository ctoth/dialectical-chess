"""EPD and core benchmark run modes for the dialectical chess sidecar."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dialectical_chess.bench_lichess import (
    run_lichess,
    run_positional_comparison,
    run_tactical_witness_comparison,
)
from dialectical_chess.bench_matrix import run_experiment_matrix
from dialectical_chess.board import PERFT_FIXTURES, OwnedBoard, owned_perft
from dialectical_chess.epd import parse_epd_case, read_epd_lines
from dialectical_chess.loss_mining import mine_loss_turning_points, reviewed_epd_lines
from dialectical_chess.matches import run_internal_uci_match, run_uci_match
from dialectical_chess.scoring import report_progress, score_board, settings


PACKAGE_ROOT = Path(__file__).resolve().parent
FIXTURES_DIR = PACKAGE_ROOT / "fixtures"
OPENINGS_PATH = FIXTURES_DIR / "dialectical_chess_openings.epd"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epd", type=Path)
    parser.add_argument("--lichess-puzzles", type=Path)
    parser.add_argument("--experiment-matrix", action="store_true")
    parser.add_argument("--compare-positional", action="store_true")
    parser.add_argument("--compare-tactical-witness", action="store_true")
    parser.add_argument("--matrix-preset", choices=("core", "smoke"), default="core")
    parser.add_argument("--perft", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--mine-loss-pgn", type=Path)
    parser.add_argument("--loss-epd-out", type=Path)
    parser.add_argument("--loss-engine-name", default="Dialectical")
    parser.add_argument("--loss-mate-depth", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--progress-every", type=int, default=5)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--full-line", action="store_true")
    parser.add_argument("--rating-min", type=int)
    parser.add_argument("--rating-max", type=int)
    parser.add_argument("--theme-include", action="append", default=[])
    parser.add_argument("--theme-exclude", action="append", default=[])
    parser.add_argument("--side-to-move", choices=("w", "b"))
    parser.add_argument("--dialectic-depth", type=int, default=1)
    parser.add_argument("--dialectic-depth-from-mate-theme", action="store_true")
    parser.add_argument("--search-depth", type=int, default=0)
    parser.add_argument(
        "--search-backend", choices=("negamax", "alphabeta"), default="negamax"
    )
    parser.add_argument(
        "--no-positional-reasons", action="store_false", dest="positional_reasons"
    )
    parser.add_argument("--reply-max-replies", type=int, default=128)
    parser.add_argument("--reply-max-defense-nodes", type=int, default=5000)
    parser.add_argument("--reply-min-defense-material", type=int, default=300)
    parser.add_argument("--no-smt-mate", action="store_false", dest="smt_mate")
    parser.add_argument("--no-smt-fork", action="store_false", dest="smt_fork")
    parser.add_argument("--uci-match-command", action="store_true")
    parser.add_argument("--run-uci-match", action="store_true")
    parser.add_argument("--internal-uci-match", action="store_true")
    parser.add_argument(
        "--match-baseline", choices=("nosmt", "random", "stockfish"), default="nosmt"
    )
    parser.add_argument("--match-openings", type=Path, default=OPENINGS_PATH)
    parser.add_argument("--match-games", type=int, default=2)
    parser.add_argument("--match-max-plies", type=int, default=40)
    parser.add_argument("--match-pgn-out", type=Path)
    parser.add_argument("--match-tc", default="1+0.01")
    parser.add_argument("--stockfish-path")
    parser.add_argument("--stockfish-elo", type=int, default=1320)
    parser.set_defaults(smt_mate=True, smt_fork=True, positional_reasons=True)
    args = parser.parse_args()

    started = time.perf_counter()
    if args.perft:
        payload = run_perft()
    elif args.mine_loss_pgn:
        payload = run_loss_mining(args)
    elif args.experiment_matrix:
        payload = run_experiment_matrix(args)
    elif args.compare_positional:
        payload = run_positional_comparison(args)
    elif args.compare_tactical_witness:
        payload = run_tactical_witness_comparison(args)
    elif args.lichess_puzzles:
        payload = run_lichess(args)
    elif args.ablation:
        payload = run_ablation(args)
    elif args.internal_uci_match:
        payload = run_internal_uci_match(args)
    elif args.uci_match_command or args.run_uci_match:
        payload = run_uci_match(args)
    else:
        payload = run_epd(args)
    payload["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
    payload["script_paths"] = {
        "probe": "dialectical-chess-probe",
        "owned": "dialectical-chess-owned",
    }
    text = json.dumps(payload, indent=2)
    print(text)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    return 0 if payload.get("ok", True) else 1


def run_loss_mining(args: argparse.Namespace) -> dict[str, Any]:
    pgn_text = args.mine_loss_pgn.read_text(encoding="utf-8")
    points = mine_loss_turning_points(
        pgn_text,
        engine_name=args.loss_engine_name,
        mate_depth=args.loss_mate_depth,
    )
    epd_lines = reviewed_epd_lines(points)
    if args.loss_epd_out:
        args.loss_epd_out.parent.mkdir(parents=True, exist_ok=True)
        args.loss_epd_out.write_text(
            "\n".join(epd_lines) + ("\n" if epd_lines else ""), encoding="utf-8"
        )
    return {
        "ok": True,
        "mode": "loss_mining",
        "pgn": str(args.mine_loss_pgn),
        "engine_name": args.loss_engine_name,
        "mate_depth": args.loss_mate_depth,
        "turning_points": [point.__dict__ for point in points],
        "epd_lines": epd_lines,
        "loss_epd_out": None if args.loss_epd_out is None else str(args.loss_epd_out),
    }


def run_epd(args: argparse.Namespace) -> dict[str, Any]:
    lines = read_epd_lines(args.epd)
    if args.limit is not None:
        lines = lines[: args.limit]
    results = []
    for index, line in enumerate(lines, start=1):
        try:
            case = parse_epd_case(line, index=index)
            result = score_board(
                case["board"],
                case["expected_uci"],
                args,
                avoid_uci=case["avoid_uci"],
            )
            result["id"] = case["id"]
            result["line"] = index
            result["fen"] = case["board"].fen()
            results.append(result)
        except Exception as exc:
            if args.fail_fast:
                raise
            results.append(
                {"line": index, "error": str(exc), "errored": True, "correct": False}
            )
        report_progress("epd", index, len(lines), args)
    errored = sum(1 for result in results if result.get("errored") or "error" in result)
    evaluated = len(results) - errored
    solved = sum(
        1 for result in results if not result.get("errored") and result.get("correct")
    )
    failed = sum(
        1
        for result in results
        if not result.get("errored") and not result.get("correct")
    )
    avoid_results = [result for result in results if result.get("avoided") is not None]
    avoided = sum(1 for result in avoid_results if result.get("avoided"))
    return {
        "ok": all("error" not in result for result in results),
        "mode": "epd",
        "suite": str(args.epd) if args.epd else "built-in-smoke",
        "total": len(results),
        "evaluated": evaluated,
        "errored": errored,
        "failed": failed,
        "solved": solved,
        "hit_rate": solved / evaluated if evaluated else 0.0,
        "avoided": avoided,
        "avoid_total": len(avoid_results),
        "avoid_rate": avoided / len(avoid_results) if avoid_results else None,
        "settings": settings(args),
        "positions": results,
    }


def run_ablation(args: argparse.Namespace) -> dict[str, Any]:
    base_epd = args.epd
    runs = []
    baseline_moves: list[str | None] | None = None
    for smt_mate in (True, False):
        for dialectic_depth in (0, 1, 2):
            for search_depth in (0, 1, 2, 3):
                for backend in ("negamax", "alphabeta"):
                    case_args = argparse.Namespace(**vars(args))
                    case_args.epd = base_epd
                    case_args.smt_mate = smt_mate
                    case_args.dialectic_depth = dialectic_depth
                    case_args.search_depth = search_depth
                    case_args.search_backend = backend
                    started = time.perf_counter()
                    payload = run_epd(case_args)
                    selected_moves = [
                        position.get("selected_uci")
                        for position in payload["positions"]
                    ]
                    if baseline_moves is None:
                        baseline_moves = selected_moves
                    runs.append(
                        {
                            "settings": settings(case_args),
                            "total": payload["total"],
                            "solved": payload["solved"],
                            "hit_rate": payload["hit_rate"],
                            "avoid_rate": payload["avoid_rate"],
                            "selected_move_deltas_vs_first": sum(
                                left != right
                                for left, right in zip(
                                    selected_moves, baseline_moves, strict=False
                                )
                            ),
                            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                        }
                    )
    return {
        "ok": True,
        "mode": "ablation",
        "suite": str(base_epd) if base_epd else "built-in-smoke",
        "runs": runs,
    }


def run_perft() -> dict[str, Any]:
    results = []
    ok = True
    for name, (fen, depths) in PERFT_FIXTURES.items():
        board = OwnedBoard.from_fen(fen)
        for depth, expected in depths.items():
            started = time.perf_counter()
            actual = owned_perft(board, depth)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            if actual != expected:
                ok = False
            results.append(
                {
                    "name": name,
                    "fen": fen,
                    "depth": depth,
                    "expected": expected,
                    "actual": actual,
                    "correct": actual == expected,
                    "elapsed_ms": elapsed_ms,
                    "nodes_per_second": actual / (elapsed_ms / 1000.0)
                    if elapsed_ms
                    else None,
                }
            )
    return {
        "ok": ok,
        "mode": "perft",
        "total": len(results),
        "passed": sum(1 for item in results if item["correct"]),
        "positions": results,
    }
