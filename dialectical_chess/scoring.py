"""Shared benchmark scoring helpers."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import chess

from dialectical_chess.engine import DialecticalChessEngine, EngineSettings
from dialectical_chess.evidence import is_report_positional_reason
from dialectical_chess.search import ReplyAnalysisSettings


def score_board(
    board: chess.Board,
    expected_uci: set[str],
    args: argparse.Namespace,
    *,
    avoid_uci: set[str] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    decision = DialecticalChessEngine(
        EngineSettings(
            dialectic_depth=args.dialectic_depth,
            search_depth=args.search_depth,
            search_backend=args.search_backend,
            smt_mate=args.smt_mate,
            smt_fork=getattr(args, "smt_fork", True),
            positional_reasons=getattr(args, "positional_reasons", True),
            reply_analysis=reply_analysis_settings(args),
        )
    ).choose_move(board)
    selected = decision.selected
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    selected_uci = None if selected is None else decision.move_uci
    avoid_uci = avoid_uci or set()
    selected_expected = selected_uci in expected_uci
    selected_avoided = None if not avoid_uci else selected_uci not in avoid_uci
    if expected_uci and avoid_uci:
        correct = selected_expected and bool(selected_avoided)
    elif expected_uci:
        correct = selected_expected
    elif avoid_uci:
        correct = bool(selected_avoided)
    else:
        correct = False
    return {
        "expected_uci": sorted(expected_uci),
        "avoid_uci": sorted(avoid_uci),
        "selected_uci": selected_uci,
        "selected_san": None if selected is None else selected.san,
        "correct": correct,
        "avoided": selected_avoided,
        "score": None if selected is None else selected.score,
        "reasons": [] if selected is None else list(selected.reasons),
        "objections": [] if selected is None else list(selected.objections),
        "reply_attacks": [] if selected is None else list(selected.reply_attacks),
        "search_score": None if selected is None else selected.search_score,
        "search_line": [] if selected is None else list(selected.search_line),
        "smt_witnesses": [] if selected is None else list(selected.smt_witnesses),
        "elapsed_ms": elapsed_ms,
    }


def score_full_line(
    board: chess.Board, moves: list[str], args: argparse.Namespace
) -> bool:
    working = board.copy(stack=False)
    for index, move_text in enumerate(moves):
        expected = chess.Move.from_uci(move_text)
        if index % 2 == 0:
            result = score_board(working, {move_text}, args)
            if not result["correct"]:
                return False
        if expected not in working.legal_moves:
            return False
        working.push(expected)
    return True


def report_progress(
    mode: str, completed: int, total: int, args: argparse.Namespace
) -> None:
    every = getattr(args, "progress_every", 5)
    if every <= 0:
        return
    if completed == total or completed % every == 0:
        print(f"progress {mode} {completed}/{total}", file=sys.stderr, flush=True)


def reply_analysis_settings(args: argparse.Namespace) -> ReplyAnalysisSettings:
    max_replies = getattr(args, "reply_max_replies", 128)
    max_defense_nodes = getattr(args, "reply_max_defense_nodes", 5000)
    return ReplyAnalysisSettings(
        max_replies=None if max_replies < 0 else max_replies,
        max_defense_nodes=None if max_defense_nodes < 0 else max_defense_nodes,
        min_defense_material=getattr(args, "reply_min_defense_material", 300),
    )


def settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "dialectic_depth": args.dialectic_depth,
        "dialectic_depth_from_mate_theme": getattr(
            args, "dialectic_depth_from_mate_theme", False
        ),
        "search_depth": args.search_depth,
        "search_backend": args.search_backend,
        "smt_mate": args.smt_mate,
        "smt_fork": getattr(args, "smt_fork", True),
        "positional_reasons": getattr(args, "positional_reasons", True),
        "reply_analysis": {
            "max_replies": reply_analysis_settings(args).max_replies,
            "max_defense_nodes": reply_analysis_settings(args).max_defense_nodes,
            "min_defense_material": reply_analysis_settings(args).min_defense_material,
        },
        "movegen": "owned",
    }


def tactical_witness_variants(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    return {
        "fork_on": {
            "search_depth": args.search_depth,
            "search_backend": args.search_backend,
            "smt_fork": True,
        },
        "fork_off": {
            "search_depth": args.search_depth,
            "search_backend": args.search_backend,
            "smt_fork": False,
        },
        "search1": {
            "search_depth": 1,
            "search_backend": "alphabeta",
            "smt_fork": True,
        },
        "search1_no_fork": {
            "search_depth": 1,
            "search_backend": "alphabeta",
            "smt_fork": False,
        },
    }


def tactical_witness_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_uci": result.get("selected_uci"),
        "selected_san": result.get("selected_san"),
        "correct": result.get("correct"),
        "score": result.get("score"),
        "reasons": result.get("reasons", []),
        "objections": result.get("objections", []),
        "reply_attacks": result.get("reply_attacks", []),
        "search_score": result.get("search_score"),
        "search_line": result.get("search_line", []),
        "smt_witnesses": result.get("smt_witnesses", []),
        "fork_reasons": [
            reason
            for reason in result.get("reasons", [])
            if reason.startswith("smt:fork")
        ],
        "search_reasons": [
            reason
            for reason in result.get("reasons", []) + result.get("objections", [])
            if reason.startswith(
                ("search:", "search_support:", "search_refutes:", "search_line:")
            )
        ],
        "elapsed_ms": result.get("elapsed_ms"),
    }


def positional_delta_entry(
    row: dict[str, str],
    positional_on: dict[str, Any],
    positional_off: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    on_move = positional_on.get("selected_uci")
    off_move = positional_off.get("selected_uci")
    return {
        "id": row.get("PuzzleId", ""),
        "fen": row["FEN"],
        "moves": row.get("Moves", "").split(),
        "expected_uci": positional_on["expected_uci"],
        "rating": int(row.get("Rating") or 0),
        "themes": row.get("Themes", "").split(),
        "dialectic_depth": args.dialectic_depth,
        "changed_decision": on_move != off_move,
        "positional_on": positional_snapshot(positional_on),
        "positional_off": positional_snapshot(positional_off),
        "classification": classify_positional_delta(positional_on, positional_off),
    }


def positional_snapshot(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_uci": result.get("selected_uci"),
        "selected_san": result.get("selected_san"),
        "correct": result.get("correct"),
        "score": result.get("score"),
        "reasons": result.get("reasons", []),
        "objections": result.get("objections", []),
        "reply_attacks": result.get("reply_attacks", []),
        "search_score": result.get("search_score"),
        "search_line": result.get("search_line", []),
        "smt_witnesses": result.get("smt_witnesses", []),
        "elapsed_ms": result.get("elapsed_ms"),
    }


def classify_positional_delta(
    positional_on: dict[str, Any], positional_off: dict[str, Any]
) -> dict[str, Any]:
    on_reasons = positional_on.get("reasons", [])
    off_reasons = positional_off.get("reasons", [])
    on_reply_attacks = positional_on.get("reply_attacks", [])
    off_reply_attacks = positional_off.get("reply_attacks", [])
    return {
        "harmful_when": (
            "positional_off_only_success"
            if positional_off.get("correct") and not positional_on.get("correct")
            else None
        ),
        "helpful_when": (
            "positional_on_only_success"
            if positional_on.get("correct") and not positional_off.get("correct")
            else None
        ),
        "positional_reason_families": sorted(
            {
                reason.split(":", 1)[0]
                for reason in on_reasons
                if is_report_positional_reason(reason)
            }
        ),
        "winning_move_tactical_markers": tactical_markers(off_reasons),
        "positional_move_tactical_markers": tactical_markers(on_reasons),
        "positional_move_unresolved_reply_attacks": [
            attack for attack in on_reply_attacks if ":defended:" not in attack
        ],
        "nonpositional_move_unresolved_reply_attacks": [
            attack for attack in off_reply_attacks if ":defended:" not in attack
        ],
        "positional_move_no_immediate_tactical_warrant": (
            "objection:no_immediate_tactical_warrant"
            in positional_on.get("objections", [])
        ),
    }


def tactical_markers(reasons: list[str]) -> list[str]:
    markers = []
    for reason in reasons:
        if reason in {"terminal:checkmate", "tactical:check", "procedural:mate_in_one"}:
            markers.append(reason)
        elif reason.startswith(("material:", "smt:", "search:")):
            markers.append(reason)
    return markers
