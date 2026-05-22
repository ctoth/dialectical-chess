"""Lichess CSV benchmark run modes and filters."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from typing import Any

import chess

from dialectical_chess.scoring import (
    classify_positional_delta,
    positional_delta_entry,
    positional_snapshot,
    report_progress,
    score_board,
    score_full_line,
    settings,
    tactical_markers,
    tactical_witness_snapshot,
    tactical_witness_variants,
)


MATE_THEME_RE = __import__("re").compile(r"^mateIn([1-9][0-9]*)$")


def run_lichess(args: argparse.Namespace) -> dict[str, Any]:
    return score_lichess_rows(selected_lichess_rows(args), args)


def run_positional_comparison(args: argparse.Namespace) -> dict[str, Any]:
    if args.lichess_puzzles is None:
        raise ValueError("--compare-positional requires --lichess-puzzles")
    rows = selected_lichess_rows(args)
    results = []
    changed_decisions = []
    solved_only_positional_on = []
    solved_only_positional_off = []
    both_fail_changed = []
    both_solve_changed = []

    for index, row in enumerate(rows, start=1):
        moves = row["Moves"].split()
        expected = {moves[0]} if moves else set()
        row_args = argparse.Namespace(**vars(args))
        row_args.dialectic_depth = dialectic_depth_for_lichess_row(row, args)

        on_args = argparse.Namespace(**vars(row_args))
        on_args.positional_reasons = True
        off_args = argparse.Namespace(**vars(row_args))
        off_args.positional_reasons = False

        board = chess.Board(row["FEN"])
        positional_on = score_board(board, expected, on_args)
        positional_off = score_board(board, expected, off_args)
        entry = positional_delta_entry(row, positional_on, positional_off, row_args)
        results.append(entry)

        if entry["changed_decision"]:
            changed_decisions.append(entry)
            if entry["positional_on"]["correct"] and entry["positional_off"]["correct"]:
                both_solve_changed.append(entry)
            elif (
                not entry["positional_on"]["correct"]
                and not entry["positional_off"]["correct"]
            ):
                both_fail_changed.append(entry)
        if entry["positional_on"]["correct"] and not entry["positional_off"]["correct"]:
            solved_only_positional_on.append(entry)
        if entry["positional_off"]["correct"] and not entry["positional_on"]["correct"]:
            solved_only_positional_off.append(entry)

        report_progress("positional_compare", index, len(rows), args)

    return {
        "ok": True,
        "mode": "positional_comparison",
        "suite": str(args.lichess_puzzles),
        "sample": summarize_lichess_rows(rows),
        "settings": {
            **settings(args),
            "compared_positional_reasons": [True, False],
        },
        "total": len(results),
        "changed_decisions": len(changed_decisions),
        "solved_only_positional_on": len(solved_only_positional_on),
        "solved_only_positional_off": len(solved_only_positional_off),
        "both_fail_changed": len(both_fail_changed),
        "both_solve_changed": len(both_solve_changed),
        "positions": results,
    }


def run_tactical_witness_comparison(args: argparse.Namespace) -> dict[str, Any]:
    if args.lichess_puzzles is None:
        raise ValueError("--compare-tactical-witness requires --lichess-puzzles")
    rows = selected_lichess_rows(args)
    variants = tactical_witness_variants(args)
    pairs = (
        ("fork_on_vs_fork_off", "fork_on", "fork_off"),
        ("fork_on_vs_search1", "fork_on", "search1"),
        ("search1_vs_search1_no_fork", "search1", "search1_no_fork"),
        ("fork_off_vs_search1_no_fork", "fork_off", "search1_no_fork"),
    )
    variant_totals: dict[str, dict[str, float | int]] = {
        name: {"total": 0, "solved": 0} for name in variants
    }
    delta_totals = {
        name: {"changed_decisions": 0, "left_only_success": 0, "right_only_success": 0}
        for name, _, _ in pairs
    }
    positions = []

    for index, row in enumerate(rows, start=1):
        moves = row["Moves"].split()
        expected = {moves[0]} if moves else set()
        row_args = argparse.Namespace(**vars(args))
        row_args.dialectic_depth = dialectic_depth_for_lichess_row(row, args)
        board = chess.Board(row["FEN"])

        results = {}
        for name, overrides in variants.items():
            variant_args = argparse.Namespace(**vars(row_args))
            for key, value in overrides.items():
                setattr(variant_args, key, value)
            result = score_board(board, expected, variant_args)
            results[name] = tactical_witness_snapshot(result)
            variant_totals[name]["total"] += 1
            if result["correct"]:
                variant_totals[name]["solved"] += 1

        deltas = {}
        for pair_name, left_name, right_name in pairs:
            left = results[left_name]
            right = results[right_name]
            changed = left["selected_uci"] != right["selected_uci"]
            left_only = left["correct"] and not right["correct"]
            right_only = right["correct"] and not left["correct"]
            deltas[pair_name] = {
                "changed_decision": changed,
                "left_only_success": left_only,
                "right_only_success": right_only,
            }
            if changed:
                delta_totals[pair_name]["changed_decisions"] += 1
            if left_only:
                delta_totals[pair_name]["left_only_success"] += 1
            if right_only:
                delta_totals[pair_name]["right_only_success"] += 1

        positions.append(
            {
                "id": row.get("PuzzleId", ""),
                "fen": row["FEN"],
                "moves": moves,
                "expected_uci": sorted(expected),
                "rating": int(row.get("Rating") or 0),
                "themes": row.get("Themes", "").split(),
                "variants": results,
                "deltas": deltas,
            }
        )
        report_progress("tactical_witness_compare", index, len(rows), args)

    for totals in variant_totals.values():
        total = totals["total"]
        totals["hit_rate"] = totals["solved"] / total if total else 0.0
    return {
        "ok": True,
        "mode": "tactical_witness_comparison",
        "suite": str(args.lichess_puzzles),
        "sample": summarize_lichess_rows(rows),
        "settings": settings(args),
        "variant_totals": variant_totals,
        "delta_totals": delta_totals,
        "positions": positions,
    }


def selected_lichess_rows(args: argparse.Namespace) -> list[dict[str, str]]:
    rows = []
    with args.lichess_puzzles.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if include_puzzle(row, args):
                rows.append(row)
            if args.limit is not None and len(rows) >= args.limit:
                break
    return rows


def score_lichess_rows(
    rows: list[dict[str, str]], args: argparse.Namespace
) -> dict[str, Any]:
    results = []
    by_rating: Counter[str] = Counter()
    rating_totals: Counter[str] = Counter()
    by_theme: Counter[str] = Counter()
    theme_totals: Counter[str] = Counter()
    for index, row in enumerate(rows, start=1):
        board = chess.Board(row["FEN"])
        moves = row["Moves"].split()
        expected = {moves[0]} if moves else set()
        row_args = argparse.Namespace(**vars(args))
        row_args.dialectic_depth = dialectic_depth_for_lichess_row(row, args)
        result = score_board(board, expected, row_args)
        result["id"] = row.get("PuzzleId", "")
        result["rating"] = int(row.get("Rating") or 0)
        result["themes"] = row.get("Themes", "").split()
        result["dialectic_depth"] = row_args.dialectic_depth
        if args.full_line and result["correct"]:
            result["full_line_correct"] = score_full_line(board, moves, row_args)
        bucket = rating_bucket(result["rating"])
        rating_totals[bucket] += 1
        by_rating[bucket] += 1 if result["correct"] else 0
        for theme in result["themes"]:
            theme_totals[theme] += 1
            if result["correct"]:
                by_theme[theme] += 1
        results.append(result)
        report_progress("lichess_csv", index, len(rows), args)
    solved = sum(1 for result in results if result.get("correct"))
    return {
        "ok": True,
        "mode": "lichess_csv",
        "suite": str(args.lichess_puzzles),
        "total": len(results),
        "solved": solved,
        "hit_rate": solved / len(results) if results else 0.0,
        "by_rating_bucket": {
            bucket: {"solved": by_rating[bucket], "total": rating_totals[bucket]}
            for bucket in sorted(rating_totals)
        },
        "by_theme": {
            theme: {"solved": by_theme[theme], "total": theme_totals[theme]}
            for theme in sorted(theme_totals)
        },
        "settings": settings(args),
        "positions": results,
    }


def summarize_lichess_rows(rows: list[dict[str, str]]) -> dict[str, Any]:
    line_move_counts: Counter[str] = Counter()
    mate_theme_counts: Counter[str] = Counter()
    theme_counts: Counter[str] = Counter()
    for row in rows:
        moves = row.get("Moves", "").split()
        line_move_counts[str(len(moves))] += 1
        for theme in row.get("Themes", "").split():
            theme_counts[theme] += 1
            if theme.startswith("mateIn"):
                mate_theme_counts[theme] += 1
    return {
        "total": len(rows),
        "scoring_target": "first engine move only",
        "line_move_counts": dict(
            sorted(line_move_counts.items(), key=lambda item: int(item[0]))
        ),
        "mate_theme_counts": dict(sorted(mate_theme_counts.items())),
        "theme_counts": dict(sorted(theme_counts.items())),
    }


def mate_theme_depth(themes: tuple[str, ...] | list[str]) -> int | None:
    depths = []
    for theme in themes:
        match = MATE_THEME_RE.match(theme)
        if match:
            depths.append(int(match.group(1)))
    return min(depths) if depths else None


def dialectic_depth_for_lichess_row(
    row: dict[str, str], args: argparse.Namespace
) -> int:
    if not getattr(args, "dialectic_depth_from_mate_theme", False):
        return args.dialectic_depth
    return mate_theme_depth(row.get("Themes", "").split()) or args.dialectic_depth


def include_puzzle(row: dict[str, str], args: argparse.Namespace) -> bool:
    rating = int(row.get("Rating") or 0)
    if args.rating_min is not None and rating < args.rating_min:
        return False
    if args.rating_max is not None and rating > args.rating_max:
        return False
    themes = set(row.get("Themes", "").split())
    if args.theme_include and not all(theme in themes for theme in args.theme_include):
        return False
    if args.theme_exclude and any(theme in themes for theme in args.theme_exclude):
        return False
    if args.side_to_move and chess.Board(row["FEN"]).turn != (args.side_to_move == "w"):
        return False
    return True


def rating_bucket(rating: int) -> str:
    low = (rating // 200) * 200
    return f"{low}-{low + 199}"
