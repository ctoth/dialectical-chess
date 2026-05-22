"""Experiment matrix benchmark run mode."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

from dialectical_chess.bench_lichess import (
    score_lichess_rows,
    selected_lichess_rows,
    summarize_lichess_rows,
)


def run_experiment_matrix(args: argparse.Namespace) -> dict[str, Any]:
    if args.lichess_puzzles is None:
        raise ValueError("--experiment-matrix requires --lichess-puzzles")
    rows = selected_lichess_rows(args)
    cases = experiment_matrix_cases(args.matrix_preset)
    runs = []
    for index, case in enumerate(cases, start=1):
        case_args = argparse.Namespace(**vars(args))
        case_args.experiment_matrix = False
        for key, value in case["overrides"].items():
            setattr(case_args, key, value)
        print(
            f"progress experiment_matrix {index}/{len(cases)} {case['name']}",
            file=sys.stderr,
            flush=True,
        )
        started = time.perf_counter()
        payload = score_lichess_rows(rows, case_args)
        runs.append(
            {
                "name": case["name"],
                "overrides": dict(case["overrides"]),
                "settings": payload["settings"],
                "total": payload["total"],
                "solved": payload["solved"],
                "hit_rate": payload["hit_rate"],
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "by_rating_bucket": payload["by_rating_bucket"],
                "by_theme": payload["by_theme"],
                "positions": payload["positions"],
            }
        )
    return {
        "ok": True,
        "mode": "lichess_experiment_matrix",
        "suite": str(args.lichess_puzzles),
        "matrix_preset": args.matrix_preset,
        "sample": summarize_lichess_rows(rows),
        "runs": runs,
    }


def experiment_matrix_cases(preset: str) -> list[dict[str, Any]]:
    if preset == "smoke":
        return [
            {"name": "argument_d0", "overrides": {"dialectic_depth": 0}},
            {"name": "argument_d1", "overrides": {"dialectic_depth": 1}},
            {
                "name": "argument_mate_theme_depth",
                "overrides": {"dialectic_depth_from_mate_theme": True},
            },
        ]
    return [
        {"name": "argument_d0", "overrides": {"dialectic_depth": 0}},
        {"name": "argument_d1", "overrides": {"dialectic_depth": 1}},
        {"name": "argument_d2", "overrides": {"dialectic_depth": 2}},
        {
            "name": "argument_d2_no_positional",
            "overrides": {"dialectic_depth": 2, "positional_reasons": False},
        },
        {
            "name": "argument_d2_no_smt",
            "overrides": {"dialectic_depth": 2, "smt_mate": False},
        },
        {
            "name": "argument_d2_no_fork",
            "overrides": {"dialectic_depth": 2, "smt_fork": False},
        },
        {
            "name": "argument_d2_search1",
            "overrides": {
                "dialectic_depth": 2,
                "search_depth": 1,
                "search_backend": "alphabeta",
            },
        },
        {
            "name": "argument_d2_search1_no_fork",
            "overrides": {
                "dialectic_depth": 2,
                "search_depth": 1,
                "search_backend": "alphabeta",
                "smt_fork": False,
            },
        },
        {
            "name": "argument_mate_theme_depth",
            "overrides": {"dialectic_depth_from_mate_theme": True},
        },
    ]
