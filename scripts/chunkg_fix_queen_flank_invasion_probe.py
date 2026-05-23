"""Chunk-G.1.fix diagnostic: prove the legitimate F11 HEURISTIC path is wired.

Builds the F11 board state (the position from
``test_argument_selector_rejects_queen_flank_invasion`` /
``test_queen_flank_invasion_gets_king_safety_objection``), runs
``probe_moves``, and confirms that at least one probe carries
``"obj:king_safety:queen_flank_invasion"`` in its ``objections`` tuple
(the core-taxonomy field populated by the
``dialectical_chess.core_labels._FIXED_OBJECTION_BY_KIND`` translator).

Before chunk-G.1.fix the row was missing from the translator, so the
HEURISTIC core key was never emitted. This script is the post-fix evidence
that the row now translates through the documented path. Commit alongside
the two-line fix for traceability.
"""

from __future__ import annotations

from dialectical_chess.probe import owned_board_from_fen, probe_moves


# Same FEN as the F11 ablation tests
# (tests/test_dialectical_chess_evidence_ablation.py:1269, 1277).
F11_FEN = "rnbqk1nr/1ppp1ppp/4p3/p7/3P2Q1/2P5/P1P2PPP/R1B1KBNR b KQkq - 0 5"

CORE_KEY = "obj:king_safety:queen_flank_invasion"


def main() -> int:
    board = owned_board_from_fen(F11_FEN)
    probes = list(probe_moves(board, smt_fork=False))

    matches: list[tuple[str, tuple[str, ...]]] = []
    for probe in probes:
        objections = tuple(probe.objections)
        if CORE_KEY in objections:
            matches.append((probe.uci, objections))

    print(f"FEN: {F11_FEN}")
    print(f"Probes: {len(probes)}")
    print(f"Core key sought: {CORE_KEY}")
    print(f"Probes carrying that key: {len(matches)}")
    for uci, objections in matches:
        print(f"  {uci}: objections = {objections}")

    if not matches:
        print("FAIL: legitimate HEURISTIC path is NOT wired.")
        return 1
    print("PASS: legitimate HEURISTIC path is wired.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
