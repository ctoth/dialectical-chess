"""F11 upstream emitter fix — diagnostic.

Verifies that the ``QUEEN_FLANK_INVASION`` emitter at
``heuristics/king_safety.py:209-228`` now provides
``moved_piece_en_pris_value`` so the chess-to-core translator at
``core_labels.core_objection_label`` produces a real
``obj:loses_exchange:{n}`` label instead of falling through to the
HEURISTIC dispatcher.

Three checks:

1. Probe inspection — for the F11 position
   ``rnbqk1nr/1ppp1ppp/4p3/p7/3P2Q1/2P5/P1P2PPP/R1B1KBNR b KQkq - 0 5``,
   inspect the chess objection_evidence and the core objections tuple on
   the ``g8f6`` and ``b8c6`` probes. Confirm
   ``king_safety:queen_flank_invasion:g8f6:g7`` carries an en-pris value of
   100 (the pawn at g7), and the translated core label is
   ``obj:loses_exchange:100`` on the inherited ``objections`` tuple.

2. FACT-key differentiation — compute
   :func:`dialectical_games.decider.fact_only_key` for every probe and
   print the per-probe key. The move(s) tagged with
   ``obj:loses_exchange:100`` should have a worse (larger) term-1 magnitude
   than the moves with no FACT objection.

3. End-to-end — call :func:`dialectical_chess.arguments.choose_move` and
   confirm F11 still picks ``g8f6``. (The pre-fix engine picked g8f6 via
   term-3 graded strength; post-fix the same move should win, ideally via
   the FACT layer now that the FACT key differentiates.)
"""

from __future__ import annotations

from dialectical_chess.arguments import MoveProbe, choose_move
from dialectical_chess.core_labels import core_objection_label
from dialectical_chess.evidence import ObjectionEvidence, Tier
from dialectical_chess.graded_policy import ChessGradedPolicy
from dialectical_chess.probe import owned_board_from_fen, probe_moves
from dialectical_games.arguments import build_root_argument_graph
from dialectical_games.decider import fact_only_key

F11_FEN = "rnbqk1nr/1ppp1ppp/4p3/p7/3P2Q1/2P5/P1P2PPP/R1B1KBNR b KQkq - 0 5"


def main() -> None:
    print(f"=== F11 emitter-fix diagnostic ===")
    print(f"FEN: {F11_FEN}")
    print()

    board = owned_board_from_fen(F11_FEN)
    probes_list = list(probe_moves(board, smt_fork=False))
    probes_by_uci = {probe.uci: probe for probe in probes_list}
    print(f"probe_moves emitted {len(probes_list)} candidate moves")
    print()

    # --- (1) emitter / translator inspection ---------------------------
    print("--- (1) chess emitter -> core translator on g8f6 ---")
    g8f6 = probes_by_uci["g8f6"]
    for ev in g8f6.objection_evidence:
        if "queen_flank_invasion" not in ev.label:
            continue
        if not isinstance(ev, ObjectionEvidence):
            continue
        translated = core_objection_label(ev)
        print(
            f"  chess.label                       = {ev.label}"
        )
        print(
            f"  ev.moved_piece_en_pris_value      = {ev.moved_piece_en_pris_value}"
        )
        print(
            f"  ev.tier                           = {ev.tier.name}"
        )
        print(
            f"  core_objection_label(ev)          = {translated!r}"
        )
        assert ev.moved_piece_en_pris_value == 100, (
            f"FAIL: expected en-pris value 100 (pawn at g7), got {ev.moved_piece_en_pris_value}"
        )
        assert translated == "obj:loses_exchange:100", (
            f"FAIL: expected 'obj:loses_exchange:100', got {translated!r}"
        )
    assert "obj:loses_exchange:100" in g8f6.objections, (
        f"FAIL: 'obj:loses_exchange:100' not in g8f6.objections={g8f6.objections}"
    )
    print(f"  g8f6.objections                   = {g8f6.objections}")
    print("  OK: g8f6 carries obj:loses_exchange:100 via FACT route")
    print()

    # --- (2) FACT-key differentiation ---------------------------------
    print("--- (2) fact_only_key per probe ---")
    policy = ChessGradedPolicy(board=board)
    graph = build_root_argument_graph(probes_list, policy)
    keys: list[tuple[tuple[int, int, int, int, int], MoveProbe]] = []
    for probe in probes_list:
        key = fact_only_key(probe, graph)
        keys.append((key, probe))
    keys.sort(key=lambda kp: kp[0])
    print("  Top 6 probes by FACT-only key (smaller = better):")
    for key, probe in keys[:6]:
        print(f"    uci={probe.uci:<6} key={key}")
    print()
    fact_key_g8f6 = fact_only_key(g8f6, graph)
    fact_key_b8c6 = fact_only_key(probes_by_uci["b8c6"], graph)
    print(f"  fact_only_key(g8f6)               = {fact_key_g8f6}")
    print(f"  fact_only_key(b8c6)               = {fact_key_b8c6}")
    print(
        f"  fact_layer differentiates g8f6 vs b8c6?  "
        f"{fact_key_g8f6 != fact_key_b8c6}"
    )
    print()

    # --- (3) end-to-end: F11 still picks g8f6 -------------------------
    print("--- (3) lexicographic_decide ---")
    chosen = choose_move(probes_list)
    assert chosen is not None
    print(f"  chosen.uci                        = {chosen.uci}")
    assert chosen.uci == "g8f6", f"FAIL: expected g8f6, got {chosen.uci}"

    # Determine FACT vs HEURISTIC route. If g8f6's fact_only_key is the
    # unique minimum among survivors, FACT alone selected it (FACT route).
    # Otherwise the FACT layer left a tie that the graded layer broke
    # (HEURISTIC route).
    min_key = min(key for key, _ in keys)
    fact_layer_winners = [probe.uci for key, probe in keys if key == min_key]
    print(f"  min fact_only_key                 = {min_key}")
    print(f"  fact_layer_winners                = {fact_layer_winners}")
    if fact_layer_winners == ["g8f6"]:
        route = "FACT (unique min FACT key)"
    elif "g8f6" in fact_layer_winners:
        route = (
            f"HEURISTIC tiebreak — FACT tied {len(fact_layer_winners)} moves, "
            "graded layer chose g8f6"
        )
    else:
        route = "UNEXPECTED — g8f6 not in FACT survivor set"
    print(f"  F11 route                         = {route}")
    print()
    print("=== diagnostic complete ===")


if __name__ == "__main__":
    main()
