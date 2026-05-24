"""Trace the F11 lex-key causal chain to verify chunk H' un-xfail is principled.

Codex chunk H' analyst (MAJOR finding 2) flagged the F11 un-xfail's stated
mechanism as directionally inconsistent: the chunk-H' coder claimed that the
``obj:king_safety:queen_flank_invasion`` BOOLEAN opinion on ``g8f6`` "sums into
the resolved opinion strongly enough to flip g8f6 over the previous selection".
But ``dialectical_games.arguments._build_graded_graph_internal`` at lines
361-372 turns every ``probe.objections`` label into an ATTACK edge against that
move's argument node, so the objection should DEFEAT g8f6 (lower its graded
opinion), not promote it.

This script reconstructs the exact F11 selection so we can read off WHICH
lexicographic-decide term decides the choice. If the graded-strength term
term-3 is what selects g8f6 in spite of the objection attack, that's
legitimate. If a non-graded term (FACT priority term-2, child_eval term-5,
move_id alphabetic term-5b) decides it, the original H' explanation is wrong
and F11's recovery is coincidental.

Run with::

    uv run python scripts/chunkh_fix_f11_causal_chain.py

NOT a oneliner -- a script file (the project's hard rule).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dialectical_games.arguments import build_root_argument_graph  # noqa: E402
from dialectical_games.decider import (  # noqa: E402
    _accepted_heuristic_pro_count,
    _fact_pro_priority,
    _GRADED_SCALE,
    _graded_strength,
    _worst_fact_objection_magnitude,
    fact_only_key,
    lexicographic_decide,
)

from dialectical_chess.graded_policy import ChessGradedPolicy  # noqa: E402
from dialectical_chess.probe import owned_board_from_fen, probe_moves  # noqa: E402


F11_FEN = "rnbqk1nr/1ppp1ppp/4p3/p7/3P2Q1/2P5/P1P2PPP/R1B1KBNR b KQkq - 0 5"
ASSERTED_MOVE = "g8f6"
QUEEN_FLANK_OBJ = "obj:king_safety:queen_flank_invasion"


def _selection_key_local(probe, graph):
    """Mirror of ``decider._selection_key`` returning component pieces."""
    objection_magnitude = _worst_fact_objection_magnitude(probe, graph)
    winning, large_material, crown, small_material = _fact_pro_priority(probe)
    graded_key = -round(_graded_strength(probe, graph) * _GRADED_SCALE)
    heuristic_pro_key = -_accepted_heuristic_pro_count(probe)
    return (
        objection_magnitude,
        -winning,
        -large_material,
        -crown,
        -small_material,
        graded_key,
        heuristic_pro_key,
        probe.child_eval,
        probe.move_id,
    )


def main() -> int:
    board = owned_board_from_fen(F11_FEN)
    probes = probe_moves(
        board,
        dialectic_depth=0,
        search_depth=2,
        search_backend="alphabeta",
        smt_mate=False,
        smt_fork=False,
    )

    policy = ChessGradedPolicy(board=board).with_probes(probes)
    graph = build_root_argument_graph(list(probes), policy=policy)
    decision = lexicographic_decide(probes, graph)
    decided_uci = decision.move_id if decision else None

    print("=" * 78)
    print(f"FEN: {F11_FEN}")
    print(f"Probes: {len(probes)}; survivors: {len(graph.survivors)}")
    print(f"Decided move: {decided_uci}")
    print("=" * 78)
    print()

    move_opinions = graph.ranking.get("move_opinions", {})

    # Print per-probe table.
    print("Per-probe state:")
    print(
        f"  {'uci':6} {'survivor':8} {'child_eval':>10} {'contested':9}  "
        f"{'opinion (b, d, u, a)':28}  "
        f"objections                                                  reasons"
    )
    by_uci = {p.move_id: p for p in probes}
    for uci in sorted(by_uci):
        probe = by_uci[uci]
        survives = probe.move_id in graph.survivors
        op = move_opinions.get(probe.move_id)
        op_str = (
            f"({op.b:.4f}, {op.d:.4f}, {op.u:.4f}, {op.a:.2f})"
            if op is not None
            else "n/a"
        )
        objs = ", ".join(probe.objections) if probe.objections else ""
        reas = ", ".join(probe.reasons) if probe.reasons else ""
        print(
            f"  {uci:6} {'Y' if survives else 'N':8} {probe.child_eval:>10d} "
            f"{str(probe.contested):9}  {op_str:28}  {objs:60}  {reas}"
        )

    print()
    print("=" * 78)
    print(f"Lex keys (smaller is better; chosen via min):")
    print(
        f"  {'uci':6} {'mag':>6} {'-win':>5} {'-lg':>5} {'-cr':>5} {'-sm':>5} "
        f"{'graded':>12} {'heur':>6} {'ceval':>8} move_id"
    )
    keyed = [(_selection_key_local(p, graph), p) for p in probes]
    keyed.sort()
    for key, probe in keyed:
        mag, nwin, nlg, ncr, nsm, gk, hk, ce, mid = key
        marker = "  <-- DECIDED" if probe.move_id == decided_uci else ""
        asserted = "  <-- ASSERTED" if probe.move_id == ASSERTED_MOVE else ""
        print(
            f"  {probe.move_id:6} {mag:>6d} {nwin:>5d} {nlg:>5d} {ncr:>5d} "
            f"{nsm:>5d} {gk:>12d} {hk:>6d} {ce:>8d} {mid}{marker}{asserted}"
        )

    print()
    print("=" * 78)
    print(f"FACT-only keys (terms 1-2) -- a unique min here would decide on FACT alone:")
    fact_keys = [(fact_only_key(p, graph), p.move_id) for p in probes]
    fact_keys.sort()
    best_fact = fact_keys[0][0]
    tied_on_fact = [uci for k, uci in fact_keys if k == best_fact]
    print(f"  best FACT-only key: {best_fact}")
    print(f"  probes tied at best FACT-only key: {len(tied_on_fact)} -> {tied_on_fact}")

    print()
    print("=" * 78)
    print(f"Survivor probes (graded layer is built only over these):")
    survivor_ucis = sorted(by_uci[u].move_id for u in by_uci if by_uci[u].move_id in graph.survivors)
    print(f"  {len(survivor_ucis)} survivors: {survivor_ucis}")

    print()
    print("=" * 78)
    print(f"Queen-flank-invasion attack on g8f6:")
    g8f6_probe = by_uci.get(ASSERTED_MOVE)
    if g8f6_probe is None:
        print("  g8f6 NOT in probes -- assertion will fail.")
    else:
        has_obj = any(QUEEN_FLANK_OBJ in o for o in g8f6_probe.objections)
        print(f"  g8f6 carries '{QUEEN_FLANK_OBJ}' in objections: {has_obj}")
        print(f"  g8f6 full objections tuple: {g8f6_probe.objections}")
        print(f"  g8f6 full reasons tuple: {g8f6_probe.reasons}")
        g8f6_op = move_opinions.get(g8f6_probe.move_id)
        if g8f6_op is not None:
            print(
                f"  g8f6 resolved graded opinion: "
                f"b={g8f6_op.b:.4f} d={g8f6_op.d:.4f} u={g8f6_op.u:.4f} "
                f"a={g8f6_op.a:.4f}, expectation={g8f6_op.expectation():.4f}"
            )

    print()
    print("=" * 78)
    print("Term-by-term analysis vs the next-best competitor:")
    if len(keyed) >= 2:
        chosen_key, chosen_probe = keyed[0]
        runner_key, runner_probe = keyed[1]
        terms = ["term1_obj_mag", "term2a_-win", "term2b_-lg", "term2c_-cr",
                 "term2d_-sm", "term3_graded", "term4_heur", "term5a_ceval",
                 "term5b_move_id"]
        for i, name in enumerate(terms):
            c = chosen_key[i]
            r = runner_key[i]
            verdict = "TIE" if c == r else ("CHOSEN<RUNNER (chosen wins here)" if c < r else "RUNNER<CHOSEN (chosen loses here)")
            if c != r:
                print(f"  {name}: chosen={chosen_probe.move_id}->{c}  runner={runner_probe.move_id}->{r}  {verdict}  *** DECIDED HERE ***")
                break
            else:
                print(f"  {name}: chosen={chosen_probe.move_id}->{c}  runner={runner_probe.move_id}->{r}  TIE")

    # Sorted move-scores top 10 for context
    print()
    print("=" * 78)
    print("Top-10 graded move_scores (term 3 input):")
    move_scores = graph.ranking.get("move_scores", {})
    top = sorted(move_scores.items(), key=lambda kv: -kv[1])[:10]
    for mid, sc in top:
        marker = "  <-- ASSERTED" if mid == ASSERTED_MOVE else ""
        print(f"  {mid}: {sc:.6f}{marker}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
