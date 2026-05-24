"""Chess MoveProbe — a frozen-dataclass subclass of the core MoveProbe.

The chess cartridge's per-move probe extends ``dialectical_games.MoveProbe``
with chess-specific scalars (uci, san, captured / promotion value, post-move
FEN) and chess-typed evidence tuples (reason_evidence / objection_evidence /
reply_attack_evidence). The inherited fields (move_id, reasons, objections,
reply_attacks, defenses, child_eval, contested, search_score, search_line,
score) carry the **core-taxonomy** labels and the core-decoder inputs; the
chess extensions carry the chess-flavoured witness data for the chess
cartridge's own reasoning and diagnostics. Per the Phase-3 foreman
directive 3, chess HEURISTIC vocabulary enters the core only through
the explicit chess-to-core translator; the chess-flavoured originals stay
on the extension fields for diagnostics.

Pyright requires the subclass to be ``@dataclass(frozen=True)`` to inherit
the parent's frozen behaviour cleanly (Python's dataclass machinery).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from dialectical_games.arguments import MoveProbe as CoreMoveProbe

from dialectical_chess.evidence import ArgumentEvidence


@dataclass(frozen=True)
class MoveProbe(CoreMoveProbe):
    """Chess move probe — extends the core probe with chess-typed evidence.

    The canonical chess move identifier is ``uci``; the inherited core
    ``move_id`` field is kept in sync with ``uci`` automatically via
    ``__post_init__`` so the core orchestrator reads a non-empty
    ``move_id`` without requiring chess callers to set both.
    """

    # Override core ``move_id`` with a default so chess callers may
    # construct a MoveProbe with chess fields only; ``__post_init__``
    # falls back to ``uci`` when the caller did not pass move_id.
    move_id: str = ""

    # Chess-flavoured scalar fields (the cartridge's per-move static data).
    uci: str = ""
    san: str = ""
    is_checkmate: bool = False
    gives_check: bool = False
    is_capture: bool = False
    captured_value: int = 0
    promotion_value: int = 0
    smt_witnesses: tuple[str, ...] = ()
    post_fen: str | None = None
    # Chess-typed evidence tuples (chess-flavoured labels for chess-side
    # reasoning / diagnostics). The inherited core ``reasons`` / ``objections``
    # / ``reply_attacks`` tuples carry the **core-taxonomy** label strings the
    # core graph builder reads.
    reason_evidence: tuple[ArgumentEvidence, ...] = ()
    objection_evidence: tuple[ArgumentEvidence, ...] = ()
    reply_attack_evidence: tuple[ArgumentEvidence, ...] = ()

    def __post_init__(self) -> None:
        # Keep inherited ``move_id`` in sync with chess ``uci`` for callers
        # that construct a chess MoveProbe with only chess fields. Frozen
        # dataclasses forbid direct assignment; ``object.__setattr__`` is
        # the standard escape hatch the dataclass docs recommend.
        if not self.move_id and self.uci:
            object.__setattr__(self, "move_id", self.uci)


def choose_move(
    probes: list[MoveProbe],
    *,
    deadline: float | None = None,
) -> MoveProbe | None:
    """Decision-only convenience wrapper for chess callers.

    Builds the core argument graph with a per-call chess graded policy and
    invokes the core lexicographic decider. Callers that already have a
    cartridge / engine should call ``dialectical_games.engine.analyze``
    directly; this helper exists for the few chess-side call sites that
    want a probe-list-in / chosen-probe-out signature.
    """
    if not probes:
        raise ValueError("position has no legal moves")
    # Lazy import to avoid a circular import chain (graded_policy imports
    # arguments via static_prior).
    from dialectical_chess.board import OwnedBoard
    from dialectical_chess.graded_policy import ChessGradedPolicy
    from dialectical_games.arguments import build_root_argument_graph
    from dialectical_games.decider import lexicographic_decide

    # Reconstruct the root board from any probe's post_fen for the graded
    # policy; the policy uses it only as a per-build cache key. When no
    # probe has a post_fen (synthetic tests), the policy is built without a
    # board reference and only reads probe.child_eval.
    board = None
    for probe in probes:
        if probe.post_fen is not None:
            board = OwnedBoard.from_fen(probe.post_fen, legal_game=False)
            break
    policy = ChessGradedPolicy(board=board)
    graph = build_root_argument_graph(list(probes), policy)
    chosen = lexicographic_decide(tuple(probes), graph)
    return chosen if isinstance(chosen, MoveProbe) else (
        # cast: lexicographic_decide returns CoreMoveProbe | None; in practice
        # every probe in the input tuple is a chess MoveProbe.
        chosen  # type: ignore[return-value]
    )


def build_argument_payload(probes: list[MoveProbe]) -> dict[str, Any]:
    return {
        "move_scores": [probe_payload(probe) for probe in probes],
    }


def probe_payload(probe: MoveProbe) -> dict[str, Any]:
    payload = asdict(probe)
    payload.pop("reason_evidence", None)
    payload.pop("objection_evidence", None)
    payload.pop("reply_attack_evidence", None)
    return payload
