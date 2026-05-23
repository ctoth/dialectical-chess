"""Chess cartridge: :class:`ChessForcedLossResolver`.

Implements the core ``dialectical_games.forced_loss.ForcedLossResolver``
Protocol by wrapping :func:`dialectical_chess.loss_mining.has_forced_mate`.
Used by the core's ``loss_mining`` diagnostic — given a position before an
engine move and the move played, returns a :class:`ForcedLoss` if the move
leaves the opponent with a proven forced mate, else ``None``.

Chess does not net material the way checkers does (chess's blunders are
typically tactical mates rather than material shots); ``material_net=0`` is
the honest "mate-only resolver, no material netting" value per the core
``ForcedLoss`` docstring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from dialectical_games.board import Board, Move
from dialectical_games.forced_loss import ForcedLoss

from dialectical_chess.board import OwnedBoard, OwnedMove
from dialectical_chess.loss_mining import has_forced_mate


@dataclass(frozen=True)
class ChessForcedLossResolver:
    """``ForcedLossResolver`` impl for chess: proven forced mate after the
    candidate move yields a ``ForcedLoss(material_net=0, wins_game=True)``.
    """

    mate_depth: int = 1
    deadline: float | None = None

    def opponent_loss(self, board: Board, move: Move) -> ForcedLoss | None:
        owned = cast(OwnedBoard, board)
        owned_move = cast(OwnedMove, move)
        child = owned.apply(owned_move)
        if has_forced_mate(
            child, mate_depth=self.mate_depth, deadline=self.deadline
        ):
            return ForcedLoss(material_net=0, wins_game=True)
        return None
