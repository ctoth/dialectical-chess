"""dialectical-chess — chess cartridge over the dialectical-games core.

The package surface re-exports the chess engine, settings, and decision /
analysis carriers. The argumentation orchestration lives in
``dialectical_games.engine.analyze``; the chess cartridge implements the
core ``Cartridge`` Protocol via :class:`DialecticalChessEngine`.

Core Phase 3: ``argumentation_cartridge`` / ``decide`` / ``opinion_graph`` /
``scheme`` / ``move_argument`` / ``skeptical_filter`` have been deleted; the
generic surface they implemented now lives in ``dialectical_games.*``. The
chess ``EngineDecision`` carries a ``move_uci`` alias property for backwards
compat with chess test sites that read ``decision.move_uci``.
"""

from dialectical_chess.engine import (
    DialecticalChessEngine,
    EngineAnalysis,
    EngineDecision,
    EngineSettings,
)

__all__ = [
    "DialecticalChessEngine",
    "EngineAnalysis",
    "EngineDecision",
    "EngineSettings",
]
