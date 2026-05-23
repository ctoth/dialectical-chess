"""Reusable modules for the dialectical chess sidecar scripts.

Core Phase 3 chunk B intermediate (delete-first): the legacy
``argumentation_cartridge`` re-exports have been removed. Chunk E rebuilds
this surface against ``dialectical_games`` (re-exporting ``EngineAnalysis``,
``EngineDecision``, ``EngineSettings`` from the core, with a chess
cartridge ``EngineDecision.move_uci`` alias).
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
