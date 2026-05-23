"""Chess cartridge: search-backend registry.

Registers the chess negamax / alphabeta search backends in the core's
``SearchBackendRegistry``. Today no chess post-decision hook dispatches
through this registry (the reply-mate fixpoint is its own hook; negamax
runs at probe time inside ``probe.py``); the registry exists for symmetry
with checkers and as the slot for a future post-decision chess backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dialectical_games.arguments import MoveProbe
from dialectical_games.search_backend import SearchBackend, SearchBackendRegistry


@dataclass(frozen=True)
class NegamaxBackend:
    """Chess negamax / alphabeta search backend.

    The ``run`` method is a stub today — chess never invokes a backend
    through this Protocol path; the negamax driver lives in
    ``dialectical_chess.search.root_search_result`` and is called from
    ``probe.py`` at probe time, before the core graph is built. The
    backend is registered so the registry is non-empty for callers that
    validate ``EngineSettings.search_backend`` against ``registry.names``.
    """

    name: str = "negamax"

    def run(
        self,
        *,
        board: object,
        probes: tuple[MoveProbe, ...],
        settings: Any,
        deadline: float | None,
    ) -> MoveProbe:
        if not probes:
            raise ValueError("NegamaxBackend.run requires at least one probe")
        # No post-decision search dispatch in the chess cartridge today.
        # A backend that wanted to swap the selection would re-run negamax
        # here; for now we return the first probe so the Protocol is
        # satisfied without doing any extra work.
        return probes[0]


@dataclass(frozen=True)
class AlphaBetaBackend:
    """Chess alphabeta search backend (same dispatch path as negamax).

    Chess's ``search.py`` distinguishes negamax from alphabeta by an
    internal flag; from the registry's perspective they are two named
    entry points to the same underlying driver. Registered separately so
    callers can validate ``settings.search_backend == "alphabeta"``.
    """

    name: str = "alphabeta"

    def run(
        self,
        *,
        board: object,
        probes: tuple[MoveProbe, ...],
        settings: Any,
        deadline: float | None,
    ) -> MoveProbe:
        if not probes:
            raise ValueError("AlphaBetaBackend.run requires at least one probe")
        return probes[0]


SEARCH_BACKEND_REGISTRY: SearchBackendRegistry = SearchBackendRegistry()
SEARCH_BACKEND_REGISTRY.register(NegamaxBackend())
SEARCH_BACKEND_REGISTRY.register(AlphaBetaBackend())


__all__ = [
    "AlphaBetaBackend",
    "NegamaxBackend",
    "SEARCH_BACKEND_REGISTRY",
]
