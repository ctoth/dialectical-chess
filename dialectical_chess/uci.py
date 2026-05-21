"""UCI protocol loop for the dialectical chess probe engine."""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Sequence, TextIO

from dialectical_chess.board import START_FEN
from dialectical_chess.engine import DialecticalChessEngine, EngineSettings
from dialectical_chess.probe import owned_board_from_fen
from dialectical_chess.search import ReplyAnalysisSettings, position_repetition_key
from dialectical_chess.tuning import (
    CRITICAL_BUDGET_MS,
    LOW_BUDGET_MS,
    REPLY_MATE_SCAN_MIN_BUDGET_MS,
    TIME_DEFAULT_MOVES_TO_GO,
    TIME_INCREMENT_FRACTION,
    TIME_MAX_MOVE_FRACTION,
    TIME_MAX_RESERVE_MS,
    TIME_MIN_BUDGET_MS,
    TIME_MIN_RESERVE_MS,
    TIME_OVERHEAD_MS,
    TIME_RESERVE_FRACTION,
)


UciSettings = EngineSettings


@dataclass(frozen=True)
class GameState:
    board: Any
    position_history: tuple[str, ...]


@dataclass(frozen=True)
class GoRequest:
    wtime: int | None = None
    btime: int | None = None
    winc: int | None = None
    binc: int | None = None
    movestogo: int | None = None
    movetime: int | None = None
    depth: int | None = None
    nodes: int | None = None
    infinite: bool = False


@dataclass(frozen=True)
class ActiveSearch:
    thread: threading.Thread
    result: queue.Queue[str]


def game_state_for_board(board: Any) -> GameState:
    return GameState(
        board=board,
        position_history=(position_repetition_key(board),),
    )


def run_uci(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    dialectic_depth: int = 1,
    search_depth: int = 0,
    search_backend: str = "negamax",
    smt_mate: bool = True,
    smt_fork: bool = True,
    positional_reasons: bool = True,
    reply_mate_scan: bool = True,
    reply_analysis: ReplyAnalysisSettings | None = None,
) -> int:
    settings = EngineSettings(
        dialectic_depth=dialectic_depth,
        search_depth=search_depth,
        search_backend=search_backend,
        smt_mate=smt_mate,
        smt_fork=smt_fork,
        positional_reasons=positional_reasons,
        reply_mate_scan=reply_mate_scan,
        reply_analysis=reply_analysis or ReplyAnalysisSettings(),
    )
    game_state = game_state_for_board(owned_board_from_fen(START_FEN))
    active_search: ActiveSearch | None = None
    while True:
        raw = input_stream.readline()
        if raw == "":
            return 0
        command = raw.strip()
        if not command:
            continue

        if command == "uci":
            _uci_write(output_stream, "id name DialecticalChessProbe")
            _uci_write(output_stream, "id author argumentation")
            _uci_write(output_stream, "uciok")
        elif command == "isready":
            _uci_write(output_stream, "readyok")
        elif command == "ucinewgame":
            game_state = game_state_for_board(owned_board_from_fen(START_FEN))
        elif command.startswith("position "):
            try:
                game_state = parse_uci_position_state(command)
            except ValueError as exc:
                _uci_write(output_stream, f"info string invalid position: {exc}")
        elif command.startswith("go"):
            request = parse_go(command.split())
            move_settings, budget_ms = settings_for_go_request(
                replace(
                    settings,
                    position_history=game_state.position_history,
                ),
                game_state.board,
                request,
            )
            if budget_ms is not None and budget_ms <= 0:
                _uci_write(output_stream, "bestmove " + best_available_move(game_state.board))
            elif request.infinite:
                active_search = start_search(game_state.board, settings=move_settings, output_stream=output_stream)
            else:
                chosen_move = choose_uci_move(game_state.board, settings=move_settings, output_stream=output_stream)
                _uci_write(output_stream, "bestmove " + chosen_move)
        elif command == "stop":
            if active_search is not None:
                _uci_write(output_stream, "bestmove " + finish_search(active_search))
                active_search = None
        elif command == "quit":
            return 0
        elif command.startswith("setoption ") or command == "ponderhit":
            continue
        else:
            _uci_write(output_stream, f"info string unsupported command: {command}")


def parse_uci_position(command: str):
    return parse_uci_position_state(command).board


def parse_uci_position_state(command: str) -> GameState:
    tokens = command.split()
    if len(tokens) < 2 or tokens[0] != "position":
        raise ValueError(command)

    index = 1
    if tokens[index] == "startpos":
        board = owned_board_from_fen(START_FEN)
        index += 1
    elif tokens[index] == "fen":
        index += 1
        fen_start = index
        while index < len(tokens) and tokens[index] != "moves":
            index += 1
        fen_fields = tokens[fen_start:index]
        if len(fen_fields) != 6:
            raise ValueError("fen position must contain six FEN fields")
        board = owned_board_from_fen(" ".join(fen_fields))
    else:
        raise ValueError("position must use startpos or fen")

    position_history = [position_repetition_key(board)]
    if index < len(tokens):
        if tokens[index] != "moves":
            raise ValueError(f"unexpected token: {tokens[index]}")
        legal_by_uci = {move.uci(): move for move in board.legal_moves()}
        for move_text in tokens[index + 1 :]:
            move = legal_by_uci.get(move_text)
            if move is None:
                raise ValueError(f"illegal move {move_text}")
            board = board.apply(move)
            position_history.append(position_repetition_key(board))
            legal_by_uci = {next_move.uci(): next_move for next_move in board.legal_moves()}
    return GameState(board=board, position_history=tuple(position_history))


def choose_uci_move(
    board,
    *,
    settings: UciSettings | None = None,
    dialectic_depth: int = 1,
    search_depth: int = 0,
    search_backend: str = "negamax",
    smt_mate: bool = True,
    smt_fork: bool = True,
    positional_reasons: bool = True,
    reply_mate_scan: bool = True,
    reply_analysis: ReplyAnalysisSettings | None = None,
    output_stream: TextIO | None = None,
) -> str:
    settings = settings or EngineSettings(
        dialectic_depth=dialectic_depth,
        search_depth=search_depth,
        search_backend=search_backend,
        smt_mate=smt_mate,
        smt_fork=smt_fork,
        positional_reasons=positional_reasons,
        reply_mate_scan=reply_mate_scan,
        reply_analysis=reply_analysis or ReplyAnalysisSettings(),
    )
    try:
        decision = DialecticalChessEngine(settings).choose_move(board)
    except ValueError as exc:
        if output_stream is not None:
            _uci_write(output_stream, f"info string {exc}")
        return "0000"
    if decision.selected is None:
        return "0000"
    if output_stream is not None:
        _uci_write(output_stream, f"info string positional_reasons={settings.positional_reasons}")
        _uci_write(output_stream, f"info string reply_mate_scan={settings.reply_mate_scan}")
        _uci_write(output_stream, f"info string reply_analysis={settings.reply_analysis}")
        _uci_write(output_stream, f"info score cp {decision.selected.score} pv {decision.move_uci}")
    return decision.move_uci


def parse_go(tokens: Sequence[str]) -> GoRequest:
    values: dict[str, int | None] = {
        "wtime": None,
        "btime": None,
        "winc": None,
        "binc": None,
        "movestogo": None,
        "movetime": None,
        "depth": None,
        "nodes": None,
    }
    infinite = False
    for index, token in enumerate(tokens):
        if token == "infinite":
            infinite = True
        elif token in values and index + 1 < len(tokens):
            values[token] = parsed_int(tokens[index + 1])
    return GoRequest(infinite=infinite, **values)


def parsed_int(text: str) -> int | None:
    try:
        return int(text)
    except ValueError:
        return None


def settings_for_go_request(settings: EngineSettings, board, request: GoRequest) -> tuple[EngineSettings, int | None]:
    if request.depth is not None:
        settings = replace(settings, search_depth=max(0, request.depth))
    if request.infinite:
        # An infinite search runs until `stop`: no budget, no deadline, even
        # when a clock is present (`go infinite wtime ...` is legal UCI).
        return replace(settings, deadline=None), None
    budget_ms = estimated_move_budget_ms(board.turn, request)
    if budget_ms is None or budget_ms <= 0:
        return settings, budget_ms
    settings = replace(settings, deadline=time.monotonic() + budget_ms / 1000.0)
    return settings_for_budget(settings, budget_ms), budget_ms


def estimated_move_budget_ms(turn: str, request: GoRequest) -> int | None:
    if request.movetime is not None:
        return max(TIME_MIN_BUDGET_MS, request.movetime - TIME_OVERHEAD_MS)
    remaining = request.wtime if turn == "w" else request.btime
    if remaining is None:
        return None
    increment = request.winc if turn == "w" else request.binc
    moves = request.movestogo or TIME_DEFAULT_MOVES_TO_GO
    base = remaining / max(moves, 1)
    budget = base + TIME_INCREMENT_FRACTION * (increment or 0) - TIME_OVERHEAD_MS
    reserve = max(TIME_MIN_RESERVE_MS, min(TIME_MAX_RESERVE_MS, TIME_RESERVE_FRACTION * remaining))
    upper_bound = min(remaining - reserve, TIME_MAX_MOVE_FRACTION * remaining)
    if upper_bound <= 0:
        return int(upper_bound)
    return int(clamp(budget, TIME_MIN_BUDGET_MS, upper_bound))


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(value, upper))


def settings_for_budget(settings: EngineSettings, budget_ms: int) -> EngineSettings:
    if budget_ms <= CRITICAL_BUDGET_MS:
        return replace(
            settings,
            dialectic_depth=0,
            search_depth=0,
            smt_mate=False,
            smt_fork=False,
            positional_reasons=False,
            reply_mate_scan=False,
        )
    if budget_ms <= LOW_BUDGET_MS:
        return replace(
            settings,
            dialectic_depth=0,
            search_depth=min(settings.search_depth, 1),
            reply_mate_scan=False,
        )
    if budget_ms <= REPLY_MATE_SCAN_MIN_BUDGET_MS:
        return replace(settings, reply_mate_scan=False)
    return settings


def start_search(board, *, settings: EngineSettings, output_stream: TextIO | None) -> ActiveSearch:
    result: queue.Queue[str] = queue.Queue()

    def search() -> None:
        try:
            result.put(choose_uci_move(board, settings=settings, output_stream=output_stream))
        except Exception:
            result.put("0000")

    thread = threading.Thread(target=search, daemon=True)
    thread.start()
    return ActiveSearch(thread=thread, result=result)


def finish_search(active_search: ActiveSearch) -> str:
    active_search.thread.join()
    try:
        return active_search.result.get_nowait()
    except queue.Empty:
        return "0000"


def best_available_move(board) -> str:
    legal_moves = sorted(board.legal_moves(), key=lambda move: move.uci())
    if not legal_moves:
        return "0000"
    return legal_moves[0].uci()


def _uci_write(output_stream: TextIO, line: str) -> None:
    print(line, file=output_stream, flush=True)
