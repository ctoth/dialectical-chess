"""UCI protocol loop for the dialectical chess probe engine."""

from __future__ import annotations

from dataclasses import replace
from typing import TextIO

from dialectical_chess.board import START_FEN
from dialectical_chess.engine import DialecticalChessEngine, EngineSettings
from dialectical_chess.probe import owned_board_from_fen
from dialectical_chess.search import ReplyAnalysisSettings


UciSettings = EngineSettings


def run_uci(
    input_stream: TextIO,
    output_stream: TextIO,
    *,
    dialectic_depth: int = 1,
    search_depth: int = 0,
    search_backend: str = "negamax",
    smt_mate: bool = True,
    smt_fork: bool = True,
    selector_mode: str = "argument",
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
        selector_mode=selector_mode,
        positional_reasons=positional_reasons,
        reply_mate_scan=reply_mate_scan,
        reply_analysis=reply_analysis or ReplyAnalysisSettings(),
    )
    board = owned_board_from_fen(START_FEN)
    recent_own_move: str | None = None
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
            board = owned_board_from_fen(START_FEN)
            recent_own_move = None
        elif command.startswith("position "):
            try:
                board, parsed_recent_own_move = parse_uci_position_state(command)
                if parsed_recent_own_move is not None or "moves" in command.split():
                    recent_own_move = parsed_recent_own_move
            except ValueError as exc:
                _uci_write(output_stream, f"info string invalid position: {exc}")
        elif command.startswith("go") or command == "stop":
            move_settings = settings_for_go(
                replace(settings, recent_own_move=recent_own_move),
                board,
                command,
            )
            chosen_move = choose_uci_move(board, settings=move_settings, output_stream=output_stream)
            if chosen_move != "0000":
                recent_own_move = chosen_move
            _uci_write(output_stream, "bestmove " + chosen_move)
        elif command == "quit":
            return 0
        elif command.startswith("setoption ") or command == "ponderhit":
            continue
        else:
            _uci_write(output_stream, f"info string unsupported command: {command}")


def parse_uci_position(command: str):
    return parse_uci_position_state(command)[0]


def parse_uci_position_state(command: str):
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

    move_history: list[str] = []
    if index < len(tokens):
        if tokens[index] != "moves":
            raise ValueError(f"unexpected token: {tokens[index]}")
        legal_by_uci = {move.uci(): move for move in board.legal_moves()}
        for move_text in tokens[index + 1 :]:
            move = legal_by_uci.get(move_text)
            if move is None:
                raise ValueError(f"illegal move {move_text}")
            move_history.append(move_text)
            board = board.apply(move)
            legal_by_uci = {next_move.uci(): next_move for next_move in board.legal_moves()}
    recent_own_move = move_history[-2] if len(move_history) >= 2 else None
    return board, recent_own_move


def choose_uci_move(
    board,
    *,
    settings: UciSettings | None = None,
    dialectic_depth: int = 1,
    search_depth: int = 0,
    search_backend: str = "negamax",
    smt_mate: bool = True,
    smt_fork: bool = True,
    selector_mode: str = "argument",
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
        selector_mode=selector_mode,
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
        _uci_write(output_stream, f"info string selector_mode={settings.selector_mode}")
        _uci_write(output_stream, f"info string positional_reasons={settings.positional_reasons}")
        _uci_write(output_stream, f"info string reply_mate_scan={settings.reply_mate_scan}")
        _uci_write(output_stream, f"info string reply_analysis={settings.reply_analysis}")
        if decision.selected.optimizer_trace:
            _uci_write(
                output_stream,
                f"info string optimizer_status={decision.selected.optimizer_trace.get('status')}",
            )
        _uci_write(output_stream, f"info score cp {decision.selected.score} pv {decision.move_uci}")
    return decision.move_uci


def settings_for_go(settings: EngineSettings, board, command: str) -> EngineSettings:
    if not command.startswith("go"):
        return settings
    remaining = own_remaining_ms(board, command)
    if remaining is None:
        return settings
    search_depth = settings.search_depth
    if remaining <= 2_500:
        search_depth = min(search_depth, 0)
        return replace(
            settings,
            dialectic_depth=0,
            search_depth=search_depth,
            smt_mate=False,
            smt_fork=False,
            positional_reasons=False,
            reply_mate_scan=False,
        )
    if remaining <= 12_000:
        return replace(
            settings,
            dialectic_depth=0,
            search_depth=min(search_depth, 0),
            reply_mate_scan=False,
        )
    elif remaining <= 20_000:
        return replace(
            settings,
            dialectic_depth=0,
            search_depth=min(search_depth, 1),
            reply_mate_scan=False,
        )
    if search_depth == settings.search_depth:
        return settings
    return replace(settings, search_depth=search_depth)


def own_remaining_ms(board, command: str) -> int | None:
    tokens = command.split()
    field = "wtime" if board.turn == "w" else "btime"
    for index, token in enumerate(tokens[:-1]):
        if token == field:
            try:
                return int(tokens[index + 1])
            except ValueError:
                return None
    return None


def _uci_write(output_stream: TextIO, line: str) -> None:
    print(line, file=output_stream, flush=True)
