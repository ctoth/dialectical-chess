"""UCI match orchestration for dialectical chess benchmarks."""

from __future__ import annotations

import re
import queue
import shutil
import subprocess
import threading
from argparse import Namespace
from collections import Counter
from pathlib import Path
from typing import Any

import chess

from dialectical_chess.baselines import fastchess_baseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INTERNAL_MATCH_MOVETIME_MS = 1000
INTERNAL_MATCH_BESTMOVE_TIMEOUT_SECONDS = 5.0


def run_uci_match(args: Namespace) -> dict[str, Any]:
    cutechess = shutil.which("cutechess-cli")
    fastchess = shutil.which("fastchess") or shutil.which("fast-chess")
    uv_executable = shutil.which("uv") or "uv"
    dialectical_args = dialectical_uci_args(args)
    cutechess_args = [
        "-engine",
        'name=Dialectical cmd="uv" '
        + " ".join(f'arg="{token}"' for token in dialectical_args)
        + " proto=uci",
        "-engine",
        'name=DialecticalNoSMT cmd="uv" arg="run" arg="dialectical-chess-probe" arg="--uci" arg="--no-smt-mate" proto=uci',
        "-each",
        f"tc={args.match_tc}",
        "-games",
        str(args.match_games),
        "-repeat",
    ]
    fastchess_args = build_fastchess_args(args, uv_executable)
    if cutechess and args.match_baseline == "nosmt":
        command = [cutechess, *cutechess_args]
        runner = "cutechess-cli"
    elif fastchess:
        command = [fastchess, *fastchess_args]
        runner = "fastchess"
    else:
        return {
            "ok": False,
            "mode": "uci_match",
            "blocked": "missing cutechess-cli, fastchess, or fast-chess executable on PATH",
            "suggested_command": "fast-chess " + " ".join(fastchess_args),
        }
    if not args.run_uci_match:
        return {
            "ok": True,
            "mode": "uci_match",
            "runner": runner,
            "baseline": args.match_baseline,
            "requested_games": args.match_games,
            "command": command,
        }
    prepare_match_outputs(args)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    failures = parse_uci_match_failures(completed.stdout)
    games_played = parse_uci_match_games_played(completed.stdout)
    return {
        "ok": completed.returncode == 0 and not any(failures.values()) and games_played == args.match_games,
        "mode": "uci_match",
        "runner": runner,
        "baseline": args.match_baseline,
        "requested_games": args.match_games,
        "games_played": games_played,
        "command": command,
        "returncode": completed.returncode,
        "failures": failures,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def build_fastchess_command(args: Namespace, *, fastchess: str, uv_executable: str) -> list[str]:
    return [fastchess, *build_fastchess_args(args, uv_executable)]


def build_fastchess_args(args: Namespace, uv_executable: str) -> list[str]:
    baseline_name, baseline_args = fastchess_baseline(args.match_baseline, uv_executable, args)
    games_per_round = 2 if args.match_games > 1 else 1
    rounds = max(1, (args.match_games + games_per_round - 1) // games_per_round)
    command = [
        "-engine",
        "name=Dialectical",
        f"cmd={uv_executable}",
        "args=" + " ".join(dialectical_uci_args(args)),
        "proto=uci",
        f"dir={PROJECT_ROOT}",
        "-engine",
        f"name={baseline_name}",
        *baseline_args,
        "-each",
        f"tc={args.match_tc}",
        "-rounds",
        str(rounds),
        "-games",
        str(games_per_round),
        "-openings",
        f"file={args.match_openings}",
        "format=epd",
        "order=sequential",
        "-maxmoves",
        str(max(1, args.match_max_plies // 2)),
        "-concurrency",
        "1",
    ]
    pgn_out = getattr(args, "match_pgn_out", None)
    if pgn_out is not None:
        pgn_path = match_output_path(pgn_out)
        command.extend(
            [
                "-pgnout",
                f"file={pgn_path}",
                "notation=uci",
                "append=false",
            ]
        )
    return command


def prepare_match_outputs(args: Namespace) -> None:
    pgn_out = getattr(args, "match_pgn_out", None)
    if pgn_out is not None:
        match_output_path(pgn_out).parent.mkdir(parents=True, exist_ok=True)


def match_output_path(path: Path | str) -> Path:
    output_path = Path(path)
    if output_path.is_absolute():
        return output_path
    return PROJECT_ROOT / output_path


def dialectical_uci_args(args: Namespace) -> list[str]:
    command = [
        "run",
        "dialectical-chess-probe",
        "--uci",
        "--dialectic-depth",
        str(args.dialectic_depth),
        "--search-depth",
        str(args.search_depth),
        "--search-backend",
        args.search_backend,
        "--reply-max-replies",
        str(args.reply_max_replies),
        "--reply-max-defense-nodes",
        str(args.reply_max_defense_nodes),
        "--reply-min-defense-material",
        str(args.reply_min_defense_material),
    ]
    if not args.smt_mate:
        command.append("--no-smt-mate")
    if not getattr(args, "smt_fork", True):
        command.append("--no-smt-fork")
    if not getattr(args, "positional_reasons", True):
        command.append("--no-positional-reasons")
    return command


def parse_uci_match_failures(stdout: str) -> dict[str, int]:
    timeouts = sum(int(match.group(1)) for match in re.finditer(r"\bTimeouts:\s+(\d+)", stdout))
    crashes = sum(int(match.group(1)) for match in re.finditer(r"\bCrashed:\s+(\d+)", stdout))
    losses_on_time = len(re.findall(r"\bloses on time\b", stdout, flags=re.IGNORECASE))
    return {
        "timeouts": timeouts,
        "crashes": crashes,
        "losses_on_time": losses_on_time,
    }


def parse_uci_match_games_played(stdout: str) -> int | None:
    for pattern in (
        r"\bGames:\s+(\d+)\b",
        r"\bFinished\s+(\d+)\s+games?\b",
        r"\b(\d+)\s+games?\s+played\b",
    ):
        match = re.search(pattern, stdout, flags=re.IGNORECASE)
        if match is not None:
            return int(match.group(1))
    return None


def run_internal_uci_match(args: Namespace) -> dict[str, Any]:
    games = []
    crashes = 0
    illegal_moves = 0
    losses_on_time = 0
    for game_index in range(args.match_games):
        white_args = [] if game_index % 2 == 0 else ["--no-smt-mate"]
        black_args = ["--no-smt-mate"] if game_index % 2 == 0 else []
        result = play_internal_uci_game(white_args, black_args, args.match_max_plies)
        crashes += result["crashes"]
        illegal_moves += result["illegal_moves"]
        losses_on_time += result["losses_on_time"]
        games.append(result)
    result_counts = Counter(game["result"] for game in games)
    return {
        "ok": crashes == 0 and illegal_moves == 0 and losses_on_time == 0 and len(games) == args.match_games,
        "mode": "selfplay_smoke",
        "requested_games": args.match_games,
        "games": len(games),
        "max_plies": args.match_max_plies,
        "movetime_ms": INTERNAL_MATCH_MOVETIME_MS,
        "result_counts": dict(sorted(result_counts.items())),
        "crashes": crashes,
        "illegal_moves": illegal_moves,
        "losses_on_time": losses_on_time,
        "results": games,
    }


def play_internal_uci_game(
    white_extra_args: list[str],
    black_extra_args: list[str],
    max_plies: int,
) -> dict[str, Any]:
    white = start_uci_engine(white_extra_args)
    black = start_uci_engine(black_extra_args)
    board = chess.Board()
    moves: list[str] = []
    crashes = 0
    illegal_moves = 0
    losses_on_time = 0
    try:
        initialize_uci(white)
        initialize_uci(black)
        for _ply in range(max_plies):
            if board.is_game_over(claim_draw=True):
                break
            engine = white if board.turn == chess.WHITE else black
            send_uci(engine, "position startpos" + ("" if not moves else " moves " + " ".join(moves)))
            send_uci(engine, f"go movetime {INTERNAL_MATCH_MOVETIME_MS}")
            try:
                bestmove = read_bestmove(engine, timeout_seconds=INTERNAL_MATCH_BESTMOVE_TIMEOUT_SECONDS)
            except TimeoutError:
                losses_on_time += 1
                break
            if bestmove == "0000":
                break
            move = chess.Move.from_uci(bestmove)
            if move not in board.legal_moves:
                illegal_moves += 1
                break
            board.push(move)
            moves.append(bestmove)
        result = board.result(claim_draw=True) if board.is_game_over(claim_draw=True) else "1/2-1/2"
    except Exception:
        crashes += 1
        result = "*"
    finally:
        stop_uci_engine(white)
        stop_uci_engine(black)
    return {
        "result": result,
        "plies": len(moves),
        "moves": moves,
        "final_fen": board.fen(),
        "crashes": crashes,
        "illegal_moves": illegal_moves,
        "losses_on_time": losses_on_time,
    }


def start_uci_engine(extra_args: list[str]) -> subprocess.Popen[str]:
    return subprocess.Popen(
        ["uv", "run", "dialectical-chess-probe", "--uci", *extra_args],
        cwd=PROJECT_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def initialize_uci(process: subprocess.Popen[str]) -> None:
    send_uci(process, "uci")
    read_until(process, "uciok")
    send_uci(process, "isready")
    read_until(process, "readyok")
    send_uci(process, "ucinewgame")


def send_uci(process: subprocess.Popen[str], command: str) -> None:
    if process.stdin is None:
        raise RuntimeError("UCI process has no stdin")
    process.stdin.write(command + "\n")
    process.stdin.flush()


def read_until(process: subprocess.Popen[str], needle: str) -> list[str]:
    lines = []
    if process.stdout is None:
        raise RuntimeError("UCI process has no stdout")
    while True:
        line = process.stdout.readline()
        if line == "":
            raise RuntimeError("UCI process exited")
        line = line.strip()
        lines.append(line)
        if line == needle:
            return lines


def read_bestmove(process: subprocess.Popen[str], *, timeout_seconds: float = INTERNAL_MATCH_BESTMOVE_TIMEOUT_SECONDS) -> str:
    if process.stdout is None:
        raise RuntimeError("UCI process has no stdout")
    stdout = process.stdout
    lines: queue.Queue[str] = queue.Queue()

    def read_line() -> None:
        lines.put(stdout.readline())

    while True:
        reader = threading.Thread(target=read_line, daemon=True)
        reader.start()
        try:
            line = lines.get(timeout=timeout_seconds)
        except queue.Empty as exc:
            process.kill()
            raise TimeoutError("timed out waiting for bestmove") from exc
        if line == "":
            raise RuntimeError("UCI process exited")
        line = line.strip()
        if line.startswith("bestmove "):
            return line.split()[1]


def stop_uci_engine(process: subprocess.Popen[str]) -> None:
    try:
        send_uci(process, "quit")
    except Exception:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
