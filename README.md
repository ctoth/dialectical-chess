# dialectical-chess

`dialectical-chess` is a UCI-capable chess engine experiment built around
explicit dialectical move arguments. It uses an owned chess move-generation
substrate, shallow tactical/search witnesses, optional Z3-backed tactical
checks, and the `formal-argumentation` package for argument semantics and
optimization.

The engine is experimental. It is meant for research, diagnostics, and small
benchmarks rather than competitive chess strength.

## Install

```powershell
uv sync
```

The package pins `formal-argumentation` to a GitHub commit SHA so a clean
checkout can resolve the same argumentation library revision.

## Commands

Run the default probe:

```powershell
uv run dialectical-chess-probe
```

Run as a UCI engine:

```powershell
uv run dialectical-chess-probe --uci
```

Run the owned move-generation self-test:

```powershell
uv run dialectical-chess-owned --selftest
```

Run the built-in EPD smoke benchmark:

```powershell
uv run dialectical-chess-bench --epd .\dialectical_chess\fixtures\dialectical_chess_smoke.epd
```

## Development

```powershell
uv run pytest
uv run pyright
```

The committed fixtures under `dialectical_chess/fixtures` are intentionally
small. Larger benchmark corpora and generated diagnostic artifacts should stay
outside the package unless they are deliberately promoted.
