# Steering testing

Research workspace for mech-interp-style LLM modification experiments.

The project is intentionally lightweight at the start: uv manages the Python
environment, development tooling is included, and heavyweight ML dependencies
are deferred until a concrete experiment needs them.

## Setup

Install dependencies:

```bash
uv sync
```

Run the baseline checks:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
```

## Workflow

- Use `notebooks/` for exploratory analysis and interactive experiments.
- Use `experiments/` for reproducible experiment runs, configs, and notes.
- Use `scripts/` for repeatable command-line workflows that are not yet package APIs.
- Move shared, tested logic into `src/mech-interp/` once scripts or notebooks start repeating it.

Large model files, checkpoints, and generated outputs should stay out of Git.
