# Project Guidelines

## Workflow

- Use versioned Marimo notebooks under `notebooks/*.py` for exploration, ideation, and validation.
- Notebooks may remain exploratory. Move logic required by the application or pipelines into `src/`.
- Treat `src/` as the source of truth. Do not leave production logic implemented only in notebooks.
- Compose reusable functions from `src/` into scripts or Airflow DAGs.
- Pipelines must be deterministic and idempotent: the same inputs and configuration must produce the same logical outputs, and reruns must be safe.
- Do not duplicate business logic between notebooks, scripts, DAGs, and application code.
- Evaluate extraction to an external package only when there is concrete reuse outside this project. Do not create speculative packages.

## Data Processing

- Prefer Polars for dataframe operations.
- Load the `polars` skill whenever a task involves loading, querying, transforming, aggregating, joining, or analyzing tabular data, including pandas-to-Polars migrations.
- Use the Polars MCP tools to verify version-specific methods, signatures, and namespaces instead of relying on memory when the API is uncertain.
- Use pandas only at integration boundaries when required by third-party libraries.
- Keep Polars-to-pandas conversions close to those boundaries and avoid propagating pandas through internal code.

## Python

- Use `uv` for dependency management and command execution.
- Add dependencies with `uv add` and run project tools with `uv run`.
- Keep the lockfile versioned.
- All code promoted to `src/` must be fully typed and pass mypy in strict mode.
- Avoid `Any` except at unavoidable third-party integration boundaries; narrow it as soon as possible.
- Prefer small, focused functions and avoid abstractions without demonstrated reuse.

## Tests

- Code promoted from notebooks to `src/` must have tests for its observable behavior and relevant edge cases.
- Test contracts and outcomes rather than implementation details.
- Reusable pipeline components must be testable independently of Airflow.
- Run Ruff, mypy, and pytest before considering a change complete.

## Conventions

- Write code, identifiers, filenames, comments, and docstrings in English.
- Write project documentation and Jira content in Brazilian Portuguese.
- Use Conventional Commits.
- Never include submitted medical text in logs, metrics, exception details, experiment tracking, or test failure messages.
- Never commit credentials, tokens, private datasets, or generated secrets.
