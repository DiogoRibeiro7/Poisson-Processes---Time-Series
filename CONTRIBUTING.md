# Contributing

Create branches from `main` and open pull requests for all changes.

Before opening a PR, run:

```bash
poetry run ruff check src/poisson_time_series tests
poetry run mypy src/poisson_time_series tests
poetry run pytest --cov --cov-report=term-missing
poetry run mkdocs build --strict
```

Public Python APIs must be typed and documented. Statistical assumptions should be explicit, and new numerical or probabilistic behaviour should include deterministic tests whenever possible.
