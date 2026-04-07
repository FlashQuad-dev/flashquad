all:
	uv run pytest
	uv run ruff format
	uv run ruff check --fix

docs:
	uv run --extra docs sphinx-build docs docs/_build/html

test:
	uv run pytest

test-cpu:
	uv run pytest --no-gpu

format:
	uv run ruff format

check:
	uv run ruff format --check
	uv run ruff check

fix:
	uv run ruff check --fix