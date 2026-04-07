all:
	uv run pytest
	uv run ruff format
	uv run ruff check --fix

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