.PHONY: test lint build clean

test:
	uv run --extra dev pytest tests/ -v

lint:
	uv run --extra dev ruff check src/ tests/
	uv run --extra dev ruff format --check src/ tests/
	uv run --extra dev pyright

build:
	uv build --out-dir dist/

clean:
	rm -rf dist/ .pytest_cache .ruff_cache
