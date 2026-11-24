.PHONY: test lint vuln build clean

test:
	uv run --extra dev pytest tests/ -v

lint:
	uv run --extra dev ruff check src/ tests/
	uv run --extra dev ruff format --check src/ tests/
	uv run --extra dev pyright

# vuln: known advisories in the locked dependencies; exits 1 on a finding.
# Run before tagging a release (CONVENTIONS §Release Checklist).
vuln:
	uv audit --frozen --preview-features audit-command

build:
	uv build --out-dir dist/

clean:
	rm -rf dist/ .pytest_cache .ruff_cache
