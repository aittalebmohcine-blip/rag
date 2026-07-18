.PHONY: install run debug clean

k ?= 2

install:
	uv sync

search:
	uv run python -m src search "$(q)" --k $(k)
index:
	uv run python -m src index

debug:
	uv run python -m pdb src

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .tox
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

lint:
	flake8 src
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
