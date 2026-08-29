install:
	uv sync

run:
	uv run python -m src --functions_definition data/input/functions_definition.json \
		--input data/input/function_calling_tests.json \
		--output data/output/function_calling_results.json

test:
	uv run -m pytest

debug:
	uv run -m pdb src/main.py

clean:
	rm -rf .pytest_cache __pycache__ .mypy_cache

lint:
	uv run -m flake8 . --exclude=.venv,llm_sdk
	uv run -m mypy . --warn-return-any --warn-unused-ignores \
	    --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs \
	    --exclude llm_sdk

lint-strict:
	uv run -m flake8 . --exclude=.venv,llm_sdk
	uv run -m mypy . --strict --exclude llm_sdk
