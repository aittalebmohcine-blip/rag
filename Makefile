.PHONY: install run debug clean

k ?= 5
max_chunk_size ?= 2000
query ?=
save_directory ?=
SEARCH_SAVE_DIR ?= data/output/search_results/UnansweredQuestions
ANSWER_SAVE_DIR ?= data/output/search_results/AnsweredQuestions
dataset_path ?=
student_search_results_path ?=

install:
	uv sync

run:
	uv run python -m src index

index:
	uv run python -m src index \
		--max_chunk_size "$(max_chunk_size)"

search:
	uv run python -m src search \
		--query "$(query)" \
		--k "$(k)"

search_dataset:
	uv run python -m src search_dataset \
		--dataset_path "$(dataset_path)" \
		--k "$(k)" \
		--save_directory "$(or $(save_directory),$(SEARCH_SAVE_DIR))"

answer:
	uv run python -m src answer --query "$(query)" --k "$(k)"


answer_dataset:
	uv run python -m src answer_dataset \
		--student_search_results_path "$(student_search_results_path)" \
		--save_directory "$(or $(save_directory),$(ANSWER_SAVE_DIR))"

evaluate:
	uv run python -m src evaluate \
		--student_search_results_path "$(student_search_results_path)" \
		--dataset_path "$(dataset_path)"

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
