.PHONY: install run debug clean lint lint-strict index search search_dataset answer answer_dataset evaluate help

UV ?= uv
PYTHON ?= $(UV) run python
MAIN_MODULE ?= -m src

k ?= 5
max_chunk_size ?= 2000
query ?=
save_directory ?=
SEARCH_SAVE_DIR ?= data/output/search_results/UnansweredQuestions
ANSWER_SAVE_DIR ?= data/output/search_results/AnsweredQuestions
dataset_path ?=
student_search_results_path ?=

help:
	@printf "Available targets:\n"
	@printf "  install         Install project dependencies with uv\n"
	@printf "  run             Run the project entrypoint\n"
	@printf "  debug           Run the project entrypoint under pdb\n"
	@printf "  clean           Remove Python cache/build artifacts\n"
	@printf "  lint            Run flake8 and mypy with the requested options\n"

install:
	$(UV) sync

run:
	$(PYTHON) $(MAIN_MODULE)

debug:
	$(PYTHON) -m pdb -m src

index:
	$(PYTHON) -m src index \
		--max_chunk_size "$(max_chunk_size)"

search:
	$(PYTHON) -m src search \
		--query "$(query)" \
		--k "$(k)"

search_dataset:
	$(PYTHON) -m src search_dataset \
		--dataset_path "$(dataset_path)" \
		--k "$(k)" \
		--save_directory "$(or $(save_directory),$(SEARCH_SAVE_DIR))"

answer:
	$(PYTHON) -m src answer \
		--query "$(query)" \
		--k "$(k)"

answer_dataset:
	$(PYTHON) -m src answer_dataset \
		--student_search_results_path "$(student_search_results_path)" \
		--save_directory "$(or $(save_directory),$(ANSWER_SAVE_DIR))"

evaluate:
	$(PYTHON) -m src evaluate \
		--student_search_results_path "$(student_search_results_path)" \
		--dataset_path "$(dataset_path)"

clean:
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf .tox
	rm -rf build
	rm -rf dist
	rm -rf *.egg-info
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" -o -name ".mypy_cache" \) -prune -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

lint:
	flake8 src
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
