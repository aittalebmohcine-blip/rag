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
	@echo "Available targets:"
	@echo "  install         Install project dependencies using uv"
	@echo "  run             Run the project entry point"
	@echo "  debug           Run the project under the Python debugger (pdb)"
	@echo "  index           Build the searchable index from data/raw/"
	@echo "  search          Search the index for a single query"
	@echo "  search_dataset  Search an entire dataset and save the results"
	@echo "  answer          Answer a single query using retrieved context"
	@echo "  answer_dataset  Generate answers for a search results dataset"
	@echo "  evaluate        Evaluate retrieval results against ground truth"
	@echo "  lint            Run flake8 and mypy checks"
	@echo "  clean           Remove Python cache and build artifacts"

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
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" \) -exec rm -rf {} +
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

lint:
	flake8 src
	mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs
