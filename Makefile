# Developer shortcuts for The Factory Floor.
#
# PYTHON defaults to whatever `python` is on PATH; override if the project env is
# elsewhere, e.g.  make test PYTHON=/opt/miniconda3/bin/python

PYTHON ?= python
NOTEBOOKS := $(sort $(wildcard notebooks/*.ipynb))

.PHONY: help test test-all test-llm lint fmt nb app

help:
	@echo "make test      - unit + integration tests (no API key, no cost)"
	@echo "make test-all  - also run llm-marked tests (needs OPENAI_API_KEY, spends money)"
	@echo "make test-llm  - only the llm-marked tests"
	@echo "make lint      - ruff check"
	@echo "make fmt       - ruff check --fix"
	@echo "make nb        - re-execute every notebook in place (regression check)"
	@echo "make app       - launch the Streamlit app"

test:
	$(PYTHON) -m pytest -m "not llm"

test-all:
	$(PYTHON) -m pytest

test-llm:
	$(PYTHON) -m pytest -m llm

lint:
	$(PYTHON) -m ruff check .

fmt:
	$(PYTHON) -m ruff check . --fix

nb:
	@for nb in $(NOTEBOOKS); do \
		echo ">>> $$nb"; \
		$(PYTHON) -m nbconvert --to notebook --execute --inplace "$$nb" || exit 1; \
	done

app:
	$(PYTHON) -m streamlit run app.py
