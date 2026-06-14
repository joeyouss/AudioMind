SHELL := /bin/bash

.PHONY: setup run

setup:
	./scripts/setup_local.sh

run:
	cd app && source .venv/bin/activate && streamlit run streamlit_app.py
