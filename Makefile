PYTHON ?= .venv/bin/python

.PHONY: setup data validate acquisition retention conversion revenue forecast lifecycle dashboard-kpis dashboard train-models monitor production serve docker-up docker-down results test test-fast lint

setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(PYTHON) -m northstar generate-data

validate:
	$(PYTHON) -m northstar validate-data

acquisition:
	$(PYTHON) -m northstar acquisition

retention:
	$(PYTHON) -m northstar retention

conversion:
	$(PYTHON) -m northstar conversion

revenue:
	$(PYTHON) -m northstar revenue

forecast:
	$(PYTHON) -m northstar forecast

lifecycle:
	$(PYTHON) -m northstar lifecycle

dashboard-kpis:
	$(PYTHON) -m northstar dashboard-kpis

dashboard:
	$(PYTHON) -m northstar dashboard

train-models:
	$(PYTHON) -m northstar train-models

monitor:
	$(PYTHON) -m northstar monitor

production: train-models monitor

serve:
	$(PYTHON) -m northstar serve

docker-up:
	docker compose up --build --wait

docker-down:
	docker compose down

results: data acquisition retention conversion revenue forecast lifecycle dashboard-kpis production

test:
	$(PYTHON) -m pytest

test-fast:
	$(PYTHON) -m pytest -m "not slow"

lint:
	$(PYTHON) -m ruff check .
