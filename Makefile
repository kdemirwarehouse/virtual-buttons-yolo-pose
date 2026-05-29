.PHONY: install run lint test clean

install:
	pip install -r requirements.txt

run:
	python src/virtual_buttons.py

lint:
	ruff check src/

test:
	pytest tests/ -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
