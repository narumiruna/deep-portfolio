install:
	poetry install

test:
	poetry run pytest -v -s tests

build-docker:
	docker build -f Dockerfile -t deepportfolio .
