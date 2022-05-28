FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /workspace

COPY pyproject.toml .
COPY poetry.lock .

RUN pip install poetry \
    && poetry config virtualenvs.create false \
    && poetry install

COPY configs configs
COPY deepportfolio deepportfolio
COPY tests tests

RUN pytest -v -s tests
