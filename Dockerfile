# syntax = docker/dockerfile
FROM python:3.10-slim-bookworm as python-base-image

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    SETUPTOOLS_USE_DISTUTILS=stdlib\
    work_dir=/function

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

ENV VIRTUAL_ENV=$work_dir/.venv \
    PATH="$work_dir/.venv/bin:$PATH"

RUN apt-get update && apt-get upgrade -y

# Install Pillow-SIMD system dependencies
RUN apt-get install --no-install-recommends -y libjpeg-dev zlib1g-dev gcc

# Install poetry
RUN apt-get install -y --no-install-recommends curl && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py && \
    curl -sSL https://install.python-poetry.org | python3 - && \
    apt-get purge -y curl && \
    rm -rf /var/lib/apt/lists/*

# Set up work directory
WORKDIR $work_dir

FROM python-base-image as build-env

# copy in code
COPY poetry.lock pyproject.toml README.md $work_dir/

RUN poetry install --only main --no-ansi --no-interaction --no-root

COPY diffusion_streaming_api $work_dir/diffusion_streaming_api

RUN poetry install --only main --no-ansi --no-interaction

FROM build-env as tests

RUN poetry install --without=test
COPY tests $work_dir/tests

CMD poetry run coverage run -m pytest ; coverage xml -o test_results/coverage.xml

FROM build-env as documentation

ENV SPHINX_APIDOC_OPTIONS=members

# Install documentation dependencies
RUN apt-get install --no-install-recommends -y make
RUN poetry install --no-ansi --no-interaction --no-dev

# Copy in documentation files
COPY docs $work_dir/docs

# Run Sphinx generation
RUN poetry run sphinx-apidoc --separate -f -o docs/source/ . test/*

# Build HTML docs
RUN cd docs && make html


FROM build-env as lambda

RUN pip install awslambdaric --target $work_dir

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

COPY new_checkpoint.pt $work_dir/new_checkpoint.pt

CMD ["run_diffusion_streaming_api"]
