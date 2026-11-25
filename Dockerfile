# syntax = docker/dockerfile
FROM python:3.10-slim-bookworm AS python-base-image

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    SETUPTOOLS_USE_DISTUTILS=stdlib\
    work_dir=/function

ENV UV_PROJECT_ENVIRONMENT=$work_dir/.venv \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1

ENV VIRTUAL_ENV=$work_dir/.venv \
    PATH="$work_dir/.venv/bin:$PATH"

RUN apt-get update && apt-get upgrade -y

# Install Pillow-SIMD system dependencies
RUN apt-get install --no-install-recommends -y libjpeg-dev zlib1g-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /uvx /bin/

# Set up work directory
WORKDIR $work_dir

FROM python-base-image AS build-env

# copy in code
COPY uv.lock pyproject.toml README.md $work_dir/

RUN uv sync --locked --no-default-groups --no-install-project

COPY diffusion_streaming_api $work_dir/diffusion_streaming_api

RUN uv sync --locked --no-default-groups --no-editable

FROM build-env AS tests

RUN uv sync --locked --no-default-groups --group test
COPY tests $work_dir/tests

CMD uv run --no-sync coverage run -m pytest ; uv run --no-sync coverage xml -o test_results/coverage.xml

FROM build-env AS documentation

ENV SPHINX_APIDOC_OPTIONS=members

# Install documentation dependencies
RUN apt-get install --no-install-recommends -y make
RUN uv sync --locked --no-default-groups --group documentation

# Copy in documentation files
COPY docs $work_dir/docs

# Run Sphinx generation
RUN uv run --no-sync sphinx-apidoc --separate -f -o docs/source/ . test/*

# Build HTML docs
RUN cd docs && make html


FROM python:3.10-slim-bookworm AS lambda
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PATH="/root/.local/bin:$PATH" \
    SETUPTOOLS_USE_DISTUTILS=stdlib\
    work_dir=/function
# Set up work directory
WORKDIR $work_dir

ENV VIRTUAL_ENV=$work_dir/.venv \
    PATH="$work_dir/.venv/bin:$PATH"

COPY --from=build-env $work_dir/.venv $work_dir/.venv
COPY --from=build-env $work_dir/diffusion_streaming_api $work_dir/diffusion_streaming_api

RUN pip install awslambdaric --target $work_dir

COPY --from=public.ecr.aws/awsguru/aws-lambda-adapter:0.8.4 /lambda-adapter /opt/extensions/lambda-adapter

COPY new_checkpoint.pt $work_dir/new_checkpoint.pt
COPY latent_checkpoint.pt $work_dir/latent_checkpoint.pt
COPY latent_unet.onnx $work_dir/latent_unet.onnx
COPY latent_autoencoder.pt $work_dir/latent_autoencoder.pt

CMD ["run_diffusion_streaming_api"]
