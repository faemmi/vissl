# This image is just for testing the installation.
# It builds much faster than the Apptainer image due to caching.
ARG ROCM_VERSION=5.2

FROM rocm/dev-ubuntu-20.04:${ROCM_VERSION}-complete as builder

ARG ROCM_VERSION
ARG PYTHON_VERSION=3.8
ARG PYTORCH_VERSION=1.13.1
ARG TORCHVISION_VERSION=0.14.1

COPY README.md/ /opt/vissl/
COPY pyproject.toml /opt/vissl/
COPY poetry.lock /opt/vissl/
COPY vissl/ /opt/vissl/vissl
COPY configs/ /opt/vissl/configs
COPY hydra_plugins/ /opt/vissl/hydra_plugins
COPY extra_scripts/ /opt/vissl/extra_scripts

SHELL ["/bin/bash", "-c"]

ARG PATH=/usr/local/bin:/opt/conda/bin:${PATH}
ENV PATH=/usr/local/bin:/opt/conda/bin:${PATH}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV POETRY_VIRTUALENVS_CREATE=false

COPY --from=vissl /venv /venv
COPY --from=vissl /opt/vissl /opt/vissl

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-venv \
      python${PYTHON_VERSION}-dev \
      curl

# Install poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python${PYTHON_VERSION} -
ENV PATH=/opt/poetry/bin:${PATH}

WORKDIR /opt/vissl
RUN python${PYTHON_VERSION} -m venv /venv \
 && . /venv/bin/activate \
 && pip install --upgrade pip \
 && poetry install --only=main,jupyter \
 # Install torch after vissl to overwrite package dependencies
 && pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --extra-index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION}

# Delete Python cache files
WORKDIR /venv
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

FROM rocm/dev-ubuntu-20.04:${ROCM_VERSION}

ARG PATH=/usr/local/bin:/venv/bin:${PATH}
ENV PATH=/usr/local/bin:/venv/bin:${PATH}

# VISSL conda env and repo
COPY --from=builder /venv /venv
COPY --from=builder /opt/vissl /opt/vissl

# Install opencv via apt to get required libraries
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      python${PYTORCH_VERSION} \
      python${PYTHON_VERSION}-dev \
      python3-opencv \
      git \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN which python \
 && python --version \
 && python -c 'import torch, vissl, cv2'

ENTRYPOINT ["python"]
