# This image is just for testing the installation.
# It builds much faster than the Apptainer image due to caching.

ARG ROCM_VERSION=5.2

FROM rocm/dev-ubuntu-20.04:${ROCM_VERSION}-complete as builder

ARG ROCM_VERSION
ARG PYTHON_VERSION=3.8
ARG PYTORCH_VERSION=1.13.1
ARG TORCHVISION_VERSION=0.14.1
ARG VISSL_VERSION=0.1.6

COPY setup.py requirements.txt vissl/ configs/ /opt/vissl

SHELL ["/bin/bash", "-c"]

ARG PATH=/usr/local/bin:/opt/conda/bin:${PATH}
ENV PATH=/usr/local/bin:/opt/conda/bin:${PATH}

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-venv \
      python${PYTHON_VERSION}-dev

# Install VISSL dependencies and VISSL in dev mode
WORKDIR /opt/vissl
RUN python${PYTHON_VERSION} -m venv /venv \
 && . /venv/bin/activate \
 && pip install --upgrade pip \
 && pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} --extra-index-url https://download.pytorch.org/whl/rocm${ROCM_VERSION} \
 && pip install --no-cache-dir -e .

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
