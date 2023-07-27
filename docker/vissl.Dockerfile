# This image is just for testing the installation.
# It builds much faster than the Apptainer image due to caching.

ARG CUDA_VERSION=11.1.1

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-devel-ubuntu18.04 as builder

ARG CUDA_VERSION
ARG PYTHON_VERSION=3.8
ARG PYTORCH_VERSION=1.9.1
ARG TORCHVISION_VERSION=0.10.1
ARG VISSL_VERSION=0.1.6
ARG PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/opt/conda/bin:${PATH}

ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/opt/conda/bin:${PATH}
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY README.md/ /opt/vissl/
COPY pyproject.toml /opt/vissl/
COPY poetry.lock /opt/vissl/
COPY vissl/ /opt/vissl/vissl
COPY configs/ /opt/vissl/configs
COPY hydra_plugins/ /opt/vissl/hydra_plugins
COPY extra_scripts/ /opt/vissl/extra_scripts

SHELL ["/bin/bash", "-c"]

# See https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN apt-key del 7fa2af80 \
 && apt-key del 3bf863cc \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      wget \
      curl
# Install conda (miniconda)
RUN wget \
      --quiet \
      -O miniconda.sh \
      https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && chmod +x miniconda.sh \
 && bash miniconda.sh -b -p /opt/conda

# Install poetry
RUN curl -sSL https://install.python-poetry.org | POETRY_HOME=/opt/poetry python -
ENV PATH=/opt/poetry/bin:${PATH}

# Create and activate conda environment
RUN conda config --add channels conda-forge \
 && conda install conda-pack \
 && conda update pip setuptools \
 && conda create --name vissl python=${PYTHON_VERSION}

# Install PyTorch and apex
RUN conda install -n vissl \
        -c pytorch -c conda-forge \
        pytorch=${PYTORCH_VERSION} \
        torchvision=${TORCHVISION_VERSION} \
        cudatoolkit=${CUDA_VERSION} \
 && conda install -n vissl -c vissl apex

# Use conda-pack to create a standalone env in /venv and install vissl
WORKDIR /opt/vissl
RUN conda-pack -n vissl -o /opt/env.tar.gz \
 && mkdir /venv \
 && tar -xzf /opt/env.tar.gz -C /venv \
 && . /venv/bin/activate \
 && conda-unpack

 # Must install vissl after unpacking since conda doesn't allow to pack
 # packages installed in editable mode.
RUN . /venv/bin/activate \
 && POETRY_VIRTUALENVS_CREATE=false poetry install --only=main,jupyter \
 # Use specific classy_vision version due to incompatibility with torchvision 0.10.1 due to
 # ImportError: cannot import name 'Kinetics' from 'torchvision.datasets.kinetics'
 && pip uninstall -y classy_vision \
 && pip install classy-vision@https://github.com/facebookresearch/ClassyVision/tarball/4785d5ee19d3bcedd5b28c1eb51ea1f59188b54d

# Delete Python cache files
WORKDIR /venv
RUN find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf

FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu18.04

ARG CUDA_VERSION

ARG PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/venv/bin:${PATH}
ENV PATH=/usr/local/cuda-${CUDA_VERSION}/bin:/usr/local/bin:/venv/bin:${PATH}

# VISSL conda env and repo
COPY --from=builder /venv /venv
COPY --from=builder /opt/vissl /opt/vissl

# See https://github.com/NVIDIA/nvidia-docker/issues/1631
RUN apt-key del 7fa2af80 \
 && apt-key del 3bf863cc \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
 && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install opencv via apt to get required libraries
RUN apt-get update \
 && DEBIAN_FRONTEND=noninteractive \
 && apt-get install -y --no-install-recommends python3-opencv \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN which python \
 && python --version \
 && pip list \
 && python -c 'import torch, apex, vissl, cv2'

ENTRYPOINT ["python"]
