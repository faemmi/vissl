# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import pathlib

import pkg_resources
from setuptools import find_namespace_packages, find_packages, setup


def fetch_requirements(name):
    with pathlib.Path(name).open() as requirements_txt:
        install_requires = [
            str(requirement)
            for requirement in pkg_resources.parse_requirements(requirements_txt)
        ]
    return install_requires


def get_version():
    init_py_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "vissl", "__init__.py"
    )
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")
    return version


packages = find_packages(include=["vissl.*", "configs.*"]) + find_namespace_packages(
    include=["hydra_plugins.*"]
)


setup(
    name="vissl",
    version=get_version(),
    author="Facebook AI Research",
    author_email="vissl@fb.com",
    license="MIT",
    url="https://github.com/facebookresearch/vissl",
    description="VISSL is an extensible, modular and scalable library for "
    "SOTA Self-Supervised Learning with images.",
    packages=packages,
    install_requires=fetch_requirements("requirements.txt"),
    include_package_data=True,
    python_requires=">=3.6.2",
    extras_require={"dev": fetch_requirements("requirements-dev.txt")},
)
