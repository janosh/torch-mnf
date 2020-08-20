# Install this package via `pip install -e .`. Remember to
# activate correct target environment first. The `-e` option
# (development mode) sometimes installs to the wrong place.
# See https://github.com/conda/conda/issues/5861 and
# https://stackoverflow.com/a/49478207. Without the -e option,
# be advised that a non-editable install requires reinstallation
# after every edit for changes take effect.

from setuptools import find_namespace_packages, setup

setup(
    name="Torch-MNF",
    version="0.1.0",
    author="Janosh Riebesell",
    author_email="janosh.riebesell@gmail.com",
    packages=find_namespace_packages(include=["torch_mnf*"]),
    url="https://github.com/janosh/torch-mnf",
    description="PyTorch implementation of Multiplicative Normalizing Flows",
)
