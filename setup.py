from setuptools import find_namespace_packages, setup


setup(
    name="Torch-MNF",
    version="0.1.0",
    author="Janosh Riebesell",
    author_email="janosh.riebesell@gmail.com",
    packages=find_namespace_packages(include=["torch_mnf*"]),
    url="https://github.com/janosh/torch-mnf",
    description="PyTorch implementation of Multiplicative Normalizing Flows",
    python_requires=">=3.8",
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "scikit-learn",
        "scipy",
        "seaborn",
        "torch",
        "torchvision",
    ],
    extras_require={"test": ["pytest", "pytest-cov"]},
)
